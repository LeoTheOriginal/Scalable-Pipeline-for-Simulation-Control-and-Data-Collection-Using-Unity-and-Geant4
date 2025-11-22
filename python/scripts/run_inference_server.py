import sys
import os
import asyncio
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import msgpack
import lz4.block
import time
from stable_baselines3 import PPO

# 1. Konfiguracja ścieżek
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from src.core.shared_types import MAX_STEPS

# 2. Konfiguracja DLL i Geant4
path_to_geant4_bin = r"C:\Geant4\install\bin"
if os.name == 'nt' and os.path.exists(path_to_geant4_bin):
    os.add_dll_directory(path_to_geant4_bin)

try:
    from src.simulation import geant4_sim

    print("[System] ✅ Geant4 loaded.")
except ImportError:
    print("[System] ❌ Geant4 import failed.")
    sys.exit(1)

# 3. Ładowanie Modelu AI
MODEL_PATH = os.path.join(project_root, "data", "models", "ppo_geant4", "geant4_final")
print(f"[AI] Loading model from {MODEL_PATH}...")
try:
    model = PPO.load(MODEL_PATH)
    print("[AI] ✅ Model loaded successfully!")
except Exception as e:
    print(f"[AI] ❌ Could not load model: {e}")
    sys.exit(1)

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class InferenceGenerator:
    def __init__(self):
        self.sim_manager = geant4_sim.SimulationManager()

    def generate_comparison_batch(self, count=10):
        """
        Generuje dane porównawcze: Prawda vs AI.
        """
        # Bufory na dane prawdziwe (Real)
        real_x, real_y, real_z = [], [], []
        # Bufory na dane przewidziane (AI)
        ai_x, ai_y, ai_z = [], [], []

        for _ in range(count):
            # A. Uruchom Geant4 (Prawda)
            raw = self.sim_manager.run_single()
            steps = len(raw['x'])

            # Pobieramy surowe wektory
            rx, ry, rz = raw['x'], raw['y'], raw['z']
            px, py, pz = raw['px'], raw['py'], raw['pz']
            energy = raw['energy']

            # B. Symulacja AI (Krok po kroku)
            # AI startuje tam gdzie prawdziwa cząstka
            current_ai_pos = np.array([rx[0], ry[0], rz[0]], dtype=np.float32)

            # Listy na ścieżkę jednej cząstki AI
            path_ai_x = [current_ai_pos[0]]
            path_ai_y = [current_ai_pos[1]]
            path_ai_z = [current_ai_pos[2]]

            # Pętla predykcji
            # Uwaga: AI używa 'Teacher Forcing' dla pędu/energii (bo jeszcze nie umie ich przewidywać)
            # Ale pozycję aktualizujemy sami!
            for i in range(steps - 1):
                # Budujemy obserwację: [x, y, z, px, py, pz, e]
                # Używamy POZYCJI z AI, ale PĘDU z Geant4 (hybryda na start)
                obs = np.array([
                    current_ai_pos[0], current_ai_pos[1], current_ai_pos[2],
                    px[i], py[i], pz[i], energy[i]
                ], dtype=np.float32)

                # Zapytaj model o akcję (deterministycznie = bez losowości)
                action, _ = model.predict(obs, deterministic=True)

                # Aktualizuj pozycję AI
                # Action to [dx, dy, dz]
                current_ai_pos += action

                path_ai_x.append(current_ai_pos[0])
                path_ai_y.append(current_ai_pos[1])
                path_ai_z.append(current_ai_pos[2])

            # C. Padding (Wyrównanie do MAX_STEPS)
            # --- REAL ---
            pad_rx = np.zeros(MAX_STEPS, dtype=np.float32)
            pad_ry = np.zeros(MAX_STEPS, dtype=np.float32)
            pad_rz = np.zeros(MAX_STEPS, dtype=np.float32)

            limit = min(steps, MAX_STEPS)
            pad_rx[:limit] = rx[:limit]
            pad_ry[:limit] = ry[:limit]
            pad_rz[:limit] = rz[:limit]

            real_x.append(pad_rx)
            real_y.append(pad_ry)
            real_z.append(pad_rz)

            # --- AI ---
            pad_ax = np.zeros(MAX_STEPS, dtype=np.float32)
            pad_ay = np.zeros(MAX_STEPS, dtype=np.float32)
            pad_az = np.zeros(MAX_STEPS, dtype=np.float32)

            ai_steps = len(path_ai_x)
            limit_ai = min(ai_steps, MAX_STEPS)
            pad_ax[:limit_ai] = path_ai_x[:limit_ai]
            pad_ay[:limit_ai] = path_ai_y[:limit_ai]
            pad_az[:limit_ai] = path_ai_z[:limit_ai]

            ai_x.append(pad_ax)
            ai_y.append(pad_ay)
            ai_z.append(pad_az)

        return {
            'real': {'x': np.array(real_x), 'y': np.array(real_y), 'z': np.array(real_z)},
            'ai': {'x': np.array(ai_x), 'y': np.array(ai_y), 'z': np.array(ai_z)}
        }


generator = InferenceGenerator()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[Network] Unity Connected (Inference Mode)")

    try:
        while True:
            start_time = time.perf_counter()

            # Generujemy mniej cząstek, bo teraz robimy 2x więcej pracy
            BATCH_SIZE = 20
            data = generator.generate_comparison_batch(BATCH_SIZE)

            # Pakujemy DWA zestawy danych
            # Format: [Real_Particles..., AI_Particles...]
            # Sklejamy je w jedną długą tablicę, żeby wysłać jedną paczką
            # Pierwsze 20 to Real, kolejne 20 to AI.

            combined_x = np.concatenate([data['real']['x'], data['ai']['x']])
            combined_y = np.concatenate([data['real']['y'], data['ai']['y']])
            combined_z = np.concatenate([data['real']['z'], data['ai']['z']])

            raw_points = np.stack([combined_x, combined_y, -combined_z], axis=2).flatten().astype(np.float32)

            # Wysyłamy 2x więcej cząstek niż BATCH_SIZE
            packed = msgpack.packb({
                'count': BATCH_SIZE * 2,
                'steps': MAX_STEPS,
                'data': raw_points.tobytes()
            })

            compressed = lz4.block.compress(packed, store_size=False)
            await websocket.send_bytes(compressed)

            process_time = time.perf_counter() - start_time
            await asyncio.sleep(max(0, 0.05 - process_time))  # 20 FPS

    except Exception as e:
        print(f"[Network] Error: {e}")
    finally:
        print("[Network] Disconnected")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)