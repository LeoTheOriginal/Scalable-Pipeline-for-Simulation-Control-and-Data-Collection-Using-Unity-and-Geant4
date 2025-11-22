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
        real_x, real_y, real_z = [], [], []
        ai_x, ai_y, ai_z = [], [], []

        generated_count = 0

        # Pętla Retry (tak samo ważna jak w start_simulation!)
        while generated_count < count:

            # A. Geant4 Run
            raw = self.sim_manager.run_single()
            steps = len(raw['x'])

            if steps < 2: continue  # Pomiń puste cząstki

            rx, ry, rz = raw['x'], raw['y'], raw['z']
            px, py, pz = raw['px'], raw['py'], raw['pz']
            energy = raw['energy']

            # B. AI Simulation
            # Pobieramy start z Geant4
            current_state = np.array([
                rx[0], ry[0], rz[0],
                px[0], py[0], pz[0],
                energy[0]
            ], dtype=np.float32)

            # Listy na AI (inicjalizacja)
            path_ai_x = [current_state[0]]
            path_ai_y = [current_state[1]]
            path_ai_z = [current_state[2]]

            for i in range(steps - 1):
                action, _ = model.predict(current_state, deterministic=True)
                current_state += action

                path_ai_x.append(current_state[0])
                path_ai_y.append(current_state[1])
                path_ai_z.append(current_state[2])

            # C. Padding (Dopychanie zerami - tak samo jak w start_simulation)
            pad_rx = np.zeros(MAX_STEPS, dtype=np.float32)
            pad_ry = np.zeros(MAX_STEPS, dtype=np.float32)
            pad_rz = np.zeros(MAX_STEPS, dtype=np.float32)

            limit = min(steps, MAX_STEPS)
            pad_rx[:limit] = rx[:limit]
            pad_ry[:limit] = ry[:limit]
            pad_rz[:limit] = rz[:limit]

            pad_ax = np.zeros(MAX_STEPS, dtype=np.float32)
            pad_ay = np.zeros(MAX_STEPS, dtype=np.float32)
            pad_az = np.zeros(MAX_STEPS, dtype=np.float32)

            ai_limit = min(len(path_ai_x), MAX_STEPS)
            pad_ax[:ai_limit] = path_ai_x[:ai_limit]
            pad_ay[:ai_limit] = path_ai_y[:ai_limit]
            pad_az[:ai_limit] = path_ai_z[:ai_limit]

            real_x.append(pad_rx);
            real_y.append(pad_ry);
            real_z.append(pad_rz)
            ai_x.append(pad_ax);
            ai_y.append(pad_ay);
            ai_z.append(pad_az)

            generated_count += 1

        # Zwracamy słownik, żeby łatwo skleić w endpointcie
        return {
            'real_x': np.array(real_x), 'real_y': np.array(real_y), 'real_z': np.array(real_z),
            'ai_x': np.array(ai_x), 'ai_y': np.array(ai_y), 'ai_z': np.array(ai_z)
        }


generator = InferenceGenerator()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[Network] Unity Connected (Inference Mode)")

    try:
        while True:
            start_time = time.perf_counter()

            # Zmniejsz batch dla bezpieczeństwa na start
            BATCH_SIZE = 20

            data = generator.generate_comparison_batch(BATCH_SIZE)

            # === TUTAJ ROBIMY TO ANALOGICZNIE DO START_SIMULATION ===

            # 1. Łączymy Real i AI w jeden ciąg
            combined_x = np.concatenate([data['real_x'], data['ai_x']])
            combined_y = np.concatenate([data['real_y'], data['ai_y']])
            combined_z = np.concatenate([data['real_z'], data['ai_z']])

            # 2. Tworzymy stos (Stack)
            # Pamiętamy o minusie przy Z dla Unity!
            raw_points = np.stack([
                combined_x,
                combined_y,
                -combined_z
            ], axis=2)

            # 3. CRITICAL FIX: "Leczenie" danych z AI
            # Zamieniamy NaN i Infinity na 0.0, żeby Unity się nie wywaliło
            if not np.isfinite(raw_points).all():
                print("⚠️ WARNING: Wykryto błędy (NaN/Inf) w danych AI! Naprawiam...")
                raw_points = np.nan_to_num(raw_points, nan=0.0, posinf=0.0, neginf=0.0)

            # 4. Spłaszczanie i rzutowanie
            flat_data = raw_points.flatten().astype(np.float32)

            # 5. Pakowanie
            packed = msgpack.packb({
                'count': BATCH_SIZE * 2,  # 20 Real + 20 AI
                'steps': MAX_STEPS,
                'data': flat_data.tobytes()
            })

            compressed = lz4.block.compress(packed, store_size=False)
            await websocket.send_bytes(compressed)

            process_time = time.perf_counter() - start_time
            await asyncio.sleep(max(0, 0.05 - process_time))

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Network] Error: {e}")
    finally:
        print("[Network] Disconnected")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)