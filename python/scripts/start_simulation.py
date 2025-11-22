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

# ============================================================================
# 1. KONFIGURACJA ŚCIEŻEK (DLL + MODUŁY)
# ============================================================================

# A. Dodajemy DLL-ki Geant4 (BEZ TEGO NIE RUSZY!)
path_to_geant4_bin = r"C:\Geant4\install\bin"  # <--- TU SPRAWDŹ CZY ŚCIEŻKA JEST OK
if os.name == 'nt' and os.path.exists(path_to_geant4_bin):
    os.add_dll_directory(path_to_geant4_bin)
    print(f"[System] Added DLL directory: {path_to_geant4_bin}")
else:
    print(f"[System] ⚠️ WARNING: DLL path not found: {path_to_geant4_bin}")

# B. Dodajemy ścieżkę do 'src' projektu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.shared_types import MAX_STEPS

# C. Importujemy moduł C++
try:
    from src.simulation import geant4_sim

    print("[System] ✅ Geant4 physics engine loaded successfully!")
except ImportError as e:
    print(f"[System] ❌ CRITICAL ERROR: Could not load Geant4 module.\n{e}")
    sys.exit(1)

# ============================================================================
# 2. SERWER FASTAPI
# ============================================================================

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Geant4SimulationServer:
    def __init__(self):
        print("[Sim] Initializing Geant4 Manager...")
        self.sim_manager = geant4_sim.SimulationManager()
        print("[Sim] Physics Engine Ready.")

    def generate_batch(self, count=20):
        """Generuje batch prawdziwych trajektorii fizycznych"""
        batch_x, batch_y, batch_z = [], [], []

        # --- DEBUG LICZNIK ---
        non_zero_particles = 0
        total_steps_collected = 0
        # ---------------------

        for i in range(count):
            # 1. Strzał fizyczny (C++)
            result = self.sim_manager.run_single()

            # 2. Pobranie danych
            rx, ry, rz = result['x'], result['y'], result['z']
            steps = len(rx)

            # --- DEBUGOWANIE PIERWSZEJ CZĄSTKI W BATCHU ---
            if i == 0:
                print(f"[DEBUG] Particle 0 steps: {steps}")
                if steps > 0:
                    print(f"[DEBUG] Part 0 Start: ({rx[0]:.2f}, {ry[0]:.2f}, {rz[0]:.2f})")
                    print(f"[DEBUG] Part 0 End:   ({rx[-1]:.2f}, {ry[-1]:.2f}, {rz[-1]:.2f})")
            # ---------------------------------------------

            if steps > 0:
                non_zero_particles += 1
                total_steps_collected += steps

            # 3. Normalizacja do Unity (Padding zerami do MAX_STEPS)
            padded_x = np.zeros(MAX_STEPS, dtype=np.float32)
            padded_y = np.zeros(MAX_STEPS, dtype=np.float32)
            padded_z = np.zeros(MAX_STEPS, dtype=np.float32)

            limit = min(steps, MAX_STEPS)
            if limit > 0:
                padded_x[:limit] = rx[:limit]
                padded_y[:limit] = ry[:limit]
                padded_z[:limit] = rz[:limit]

            batch_x.append(padded_x)
            batch_y.append(padded_y)
            batch_z.append(padded_z)

        # --- RAPORT BATCHA ---
        if non_zero_particles == 0:
            print(f"⚠️ OSTRZEŻENIE: Cały batch {count} cząstek jest PUSTY (0 kroków)!")
        # ---------------------

        return {
            'x': np.array(batch_x),
            'y': np.array(batch_y),
            'z': np.array(batch_z)
        }


sim_server = Geant4SimulationServer()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[Network] Unity Client Connected")

    try:
        while True:
            start_time = time.perf_counter()

            # Generujemy 20-50 cząstek na klatkę (zależy od CPU)
            BATCH_SIZE = 50

            # Generowanie fizyki
            batch_data = sim_server.generate_batch(count=BATCH_SIZE)

            # Formatowanie danych dla Unity (płaska tablica)
            raw_points = np.stack([
                batch_data['x'],
                batch_data['y'],
                -batch_data['z']
            ], axis=2).flatten().astype(np.float32)

            # Kompresja
            packed = msgpack.packb({
                'count': BATCH_SIZE,
                'steps': MAX_STEPS,
                'data': raw_points.tobytes()
            })

            # store_size=False jest krytyczne dla Twojego kodu C#
            compressed = lz4.block.compress(packed, store_size=False)

            # Wysyłka
            await websocket.send_bytes(compressed)

            # Kontrola FPS (celujemy w ~30Hz wysyłania)
            process_time = time.perf_counter() - start_time
            await asyncio.sleep(max(0, 0.033 - process_time))

    except Exception as e:
        print(f"[Network] Connection closed: {e}")
    finally:
        print("[Network] Unity Disconnected")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)