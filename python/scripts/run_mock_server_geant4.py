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

# Importy ścieżek: Jesteśmy w python/scripts/
# Musimy wyjść do python/, żeby widzieć 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.shared_types import MAX_STEPS

# --- IMPORT PRAWDZIWEGO GEANT4 ---
try:
    # Importujemy bezpośrednio z miejsca docelowego
    from src.simulation import geant4_sim

    print("✅ Geant4 module loaded successfully from src.simulation")
except ImportError as e:
    print(f"❌ CRITICAL: Could not load Geant4 module. Error: {e}")
    print("Ensure you have built the C++ project with the new CMake configuration.")
    sys.exit(1)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RealGeant4Generator:
    def __init__(self):
        print("Initializing Geant4 Simulation Manager...")
        self.sim_manager = geant4_sim.SimulationManager()
        print("Geant4 Ready!")

    def generate_batch(self, count=50):
        batch_x = []
        batch_y = []
        batch_z = []

        for _ in range(count):
            result = self.sim_manager.run_single()

            steps = len(result['x'])

            rx = result['x']
            ry = result['y']
            rz = result['z']

            # Padding (wyrównanie do MAX_STEPS)
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

        return {
            'x': np.array(batch_x),
            'y': np.array(batch_y),
            'z': np.array(batch_z)
        }


generator = RealGeant4Generator()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Unity connected!")

    try:
        while True:
            start_time = time.perf_counter()

            # Mniejszy batch na początek, żeby nie zamrozić komputera fizyką
            BATCH_SIZE = 20

            batch_data = generator.generate_batch(count=BATCH_SIZE)

            raw_points = np.stack([
                batch_data['x'],
                batch_data['y'],
                batch_data['z']
            ], axis=2).flatten().astype(np.float32)

            packed = msgpack.packb({
                'count': BATCH_SIZE,
                'steps': MAX_STEPS,
                'data': raw_points.tobytes()
            })

            compressed = lz4.block.compress(packed, store_size=False)

            await websocket.send_bytes(compressed)

            process_time = time.perf_counter() - start_time

            # Celujemy w ~30 FPS, ale Geant4 może być wolniejszy
            await asyncio.sleep(max(0, 0.033 - process_time))

    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        print("Unity disconnected")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)