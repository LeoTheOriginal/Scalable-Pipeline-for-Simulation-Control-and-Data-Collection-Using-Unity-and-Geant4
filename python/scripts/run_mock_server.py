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

# --- FIX IMPORTÓW DLA STRUKTURY PROJEKTU ---
# Dodajemy katalog główny projektu do sys.path, aby widzieć pakiet 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Teraz importujemy z src.core
from src.core.shared_types import MAX_TRAJECTORIES, MAX_STEPS, POINT_DTYPE

# -------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class MockGeant4Generator:
    def __init__(self):
        self.running = False
        print(f"Initialising Mock Generator...")

    def generate_batch(self, count=100):
        trajectories = np.zeros((count, MAX_STEPS), dtype=POINT_DTYPE)
        t = np.linspace(0, 10, MAX_STEPS)

        for i in range(count):
            spread = np.random.uniform(-0.5, 0.5)
            scale = 1.0 + np.random.uniform(-0.1, 0.1)

            trajectories[i]['x'] = (t * scale) - 5.0
            trajectories[i]['y'] = np.sin(t + spread) * scale
            trajectories[i]['z'] = np.cos(t + spread) * scale * 0.5
            trajectories[i]['energy'] = np.linspace(10, 0, MAX_STEPS)

        return trajectories


generator = MockGeant4Generator()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Unity connected!")

    try:
        while True:
            start_time = time.perf_counter()

            # Generujemy 500 trajektorii (zgodnie z celem testu)
            batch_data = generator.generate_batch(count=500)

            # Spłaszczanie do formatu Unity [x,y,z, x,y,z...]
            raw_points = np.stack([
                batch_data['x'],
                batch_data['y'],
                batch_data['z']
            ], axis=2).flatten().astype(np.float32)

            packed = msgpack.packb({
                'count': 500,
                'steps': MAX_STEPS,
                'data': raw_points.tobytes()
            })

            compressed = lz4.block.compress(packed, store_size=False)

            await websocket.send_bytes(compressed)

            process_time = time.perf_counter() - start_time
            await asyncio.sleep(max(0, 0.033 - process_time))  # ~30 FPS

    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        print("Unity disconnected")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)