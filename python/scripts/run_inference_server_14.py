"""
inference_server.py - AI vs Geant4 Comparison Server (14-dim compatible)

UPDATED to match the advanced physics-informed environment with 14-dimensional observations.
Generates real-time comparison of AI predictions vs Geant4 ground truth.

Author: Dawid (Warsaw University of Technology)
"""

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

# ============================================================================
# 1. PATH CONFIGURATION
# ============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from src.core.shared_types import MAX_STEPS

# 2. Geant4 DLL Configuration
path_to_geant4_bin = r"C:\Geant4\install\bin"
if os.name == 'nt' and os.path.exists(path_to_geant4_bin):
    os.add_dll_directory(path_to_geant4_bin)

try:
    from src.simulation import geant4_sim
    print("[System] ✅ Geant4 loaded.")
except ImportError as e:
    print(f"[System] ❌ Geant4 import failed: {e}")
    sys.exit(1)

# ============================================================================
# 3. LOAD TRAINED MODEL
# ============================================================================
MODEL_PATH = os.path.join(project_root, "data", "models", "ppo_geant4_advanced", "best", "best_model")
print(f"[AI] Loading model from {MODEL_PATH}...")

try:
    model = PPO.load(MODEL_PATH)
    print("[AI] ✅ Model loaded successfully!")
except Exception as e:
    print(f"[AI] ❌ Could not load model: {e}")
    print("[AI] Trying alternative path...")

    # Fallback to final model
    MODEL_PATH_ALT = os.path.join(project_root, "data", "models", "ppo_geant4_advanced", "final_model")
    try:
        model = PPO.load(MODEL_PATH_ALT)
        print("[AI] ✅ Model loaded from alternative path!")
    except Exception as e2:
        print(f"[AI] ❌ Could not load model from alternative path either: {e2}")
        sys.exit(1)

# ============================================================================
# 4. FASTAPI APP
# ============================================================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================================
# 5. PHYSICS CONSTANTS (Match environment.py)
# ============================================================================

class PhysicsConstants:
    """Physical constants - must match environment.py"""
    ELECTRON_MASS = 0.511  # MeV/c²
    BEAM_START = np.array([-6.0, 0.0, 0.0])
    PHANTOM_SIZE = 10.0  # cm


# ============================================================================
# 6. OBSERVATION BUILDER (Match environment.py EXACTLY)
# ============================================================================

class ObservationBuilder:
    """
    Builds 14-dimensional observations for the trained model.

    CRITICAL: This MUST match environment.py's _build_observation() method exactly!
    """

    @staticmethod
    def calculate_range(energy_mev: float) -> float:
        """Empirical range formula for electrons in water"""
        if energy_mev < 0.1:
            return 0.0
        return 0.412 * (energy_mev ** 1.265)

    @staticmethod
    def build_observation(state_7d: np.ndarray, previous_delta: np.ndarray) -> np.ndarray:
        """
        Build 14-dimensional observation from 7D state.

        Args:
            state_7d: [x, y, z, px, py, pz, energy]
            previous_delta: [dx, dy, dz] from last step

        Returns:
            observation: 14-dimensional array (NORMALIZED)
        """
        pos = state_7d[:3]
        momentum = state_7d[3:6]
        energy = state_7d[6]

        # 1. Momentum direction (normalized)
        p_mag = np.linalg.norm(momentum)
        if p_mag > 1e-6:
            direction = momentum / p_mag
        else:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # 2. Depth in phantom
        depth = pos[0] - PhysicsConstants.BEAM_START[0]
        depth = np.clip(depth, 0, PhysicsConstants.PHANTOM_SIZE)

        # 3. Angle relative to beam axis
        beam_axis = np.array([1.0, 0.0, 0.0])
        angle = np.arccos(np.clip(np.dot(direction, beam_axis), -1.0, 1.0))

        # 4. Velocity magnitude
        velocity = np.linalg.norm(previous_delta)

        # 5. Remaining range
        remaining_range = ObservationBuilder.calculate_range(energy)

        # Assemble 14-dim observation
        obs = np.array([
            pos[0], pos[1], pos[2],                          # [0:3] Position
            direction[0], direction[1], direction[2],        # [3:6] Direction
            energy,                                          # [6] Energy
            depth,                                           # [7] Depth
            angle,                                           # [8] Angle
            velocity,                                        # [9] Velocity
            remaining_range,                                 # [10] Range
            previous_delta[0], previous_delta[1], previous_delta[2]  # [11:14] Previous delta
        ], dtype=np.float32)

        # NORMALIZE (CRITICAL - must match environment.py!)
        normalized = obs.copy()
        normalized[0:3] /= 10.0      # Position
        # normalized[3:6] already normalized (direction)
        normalized[6] /= 10.0         # Energy
        normalized[7] /= 10.0         # Depth
        normalized[8] /= np.pi        # Angle
        normalized[9] /= 1.0          # Velocity
        normalized[10] /= 5.0         # Range
        normalized[11:14] /= 1.0      # Previous delta

        return np.clip(normalized, -3.0, 3.0)


# ============================================================================
# 7. INFERENCE GENERATOR
# ============================================================================

class InferenceGenerator:
    """
    Generates AI predictions and compares with Geant4 ground truth.
    Uses 14-dimensional observations for model inference.
    """

    def __init__(self):
        self.sim_manager = geant4_sim.SimulationManager()
        self.obs_builder = ObservationBuilder()
        print("[Generator] ✅ Inference generator initialized (14-dim mode)")

    def generate_comparison_batch(self, count: int = 10):
        """
        Generate batch of AI vs Geant4 trajectory comparisons.

        Args:
            count: Number of trajectory pairs to generate

        Returns:
            Dictionary with real and AI trajectories (positions only for viz)
        """
        real_x, real_y, real_z = [], [], []
        ai_x, ai_y, ai_z = [], [], []

        generated_count = 0

        while generated_count < count:
            # ================================================================
            # A. GEANT4 GROUND TRUTH
            # ================================================================
            raw = self.sim_manager.run_single()
            steps = len(raw['x'])

            if steps < 2:
                continue  # Skip empty trajectories

            rx, ry, rz = raw['x'], raw['y'], raw['z']
            px, py, pz = raw['px'], raw['py'], raw['pz']
            energy = raw['energy']

            # ================================================================
            # B. AI PREDICTION (14-DIM OBSERVATIONS)
            # ================================================================
            # Initial 7D state
            current_state_7d = np.array([
                rx[0], ry[0], rz[0],
                px[0], py[0], pz[0],
                energy[0]
            ], dtype=np.float32)

            # Track AI trajectory (positions only)
            path_ai_x = [current_state_7d[0]]
            path_ai_y = [current_state_7d[1]]
            path_ai_z = [current_state_7d[2]]

            # Previous delta (initially zero)
            previous_delta = np.zeros(3, dtype=np.float32)

            # Simulate AI trajectory
            MAX_AI_STEPS = min(steps * 3, 150)  # At most 3x Geant4 or 150

            for i in range(MAX_AI_STEPS):
                # ============================================================
                # CRITICAL: Build 14-dimensional observation
                # ============================================================
                obs_14d = self.obs_builder.build_observation(
                    current_state_7d,
                    previous_delta
                )

                # Model predicts action (7D delta)
                action, _ = model.predict(obs_14d, deterministic=True)

                # Apply action to 7D state
                current_state_7d += action

                # Update previous delta
                previous_delta = action[:3]

                # ========================================================
                # EARLY STOPPING CONDITIONS
                # ========================================================
                pos = current_state_7d[:3]
                energy_current = current_state_7d[6]
                depth = pos[0] - PhysicsConstants.BEAM_START[0]

                # Stop if energy depleted
                if energy_current < 0.3:
                    break

                # Stop if exceeded phantom
                if depth > 12.0:
                    break

                # Stop if out of lateral bounds
                if abs(pos[1]) > 8.0 or abs(pos[2]) > 8.0:
                    break

                # Stop if negative depth
                if depth < -1.0:
                    break
                # ========================================================

                # Store position for visualization
                path_ai_x.append(current_state_7d[0])
                path_ai_y.append(current_state_7d[1])
                path_ai_z.append(current_state_7d[2])

            # ================================================================
            # C. PADDING TO MAX_STEPS
            # ================================================================
            # Ground truth
            pad_rx = np.zeros(MAX_STEPS, dtype=np.float32)
            pad_ry = np.zeros(MAX_STEPS, dtype=np.float32)
            pad_rz = np.zeros(MAX_STEPS, dtype=np.float32)

            limit = min(steps, MAX_STEPS)
            pad_rx[:limit] = rx[:limit]
            pad_ry[:limit] = ry[:limit]
            pad_rz[:limit] = rz[:limit]

            # AI prediction
            pad_ax = np.zeros(MAX_STEPS, dtype=np.float32)
            pad_ay = np.zeros(MAX_STEPS, dtype=np.float32)
            pad_az = np.zeros(MAX_STEPS, dtype=np.float32)

            ai_limit = min(len(path_ai_x), MAX_STEPS)
            pad_ax[:ai_limit] = path_ai_x[:ai_limit]
            pad_ay[:ai_limit] = path_ai_y[:ai_limit]
            pad_az[:ai_limit] = path_ai_z[:ai_limit]

            # Append to batch
            real_x.append(pad_rx)
            real_y.append(pad_ry)
            real_z.append(pad_rz)
            ai_x.append(pad_ax)
            ai_y.append(pad_ay)
            ai_z.append(pad_az)

            generated_count += 1

        return {
            'real_x': np.array(real_x),
            'real_y': np.array(real_y),
            'real_z': np.array(real_z),
            'ai_x': np.array(ai_x),
            'ai_y': np.array(ai_y),
            'ai_z': np.array(ai_z)
        }


generator = InferenceGenerator()


# ============================================================================
# 8. WEBSOCKET ENDPOINT
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time AI vs Geant4 comparison.
    Sends trajectory data to Unity for visualization.
    """
    await websocket.accept()
    print("[Network] Unity Connected (Inference Mode - 14-dim)")

    try:
        while True:
            start_time = time.perf_counter()

            # Generate comparison batch
            BATCH_SIZE = 20
            data = generator.generate_comparison_batch(BATCH_SIZE)

            # ================================================================
            # COMBINE REAL AND AI TRAJECTORIES
            # ================================================================
            # Concatenate: [Real_1, ..., Real_N, AI_1, ..., AI_N]
            combined_x = np.concatenate([data['real_x'], data['ai_x']])
            combined_y = np.concatenate([data['real_y'], data['ai_y']])
            combined_z = np.concatenate([data['real_z'], data['ai_z']])

            # Stack into [count, steps, 3] array
            # Unity uses -Z convention
            raw_points = np.stack([
                combined_x,
                combined_y,
                -combined_z
            ], axis=2)

            # ================================================================
            # DATA VALIDATION
            # ================================================================
            if not np.isfinite(raw_points).all():
                print("⚠️ WARNING: Detected NaN/Inf - sanitizing...")
                raw_points = np.nan_to_num(
                    raw_points,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                )

            # ================================================================
            # COMPRESSION AND TRANSMISSION
            # ================================================================
            flat_data = raw_points.flatten().astype(np.float32)

            packed = msgpack.packb({
                'count': BATCH_SIZE * 2,
                'steps': MAX_STEPS,
                'data': flat_data.tobytes()
            })

            compressed = lz4.block.compress(packed, store_size=False)
            await websocket.send_bytes(compressed)

            # Frame rate control (~20 FPS)
            process_time = time.perf_counter() - start_time
            await asyncio.sleep(max(0, 0.05 - process_time))

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Network] Error: {e}")
    finally:
        print("[Network] Unity Disconnected")


# ============================================================================
# 9. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("INFERENCE SERVER - AI vs GEANT4 COMPARISON (14-DIM MODE)")
    print("="*80)
    print(f"Model: {MODEL_PATH}")
    print(f"Observation space: 14 dimensions")
    print(f"Listening on: ws://0.0.0.0:8000/ws")
    print("="*80 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)