"""
debug_inference.py - Diagnose AI predictions
"""

import sys
import os
import numpy as np
from stable_baselines3 import PPO

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# DLL path
path_to_geant4_bin = r"C:\Geant4\install\bin"
if os.name == 'nt' and os.path.exists(path_to_geant4_bin):
    os.add_dll_directory(path_to_geant4_bin)

from src.simulation import geant4_sim

# Load model
MODEL_PATH = os.path.join(project_root, "data", "models", "ppo_geant4_advanced", "best", "best_model")
print(f"Loading model: {MODEL_PATH}")
model = PPO.load(MODEL_PATH)


# Observation builder (simplified)
class SimpleObsBuilder:
    BEAM_START = np.array([-6.0, 0.0, 0.0])

    @staticmethod
    def calculate_range(energy_mev: float) -> float:
        if energy_mev < 0.1:
            return 0.0
        return 0.412 * (energy_mev ** 1.265)

    @classmethod
    def build_observation(cls, state_7d: np.ndarray, previous_delta: np.ndarray) -> np.ndarray:
        pos = state_7d[:3]
        momentum = state_7d[3:6]
        energy = state_7d[6]

        p_mag = np.linalg.norm(momentum)
        direction = momentum / p_mag if p_mag > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float32)

        depth = np.clip(pos[0] - cls.BEAM_START[0], 0, 10.0)
        beam_axis = np.array([1.0, 0.0, 0.0])
        angle = np.arccos(np.clip(np.dot(direction, beam_axis), -1.0, 1.0))
        velocity = np.linalg.norm(previous_delta)
        remaining_range = cls.calculate_range(energy)

        obs = np.array([
            pos[0], pos[1], pos[2],
            direction[0], direction[1], direction[2],
            energy, depth, angle, velocity, remaining_range,
            previous_delta[0], previous_delta[1], previous_delta[2]
        ], dtype=np.float32)

        # Normalize
        normalized = obs.copy()
        normalized[0:3] /= 10.0
        normalized[6] /= 10.0
        normalized[7] /= 10.0
        normalized[8] /= np.pi
        normalized[9] /= 1.0
        normalized[10] /= 5.0
        normalized[11:14] /= 1.0

        return np.clip(normalized, -3.0, 3.0)


def analyze_single_trajectory():
    """Analyze one trajectory in detail"""

    print("\n" + "=" * 80)
    print("TRAJECTORY ANALYSIS")
    print("=" * 80 + "\n")

    # Generate Geant4 ground truth
    manager = geant4_sim.SimulationManager()

    raw = None
    while raw is None or len(raw['x']) < 30:
        raw = manager.run_single()

    gt_steps = len(raw['x'])
    print(f"[Geant4] Generated trajectory: {gt_steps} steps")
    print(f"  Start: ({raw['x'][0]:.2f}, {raw['y'][0]:.2f}, {raw['z'][0]:.2f})")
    print(f"  End:   ({raw['x'][-1]:.2f}, {raw['y'][-1]:.2f}, {raw['z'][-1]:.2f})")
    print(f"  Energy: {raw['energy'][0]:.2f} → {raw['energy'][-1]:.2f} MeV")
    print(f"  Penetration: {raw['x'][-1] - raw['x'][0]:.2f} cm\n")

    # AI simulation
    obs_builder = SimpleObsBuilder()

    current_state = np.array([
        raw['x'][0], raw['y'][0], raw['z'][0],
        raw['px'][0], raw['py'][0], raw['pz'][0],
        raw['energy'][0]
    ], dtype=np.float32)

    ai_trajectory = [current_state.copy()]
    previous_delta = np.zeros(3, dtype=np.float32)

    print("[AI] Simulating trajectory (max 200 steps)...")

    for step in range(200):
        # Build observation
        obs = obs_builder.build_observation(current_state, previous_delta)

        # Predict action
        action, _ = model.predict(obs, deterministic=True)

        # Apply action
        current_state += action
        previous_delta = action[:3]

        ai_trajectory.append(current_state.copy())

        # Check stopping conditions
        energy = current_state[6]
        depth = current_state[0] - (-6.0)

        # CRITICAL: Should stop if energy too low or too deep
        if energy < 0.5:  # Below 0.5 MeV
            print(f"  [Stop] Step {step}: Energy depleted ({energy:.2f} MeV)")
            break

        if depth > 15.0:  # Too far
            print(f"  [Stop] Step {step}: Exceeded phantom ({depth:.2f} cm)")
            break

        if step % 50 == 0:
            print(f"  Step {step}: pos=({current_state[0]:.2f}, {current_state[1]:.2f}, {current_state[2]:.2f}), "
                  f"E={energy:.2f} MeV")

    ai_steps = len(ai_trajectory)
    final_state = ai_trajectory[-1]

    print(f"\n[AI] Trajectory finished: {ai_steps} steps")
    print(f"  Start: ({ai_trajectory[0][0]:.2f}, {ai_trajectory[0][1]:.2f}, {ai_trajectory[0][2]:.2f})")
    print(f"  End:   ({final_state[0]:.2f}, {final_state[1]:.2f}, {final_state[2]:.2f})")
    print(f"  Energy: {ai_trajectory[0][6]:.2f} → {final_state[6]:.2f} MeV")
    print(f"  Penetration: {final_state[0] - ai_trajectory[0][0]:.2f} cm")

    # Analysis
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    print(f"\nStep count:")
    print(f"  Geant4: {gt_steps}")
    print(f"  AI:     {ai_steps}")
    print(f"  Ratio:  {ai_steps / gt_steps:.2f}x")

    gt_penetration = raw['x'][-1] - raw['x'][0]
    ai_penetration = final_state[0] - ai_trajectory[0][0]

    print(f"\nPenetration:")
    print(f"  Geant4: {gt_penetration:.2f} cm")
    print(f"  AI:     {ai_penetration:.2f} cm")
    print(f"  Ratio:  {ai_penetration / gt_penetration:.2f}x")

    # Lateral spread
    gt_lateral = np.sqrt(raw['y'][-1] ** 2 + raw['z'][-1] ** 2)
    ai_lateral = np.sqrt(final_state[1] ** 2 + final_state[2] ** 2)

    print(f"\nLateral spread (Y²+Z²)^0.5:")
    print(f"  Geant4: {gt_lateral:.2f} cm")
    print(f"  AI:     {ai_lateral:.2f} cm")

    if ai_lateral < 0.1:
        print(f"  ⚠️ WARNING: AI has almost NO scattering!")

    # Energy loss
    gt_energy_loss = raw['energy'][0] - raw['energy'][-1]
    ai_energy_loss = ai_trajectory[0][6] - final_state[6]

    print(f"\nEnergy loss:")
    print(f"  Geant4: {gt_energy_loss:.2f} MeV")
    print(f"  AI:     {ai_energy_loss:.2f} MeV")

    # Action statistics
    print(f"\n" + "=" * 80)
    print("ACTION STATISTICS (First 50 steps)")
    print("=" * 80)

    obs_builder2 = SimpleObsBuilder()
    state = np.array([
        raw['x'][0], raw['y'][0], raw['z'][0],
        raw['px'][0], raw['py'][0], raw['pz'][0],
        raw['energy'][0]
    ], dtype=np.float32)
    prev_delta = np.zeros(3, dtype=np.float32)

    actions = []
    for i in range(min(50, ai_steps)):
        obs = obs_builder2.build_observation(state, prev_delta)
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action)
        state += action
        prev_delta = action[:3]

    actions = np.array(actions)

    print(f"\nPosition deltas (dx, dy, dz):")
    print(f"  Mean:   [{actions[:, 0].mean():.4f}, {actions[:, 1].mean():.4f}, {actions[:, 2].mean():.4f}]")
    print(f"  Std:    [{actions[:, 0].std():.4f}, {actions[:, 1].std():.4f}, {actions[:, 2].std():.4f}]")

    if actions[:, 1].std() < 0.01 and actions[:, 2].std() < 0.01:
        print(f"  ⚠️ WARNING: Almost no variation in Y/Z - AI not learning scattering!")

    print(f"\nEnergy deltas (dE):")
    print(f"  Mean:   {actions[:, 6].mean():.4f} MeV")
    print(f"  Std:    {actions[:, 6].std():.4f} MeV")

    if actions[:, 6].mean() > -0.05:
        print(f"  ⚠️ WARNING: AI not losing enough energy!")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    analyze_single_trajectory()