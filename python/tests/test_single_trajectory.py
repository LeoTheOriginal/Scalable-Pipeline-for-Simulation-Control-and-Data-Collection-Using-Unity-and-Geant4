"""
Test Single Trajectory: Unity → Geant4 → Reward
Simple end-to-end test of the pipeline
"""

import sys
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from unity_integration.unity_connector import UnityConnector
from rl_training.trajectory_buffer import TrajectoryBuffer
from rl_training.trajectory_data import ParticleTrajectory


def test_single_trajectory():
    """
    Test complete pipeline:
    1. Unity agent generates trajectory
    2. Python collects observations
    3. Send to Geant4 via buffer
    4. Compare and calculate reward
    5. Display results
    """

    print("\n" + "🧪" * 35)
    print("SINGLE TRAJECTORY PIPELINE TEST")
    print("🧪" * 35 + "\n")

    print("📋 SETUP:")
    print("   1. Unity: Play mode with 1 agent (bounds DISABLED!)")
    print("   2. Geant4: WaterPhantomSim.exe ready")
    print("   3. Press Enter when ready...")
    input()

    # ========================================
    # STEP 1: Connect to Unity
    # ========================================
    print("\n📡 STEP 1: Connecting to Unity...")
    connector = UnityConnector()

    if not connector.connect():
        print("❌ Failed to connect to Unity!")
        return

    print(f"✅ Connected! {connector.num_agents} agents found")

    # ========================================
    # STEP 2: Collect ONE trajectory
    # ========================================
    print("\n📊 STEP 2: Collecting trajectory from Unity...")
    print("   (This will take ~30-60 seconds until agent depletes energy)")

    observations = []
    max_steps = 5000

    print("Collecting steps (agent will move until energy depletes)...")

    for step in range(max_steps):
        # Get observations
        obs = connector.get_observations()

        if len(obs) == 0:
            print(f"   Episode ended at step {step}")
            break

        # Record first agent's observation
        observations.append(obs[0].copy())

        # Send small random actions (gentle movement)
        actions = np.random.uniform(-0.05, 0.05, size=(len(obs), 3))  # Very small movements!
        connector.send_actions(actions)

        if step % 100 == 0:
            energy_normalized = obs[0][6]
            print(f"   Step {step}: collecting... (normalized energy={energy_normalized:.3f})")

        time.sleep(0.005)  # Small delay

    if len(observations) < 10:
        print(f"\n⚠️  WARNING: Only {len(observations)} steps collected!")
        print("   Agent episode ended too quickly. Did you disable bounds check?")
        connector.close()
        return

    print(f"\n✅ Collected {len(observations)} steps")

    # ========================================
    # STEP 3: Convert to ParticleTrajectory
    # ========================================
    print("\n🔄 STEP 3: Converting to ParticleTrajectory...")

    # Debug: print first observation
    print(f"   First observation: {observations[0]}")
    print(f"   Observation shape: (10 values)")
    print(f"   Relative position (0-2): {observations[0][0:3]}")
    print(f"   Velocity (3-5): {observations[0][3:6]}")
    print(f"   Normalized energy (6): {observations[0][6]}")
    print(f"   Direction (7-9): {observations[0][7:10]}")

    # Unity agent parameters
    start_position = np.array([-6.0, 0.0, 0.0])  # Agent starts at (-6, 0, 0)
    initial_energy_mev = 10.0  # Assume 10 MeV (middle of 5-15 range)

    # In real system, we would get this from Unity directly
    # For now, we use reasonable defaults

    trajectory = ParticleTrajectory.from_unity_observation(
        trajectory_id=0,
        agent_id=0,
        observations=observations,
        start_position=start_position,
        initial_energy_mev=initial_energy_mev
    )

    print(f"\n✅ Trajectory created:")
    print(f"   Initial energy: {trajectory.initial_energy:.2f} MeV")
    print(f"   Initial position: {trajectory.initial_position}")
    print(f"   Initial direction: {trajectory.initial_direction}")
    print(f"   Steps: {trajectory.num_steps}")
    print(f"   Final position: {trajectory.final_position}")
    print(f"   Final energy: {trajectory.final_energy:.2f} MeV")
    print(f"   Total energy deposited: {trajectory.total_energy_deposited:.3f} MeV")

    # ========================================
    # STEP 4: Send to Geant4 via Buffer
    # ========================================
    print("\n⚛️  STEP 4: Running Geant4 simulation...")

    # Path to your Geant4 executable - CORRECTED PATH!
    geant4_exe = r"C:\Thesis\geant4\Water-Phantom\build\Release\WaterPhantomSim.exe"

    # Check if executable exists
    if not Path(geant4_exe).exists():
        print(f"❌ Geant4 executable not found at: {geant4_exe}")
        print("   Please check the path and try again.")
        connector.close()
        return

    print(f"   Geant4 exe: {geant4_exe}")

    # Create buffer (with 1 worker for simplicity)
    buffer = TrajectoryBuffer(
        geant4_executable=geant4_exe,
        buffer_size=1,  # Process immediately
        num_workers=1,
        auto_process=False  # Manual control
    )

    # Add trajectory
    buffer.add_unity_trajectory(trajectory)

    # Process (runs Geant4)
    print("   Running Geant4 with same initial conditions...")
    print("   (This may take 10-30 seconds...)")

    try:
        pairs = buffer.process_buffer()
    except Exception as e:
        print(f"❌ Geant4 processing failed with error: {e}")
        import traceback
        traceback.print_exc()
        connector.close()
        return

    if not pairs or len(pairs) == 0:
        print("❌ Geant4 processing returned no results!")
        connector.close()
        return

    pair = pairs[0]

    print(f"\n✅ Geant4 simulation complete!")
    print(f"   Energy deposited: {pair.geant4_trajectory.total_energy_deposited:.3f} MeV")
    print(f"   Steps: {pair.geant4_trajectory.num_steps}")

    # ========================================
    # STEP 5: Compare & Calculate Reward
    # ========================================
    print("\n📊 STEP 5: Comparing trajectories...")

    print(f"\n📈 COMPARISON METRICS:")
    print(f"   Position distance: {pair.position_distance:.3f} cm")
    print(f"   Energy difference: {pair.energy_difference:.3f} MeV")
    print(f"   Step count ratio: {pair.step_count_ratio:.2f}")
    print(f"\n🎁 REWARD: {pair.reward:.3f}")

    # ========================================
    # STEP 6: Visualize Results
    # ========================================
    print("\n📊 STEP 6: Trajectory Summary...")

    unity_pos = trajectory.get_positions_array()
    geant4_pos = pair.geant4_trajectory.get_positions_array()

    print(f"\nUnity trajectory:")
    print(f"   Start: {unity_pos[0]}")
    print(f"   End: {unity_pos[-1]}")
    if len(unity_pos) > 1:
        distance = np.sum([np.linalg.norm(unity_pos[i + 1] - unity_pos[i]) for i in range(len(unity_pos) - 1)])
        print(f"   Distance traveled: {distance:.2f} cm")

    if len(geant4_pos) == 0:
        print(f"\n⚠️  WARNING: Geant4 returned no trajectory data!")
        print("   This usually means:")
        print("   - Geant4 simulation failed")
        print("   - CSV output wasn't generated")
        print("   - Output parser couldn't read files")
        print("\n   Check Geant4 logs in: ./trajectory_buffer_geant4/sim_000000/")
        print("   Look at event_0.csv to see what Geant4 produced")
    else:
        print(f"\nGeant4 trajectory:")
        print(f"   Start: {geant4_pos[0]}")
        print(f"   End: {geant4_pos[-1]}")
        if len(geant4_pos) > 1:
            distance = np.sum([np.linalg.norm(geant4_pos[i + 1] - geant4_pos[i]) for i in range(len(geant4_pos) - 1)])
            print(f"   Distance traveled: {distance:.2f} cm")

    # ========================================
    # Cleanup
    # ========================================
    connector.close()

    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE!")
    print("=" * 70)

    if pair.reward > -1.0 and len(geant4_pos) > 0:
        print("\n🎉 SUCCESS! Pipeline is working!")
        print("\nNEXT STEPS:")
        print("   1. Scale to multiple agents (16)")
        print("   2. Implement full training loop")
        print("   3. Train RL agent to match Geant4!")
    else:
        print("\n⚠️  Reward is low or missing Geant4 data")
        print("\nNEXT STEPS:")
        print("   1. Check Geant4 CSV output: ./trajectory_buffer_geant4/sim_000000/event_0.csv")
        print("   2. Verify SteppingAction in Geant4 is writing step data")
        print("   3. If CSV is empty → fix Geant4 output")
        print("   4. If CSV has data → check output_parser.py")


if __name__ == "__main__":
    test_single_trajectory()