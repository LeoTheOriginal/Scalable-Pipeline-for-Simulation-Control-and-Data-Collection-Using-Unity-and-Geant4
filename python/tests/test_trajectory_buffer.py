"""
Test Trajectory Buffer System
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_training.trajectory_data import ParticleTrajectory, ParticleStep
from rl_training.trajectory_buffer import TrajectoryBuffer


def create_dummy_unity_trajectory(traj_id: int) -> ParticleTrajectory:
    """Create dummy Unity trajectory for testing"""

    # Random initial conditions
    energy = np.random.uniform(5.0, 15.0)
    position = np.array([-6.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])

    trajectory = ParticleTrajectory(
        trajectory_id=traj_id,
        agent_id=traj_id % 16,  # Simulate 16 agents
        initial_energy=energy,
        initial_position=position,
        initial_direction=direction,
        source='unity'
    )

    # Simulate some steps
    num_steps = np.random.randint(10, 50)
    current_pos = position.copy()
    current_energy = energy

    for i in range(num_steps):
        # Move forward
        current_pos += direction * 0.1  # 0.1 cm steps

        # Lose energy
        energy_loss = 0.05 + np.random.uniform(0, 0.1)
        current_energy -= energy_loss

        if current_energy < 0:
            current_energy = 0

        step = ParticleStep(
            step_number=i,
            position=current_pos.copy(),
            direction=direction.copy(),
            energy=current_energy,
            energy_deposit=energy_loss,
            step_length=0.1
        )

        trajectory.add_step(step)

        if current_energy < 0.01:
            break

    trajectory.completed = True
    trajectory.exit_reason = "energy_depleted"

    return trajectory


def test_trajectory_buffer():
    """Test trajectory buffer with dummy data"""

    print("\n" + "🧪" * 35)
    print("TRAJECTORY BUFFER TEST")
    print("🧪" * 35 + "\n")

    # Configuration
    GEANT4_EXE = r"C:\Thesis\geant4\Water-Phantom\build\Release\WaterPhantomSim.exe"
    BUFFER_SIZE = 20  # Small for testing
    NUM_WORKERS = 4

    # Check Geant4
    if not Path(GEANT4_EXE).exists():
        print(f"❌ Geant4 not found: {GEANT4_EXE}")
        return

    print("Configuration:")
    print(f"  Buffer size: {BUFFER_SIZE}")
    print(f"  Workers: {NUM_WORKERS}")
    print()

    # Create buffer
    buffer = TrajectoryBuffer(
        geant4_executable=GEANT4_EXE,
        buffer_size=BUFFER_SIZE,
        num_workers=NUM_WORKERS,
        auto_process=False  # Manual processing for demo
    )

    # Generate dummy trajectories
    print(f"Generating {BUFFER_SIZE} dummy Unity trajectories...")
    for i in range(BUFFER_SIZE):
        traj = create_dummy_unity_trajectory(i)
        buffer.add_unity_trajectory(traj)
        print(f"  Added trajectory {i}: "
              f"Energy={traj.initial_energy:.2f} MeV, "
              f"Steps={traj.num_steps}")

    # Show statistics
    stats = buffer.get_statistics()
    print(f"\n📊 Buffer statistics:")
    print(f"  Buffered: {stats['buffer_size']}")
    print(f"  Collected: {stats['total_collected']}")
    print(f"  Processed: {stats['total_processed']}")

    # Process buffer
    print(f"\n🔄 Processing buffer...")
    pairs = buffer.process_buffer()

    # Show results
    print(f"\n✅ Processing complete!")
    print(f"   Created {len(pairs)} trajectory pairs")
    print()

    # Analyze pairs
    print("📊 Trajectory Pair Analysis:")
    print("=" * 70)

    for i, pair in enumerate(pairs[:5]):  # Show first 5
        print(f"\nPair {i}:")
        print(f"  Unity:  {pair.unity_trajectory.num_steps} steps, "
              f"Energy deposited: {pair.unity_trajectory.total_energy_deposited:.3f} MeV")
        print(f"  Geant4: {pair.geant4_trajectory.num_steps} steps, "
              f"Energy deposited: {pair.geant4_trajectory.total_energy_deposited:.3f} MeV")
        print(f"  Distance: {pair.position_distance:.3f} cm")
        print(f"  Reward: {pair.reward:.3f}")

    # Overall statistics
    distances = [p.position_distance for p in pairs]
    rewards = [p.reward for p in pairs]

    print("\n" + "=" * 70)
    print("OVERALL STATISTICS:")
    print("=" * 70)
    print(f"Average distance: {np.mean(distances):.3f} ± {np.std(distances):.3f} cm")
    print(f"Average reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"Best reward: {np.max(rewards):.3f}")
    print(f"Worst reward: {np.min(rewards):.3f}")
    print("=" * 70)

    print("\n✅ TEST COMPLETE!")


if __name__ == "__main__":
    test_trajectory_buffer()