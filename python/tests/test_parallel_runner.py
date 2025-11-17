"""
Test Parallel Geant4 Runner
"""

import sys
from pathlib import Path
import numpy as np
import time


sys.path.insert(0, str(Path(__file__).parent.parent))

from geant4_interface.parallel_runner import ParallelGeant4Runner, BatchStatistics


def test_parallel_batch():
    """Test parallel batch execution"""

    print("\n" + "🔬" * 35)
    print("PARALLEL GEANT4 RUNNER TEST")
    print("🔬" * 35 + "\n")

    # Configuration
    GEANT4_EXE = r"C:\Thesis\geant4\Water-Phantom\build\Release\WaterPhantomSim.exe"
    OUTPUT_DIR = r"C:\Thesis\python\test_parallel_output"
    NUM_SIMULATIONS = 50  # Test with 50 simulations
    NUM_WORKERS = 8  # Use 8 parallel workers

    # Check if Geant4 exists
    if not Path(GEANT4_EXE).exists():
        print(f"❌ Geant4 executable not found: {GEANT4_EXE}")
        return

    print(f"Configuration:")
    print(f"  Geant4 exe: {GEANT4_EXE}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Simulations: {NUM_SIMULATIONS}")
    print(f"  Workers: {NUM_WORKERS}")
    print()

    # Create parallel runner
    runner = ParallelGeant4Runner(
        geant4_executable=GEANT4_EXE,
        output_directory=OUTPUT_DIR,
        num_workers=NUM_WORKERS
    )

    # Generate random parameters
    print("Generating parameters...")
    parameters_list = []

    for i in range(NUM_SIMULATIONS):
        energy = np.random.uniform(1.0, 20.0)  # Random 1-20 MeV

        params = {
            'particle_type': 'e-',
            'particle_energy': energy,
            'particle_position': [-6, 0, 0],
            'particle_direction': [1, 0, 0],
            'num_events': 1
        }

        parameters_list.append(params)

    print(f"✅ Generated {len(parameters_list)} parameter sets")
    print()

    # Run batch
    print("=" * 70)
    print("RUNNING PARALLEL BATCH")
    print("=" * 70)

    start_time = time.time()
    results = runner.run_batch(parameters_list, show_progress=True)
    elapsed_time = time.time() - start_time

    print()
    print("=" * 70)
    print("BATCH COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Time per simulation: {elapsed_time / NUM_SIMULATIONS:.2f}s")
    print(f"Estimated speedup: {NUM_SIMULATIONS * 2 / elapsed_time:.1f}x")
    print()

    # Statistics
    stats = BatchStatistics()
    for result in results:
        stats.add_result(result)

    stats.print_summary()

    # Show some sample results
    print("\n📊 Sample results:")
    for i, result in enumerate(results[:5]):
        if result['success']:
            print(f"  Sim {i}: Energy={result['parameters']['particle_energy']:.2f} MeV, "
                  f"Deposited={result['total_energy_deposit']:.2f} MeV")

    print("\n✅ TEST COMPLETE!")


def test_streaming_batch():
    """Test streaming batch execution"""

    print("\n" + "🔬" * 35)
    print("STREAMING BATCH TEST")
    print("🔬" * 35 + "\n")

    # Configuration
    GEANT4_EXE = r"C:\Thesis\geant4\Water-Phantom\build\Release\WaterPhantomSim.exe"
    OUTPUT_DIR = r"C:\Thesis\python\test_streaming_output"
    NUM_SIMULATIONS = 20

    # Create runner
    runner = ParallelGeant4Runner(
        geant4_executable=GEANT4_EXE,
        output_directory=OUTPUT_DIR,
        num_workers=4
    )

    # Generate parameters
    parameters_list = []
    for i in range(NUM_SIMULATIONS):
        energy = np.random.uniform(5.0, 15.0)
        parameters_list.append({
            'particle_type': 'e-',
            'particle_energy': energy,
            'particle_position': [-6, 0, 0],
            'particle_direction': [1, 0, 0],
            'num_events': 1
        })

    print(f"Running streaming batch: {NUM_SIMULATIONS} simulations")
    print("Results will appear as they complete...\n")

    # Process results as they complete
    completed = 0
    for result in runner.run_streaming_batch(parameters_list):
        completed += 1

        if result['success']:
            energy_in = result['parameters']['particle_energy']
            energy_out = result['total_energy_deposit']

            print(f"✅ [{completed}/{NUM_SIMULATIONS}] "
                  f"Energy: {energy_in:.2f} → {energy_out:.2f} MeV "
                  f"({energy_out / energy_in * 100:.1f}%)")
        else:
            print(f"❌ [{completed}/{NUM_SIMULATIONS}] Failed: {result.get('error')}")

    print("\n✅ STREAMING TEST COMPLETE!")


if __name__ == "__main__":
    print("\nSelect test:")
    print("1. Parallel batch (50 simulations)")
    print("2. Streaming batch (20 simulations)")
    print("3. Both")

    choice = input("\nChoice (1/2/3): ").strip()

    if choice == "1":
        test_parallel_batch()
    elif choice == "2":
        test_streaming_batch()
    elif choice == "3":
        test_parallel_batch()
        test_streaming_batch()
    else:
        print("Invalid choice")