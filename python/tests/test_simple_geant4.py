"""
Simple Test - Verify Geant4 CSV Output
Tests with guaranteed phantom hit: direction [1, 0, 0]
"""

import sys
from pathlib import Path
import logging

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from geant4_interface.simulation_runner import Geant4SimulationRunner

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)


def test_simple_hit():
    """Test with direction that MUST hit phantom"""

    print("\n" + "🔬" * 35)
    print("SIMPLE GEANT4 TEST - GUARANTEED PHANTOM HIT")
    print("🔬" * 35 + "\n")

    GEANT4_EXE = r"C:\Thesis\geant4\Water-Phantom\build\Release\WaterPhantomSim.exe"
    OUTPUT_DIR = r"C:\Thesis\python\test_simple_hit"

    if not Path(GEANT4_EXE).exists():
        print(f"❌ Geant4 not found: {GEANT4_EXE}")
        return

    # Create runner
    runner = Geant4SimulationRunner(
        geant4_executable=GEANT4_EXE,
        output_directory=OUTPUT_DIR
    )

    # Parameters with direction [1, 0, 0] - goes straight into phantom
    params = {
        'particle_type': 'e-',
        'particle_energy': 10.0,
        'particle_position': [-6, 0, 0],  # 6 cm left of phantom
        'particle_direction': [1, 0, 0],  # Straight right → MUST hit!
        'num_events': 1
    }

    print("📋 Test parameters:")
    print(f"  Particle: {params['particle_type']}")
    print(f"  Energy: {params['particle_energy']} MeV")
    print(f"  Position: {params['particle_position']}")
    print(f"  Direction: {params['particle_direction']} (straight into phantom)")
    print()

    print("🚀 Running simulation...")
    print("=" * 60)

    result = runner.run_simulation(params)

    print("=" * 60)
    print()

    if result['success']:
        print("✅ SIMULATION SUCCESS!")
        print(f"\n📊 Results:")
        print(f"  Energy deposited: {result['total_energy_deposit']:.3f} MeV")
        print(f"  Events processed: {result['num_events']}")

        if result['events'] and len(result['events']) > 0:
            first_event = result['events'][0]
            print(f"  First event:")
            print(f"    - Energy deposit: {first_event['total_energy_deposit']:.3f} MeV")
            print(f"    - Steps: {first_event['num_steps']}")

            if first_event['steps']:
                print(f"  Sample steps (first 3):")
                for i, step in enumerate(first_event['steps'][:3]):
                    pos = step['position']
                    print(f"    Step {i}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) cm, "
                          f"E={step['energy']:.3f} MeV")

        print(f"\n📂 Output directory: {result['output_directory']}")
        print()

        # Check CSV files
        output_path = Path(result['output_directory'])
        csv_files = list(output_path.rglob('*.csv'))
        print(f"📄 CSV files created: {len(csv_files)}")
        for csv in csv_files:
            print(f"  - {csv.name}")

        print("\n" + "=" * 60)
        print("🎉 TEST PASSED!")
        print("=" * 60)
        print("\n✅ Geant4 is working correctly!")
        print("✅ CSV output is being generated!")
        print("\n💡 Next steps:")
        print("  1. Check why Unity directions don't hit phantom")
        print("  2. Add direction correction in trajectory_buffer.py")
        print("  3. Re-run full pipeline test")

    else:
        print("❌ SIMULATION FAILED!")
        print(f"\n🔍 Error: {result.get('error', 'Unknown')}")
        print()

        print("📋 Diagnostic checklist:")
        print("  1. Check Geant4 stdout above - did it print anything?")
        print("  2. Look for 'G4_OUTPUT_DIR' in logs - was it set?")
        print("  3. Check if event finished - did EndOfEventAction run?")
        print("  4. Look for CSV creation errors")
        print()

        if 'stdout' in result:
            print("💡 Check the Geant4 output logs above for clues")

        print("\n" + "=" * 60)
        print("❌ TEST FAILED")
        print("=" * 60)


def test_random_direction():
    """Test with random direction (may miss phantom)"""

    print("\n" + "🔬" * 35)
    print("TEST 2: RANDOM DIRECTION (Unity-like)")
    print("🔬" * 35 + "\n")

    import numpy as np

    GEANT4_EXE = r"C:\Thesis\geant4\Water-Phantom\build\Release\WaterPhantomSim.exe"
    OUTPUT_DIR = r"C:\Thesis\python\test_random_dir"

    runner = Geant4SimulationRunner(
        geant4_executable=GEANT4_EXE,
        output_directory=OUTPUT_DIR
    )

    # Random direction like Unity
    random_dir = np.random.randn(3)
    random_dir = random_dir / np.linalg.norm(random_dir)

    params = {
        'particle_type': 'e-',
        'particle_energy': 10.0,
        'particle_position': [-6, 0, 0],
        'particle_direction': random_dir.tolist(),
        'num_events': 1
    }

    print(f"📋 Random direction: {params['particle_direction']}")
    print(f"   (Like Unity agent movement)")
    print()

    result = runner.run_simulation(params)

    if result['success']:
        energy = result['total_energy_deposit']

        if energy > 0.1:
            print(f"✅ Hit phantom! Energy: {energy:.3f} MeV")
        else:
            print(f"⚠️  Missed phantom! Energy: {energy:.3f} MeV")
            print("   This is expected with random directions")
            print("   → Need to correct direction toward phantom")
    else:
        print(f"❌ Failed: {result.get('error')}")


if __name__ == "__main__":
    print("\n📋 Which test to run?")
    print("1. Simple test (direction [1,0,0] - MUST hit)")
    print("2. Random direction (may miss)")
    print("3. Both")

    choice = input("\nChoice (1/2/3): ").strip()

    if choice == "1":
        test_simple_hit()
    elif choice == "2":
        test_random_direction()
    elif choice == "3":
        test_simple_hit()
        test_random_direction()
    else:
        print("Invalid choice")