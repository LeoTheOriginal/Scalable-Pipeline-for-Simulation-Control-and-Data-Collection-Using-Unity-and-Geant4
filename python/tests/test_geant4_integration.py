"""
Test Geant4 Integration
Complete test of Geant4 simulation runner and parser
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# NOW use absolute imports
from geant4_interface.simulation_runner import Geant4SimulationRunner
from geant4_interface.output_parser import Geant4OutputParser
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)


def test_single_simulation():
    """Test single Geant4 simulation with default parameters"""

    print("=" * 60)
    print("Test 1: Single Simulation (10 MeV electron)")
    print("=" * 60)

    # Initialize runner
    runner = Geant4SimulationRunner(
        geant4_executable=r"C:\Thesis\geant4\Water-Phantom\build\Release\WaterPhantomSim.exe",
        output_directory=r"C:\Thesis\python\test_geant4_output"
    )

    # Test parameters
    parameters = {
        'particle_type': 'e-',
        'particle_energy': 10.0,
        'particle_position': [-6, 0, 0],
        'particle_direction': [1, 0, 0],
        'num_events': 1
    }

    print("\nRunning simulation with parameters:")
    for key, value in parameters.items():
        print(f"  {key}: {value}")

    # Run simulation
    result = runner.run_simulation(parameters)

    if result['success']:
        print("\n✅ Simulation successful!")
        print(f"   Total energy deposited: {result['total_energy_deposit']:.3f} MeV")
        print(f"   Number of events: {result['num_events']}")
        print(f"   Output directory: {result['output_directory']}")

        # Show first event details
        if result['events']:
            first_event = result['events'][0]
            print(f"\n📊 First event details:")
            print(f"   Event ID: {first_event['event_id']}")
            print(f"   Energy deposit: {first_event['total_energy_deposit']:.3f} MeV")
            print(f"   Number of steps: {first_event['num_steps']}")
    else:
        print(f"\n❌ Simulation failed: {result.get('error', 'Unknown error')}")
        return False

    print("=" * 60)
    return True


def test_multiple_energies():
    """Test simulations with different energies"""

    print("\n" + "=" * 60)
    print("Test 2: Multiple Energies (1, 5, 10, 20 MeV)")
    print("=" * 60)

    runner = Geant4SimulationRunner(
        geant4_executable=r"C:\Thesis\geant4\Water-Phantom\build\Release\WaterPhantomSim.exe",
        output_directory=r"C:\Thesis\python\test_geant4_output"
    )

    energies = [1.0, 5.0, 10.0, 20.0]
    results = []

    for energy in energies:
        print(f"\n⚡ Testing {energy} MeV...")

        parameters = {
            'particle_type': 'e-',
            'particle_energy': energy,
            'particle_position': [-6, 0, 0],
            'particle_direction': [1, 0, 0],
            'num_events': 1
        }

        result = runner.run_simulation(parameters)

        if result['success']:
            deposit = result['total_energy_deposit']
            print(f"   ✅ Energy deposited: {deposit:.3f} MeV ({deposit / energy * 100:.1f}%)")
            results.append((energy, deposit))
        else:
            print(f"   ❌ Failed: {result.get('error')}")

    print("\n📊 Summary:")
    print("   Energy [MeV] | Deposited [MeV] | Percentage")
    print("   " + "-" * 50)
    for energy, deposit in results:
        print(f"   {energy:>12.1f} | {deposit:>15.3f} | {deposit / energy * 100:>9.1f}%")

    print("=" * 60)
    return len(results) == len(energies)


def test_parser_only():
    """Test parser with existing CSV file"""

    print("\n" + "=" * 60)
    print("Test 3: Parser Test (if you have existing CSV)")
    print("=" * 60)

    # Look for existing test file
    test_file = Path(r"C:\Thesis\geant4\Water-Phantom\build\Release\test_output\event_000000.csv")

    if test_file.exists():
        print(f"\nParsing: {test_file}")

        parser = Geant4OutputParser()
        result = parser.parse_event_file(str(test_file))

        if result['success']:
            print("✅ Parsing successful!")
            print(f"   Event ID: {result['event_id']}")
            print(f"   Energy deposit: {result['total_energy_deposit']:.3f} MeV")
            print(f"   Steps: {result['num_steps']}")

            if result['steps']:
                print(f"\n📍 Sample steps (first 3):")
                for i, step in enumerate(result['steps'][:3]):
                    print(f"   Step {i}: Pos=({step['PosX_cm']:.2f}, {step['PosY_cm']:.2f}, {step['PosZ_cm']:.2f}) cm")
                    print(
                        f"           KE={step['KineticEnergy_MeV']:.3f} MeV, dE={step['EnergyDeposited_MeV']:.4f} MeV")

            return True
    else:
        print(f"⚠️  Test file not found: {test_file}")
        print("   Run Geant4 manually first to create test data")

    print("=" * 60)
    return False


if __name__ == "__main__":
    print("\n" + "🔬" * 30)
    print("GEANT4 INTEGRATION TEST SUITE")
    print("🔬" * 30 + "\n")

    tests = [
        ("Single Simulation", test_single_simulation),
        ("Multiple Energies", test_multiple_energies),
        ("Parser Only", test_parser_only),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            import traceback

            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")

    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nTotal: {passed}/{total} passed")
    print("=" * 60)

    sys.exit(0 if passed == total else 1)