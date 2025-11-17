"""
Test Per-Step Reward Calculator
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from rl_training.reward_calculator import StepRewardCalculator


def test_perfect_match():
    """Test reward when Unity matches Geant4 perfectly"""

    print("\n" + "=" * 60)
    print("TEST 1: Perfect Match (should give reward close to 0)")
    print("=" * 60)

    calculator = StepRewardCalculator()

    # Identical states
    unity_state = {
        'position': np.array([0.0, 0.0, 0.0]),
        'energy': 10.0,
        'direction': np.array([1.0, 0.0, 0.0]),
        'energy_deposited': 0.1
    }

    geant4_state = {
        'position': np.array([0.0, 0.0, 0.0]),
        'energy': 10.0,
        'direction': np.array([1.0, 0.0, 0.0]),
        'energy_deposited': 0.1,
        'particle_stopped': False
    }

    result = calculator.calculate_step_reward(unity_state, geant4_state)

    print(f"\nReward: {result['reward']:.4f}")
    print(f"Position error: {result['position_error']:.4f} cm")
    print(f"Energy error: {result['energy_error']:.4f} MeV")
    print(f"Direction error: {result['direction_error']:.4f} rad")
    print(f"Should terminate: {result['should_terminate']}")

    # Perfect match should give reward close to 0 (only living penalty)
    assert abs(result['reward']) < 0.01, "Perfect match should give near-zero reward!"
    assert result['position_error'] < 0.001
    assert result['energy_error'] < 0.001

    print("✅ PASSED")


def test_position_error():
    """Test reward with position error"""

    print("\n" + "=" * 60)
    print("TEST 2: Position Error (1 cm difference)")
    print("=" * 60)

    calculator = StepRewardCalculator()

    unity_state = {
        'position': np.array([0.0, 0.0, 0.0]),
        'energy': 10.0,
        'direction': np.array([1.0, 0.0, 0.0]),
        'energy_deposited': 0.1
    }

    geant4_state = {
        'position': np.array([1.0, 0.0, 0.0]),  # 1 cm difference
        'energy': 10.0,
        'direction': np.array([1.0, 0.0, 0.0]),
        'energy_deposited': 0.1,
        'particle_stopped': False
    }

    result = calculator.calculate_step_reward(unity_state, geant4_state)

    print(f"\nReward: {result['reward']:.4f}")
    print(f"Position error: {result['position_error']:.4f} cm")

    # 1 cm error should give negative reward
    assert result['reward'] < -0.5, "1 cm error should give penalty!"
    assert abs(result['position_error'] - 1.0) < 0.001

    print("✅ PASSED")


def test_large_position_error():
    """Test termination with large position error"""

    print("\n" + "=" * 60)
    print("TEST 3: Large Position Error (should terminate)")
    print("=" * 60)

    calculator = StepRewardCalculator(max_position_error=2.0)

    unity_state = {
        'position': np.array([0.0, 0.0, 0.0]),
        'energy': 10.0,
        'direction': np.array([1.0, 0.0, 0.0]),
        'energy_deposited': 0.1
    }

    geant4_state = {
        'position': np.array([3.0, 0.0, 0.0]),  # 3 cm difference
        'energy': 10.0,
        'direction': np.array([1.0, 0.0, 0.0]),
        'energy_deposited': 0.1,
        'particle_stopped': False
    }

    result = calculator.calculate_step_reward(unity_state, geant4_state)

    print(f"\nReward: {result['reward']:.4f}")
    print(f"Position error: {result['position_error']:.4f} cm")
    print(f"Should terminate: {result['should_terminate']}")
    print(f"Reason: {result['termination_reason']}")

    # Should terminate due to large error
    assert result['should_terminate'], "Should terminate with large position error!"
    assert result['reward'] < -5.0, "Should give large penalty!"
    assert "position_error_too_large" in result['termination_reason']

    print("✅ PASSED")


def test_energy_error():
    """Test reward with energy error"""

    print("\n" + "=" * 60)
    print("TEST 4: Energy Error")
    print("=" * 60)

    calculator = StepRewardCalculator()

    unity_state = {
        'position': np.array([0.0, 0.0, 0.0]),
        'energy': 10.0,
        'direction': np.array([1.0, 0.0, 0.0]),
        'energy_deposited': 0.1
    }

    geant4_state = {
        'position': np.array([0.0, 0.0, 0.0]),
        'energy': 8.0,  # 2 MeV difference
        'direction': np.array([1.0, 0.0, 0.0]),
        'energy_deposited': 0.1,
        'particle_stopped': False
    }

    result = calculator.calculate_step_reward(unity_state, geant4_state)

    print(f"\nReward: {result['reward']:.4f}")
    print(f"Energy error: {result['energy_error']:.4f} MeV")

    assert result['reward'] < -0.5, "Energy error should give penalty!"
    assert abs(result['energy_error'] - 2.0) < 0.001

    print("✅ PASSED")


def test_direction_error():
    """Test reward with direction error"""

    print("\n" + "=" * 60)
    print("TEST 5: Direction Error (90 degrees)")
    print("=" * 60)

    calculator = StepRewardCalculator()

    unity_state = {
        'position': np.array([0.0, 0.0, 0.0]),
        'energy': 10.0,
        'direction': np.array([1.0, 0.0, 0.0]),  # Along X
        'energy_deposited': 0.1
    }

    geant4_state = {
        'position': np.array([0.0, 0.0, 0.0]),
        'energy': 10.0,
        'direction': np.array([0.0, 1.0, 0.0]),  # Along Y (90° different)
        'energy_deposited': 0.1,
        'particle_stopped': False
    }

    result = calculator.calculate_step_reward(unity_state, geant4_state)

    print(f"\nReward: {result['reward']:.4f}")
    print(f"Direction error: {result['direction_error']:.4f} rad ({np.rad2deg(result['direction_error']):.1f} deg)")

    # 90 degrees should give penalty
    assert result['reward'] < -0.2, "Direction error should give penalty!"
    assert abs(result['direction_error'] - np.pi / 2) < 0.1  # ~90 degrees

    print("✅ PASSED")


def test_particle_stopped():
    """Test completion bonus when Geant4 particle stops"""

    print("\n" + "=" * 60)
    print("TEST 6: Particle Stopped (should give bonus)")
    print("=" * 60)

    calculator = StepRewardCalculator()

    unity_state = {
        'position': np.array([0.0, 0.0, 0.0]),
        'energy': 0.05,  # Low energy
        'direction': np.array([1.0, 0.0, 0.0]),
        'energy_deposited': 0.05
    }

    geant4_state = {
        'position': np.array([0.0, 0.0, 0.0]),
        'energy': 0.0,  # Stopped
        'direction': np.array([1.0, 0.0, 0.0]),
        'energy_deposited': 0.05,
        'particle_stopped': True  # ← Important!
    }

    result = calculator.calculate_step_reward(unity_state, geant4_state)

    print(f"\nReward: {result['reward']:.4f}")
    print(f"Should terminate: {result['should_terminate']}")
    print(f"Reason: {result['termination_reason']}")

    # Should get bonus for completion
    assert result['should_terminate']
    assert result['reward'] > 1.0, "Should give completion bonus!"
    assert result['termination_reason'] == "geant4_particle_stopped"

    print("✅ PASSED")


def main():
    """Run all tests"""

    print("\n" + "🧪" * 30)
    print("STEP REWARD CALCULATOR TESTS")
    print("🧪" * 30)

    try:
        test_perfect_match()
        test_position_error()
        test_large_position_error()
        test_energy_error()
        test_direction_error()
        test_particle_stopped()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())