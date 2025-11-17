"""
Test REST API Integration
Tests the complete REST server with all endpoints
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import json
import time
import numpy as np

BASE_URL = "http://localhost:5000"


def test_health_check():
    """Test /health endpoint"""

    print("\n" + "=" * 60)
    print("TEST 1: Health Check")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/health")

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    assert response.json()['status'] == 'ok'

    print("✅ PASSED")
    return response.json()


def test_initialize_agent():
    """Test /initialize endpoint"""

    print("\n" + "=" * 60)
    print("TEST 2: Initialize Agent")
    print("=" * 60)

    agent_data = {
        "agent_id": 0,
        "particle_type": "e-",
        "initial_energy": 10.0,
        "initial_position": [-6.0, 0.0, 0.0],
        "initial_direction": [1.0, 0.0, 0.0]
    }

    response = requests.post(
        f"{BASE_URL}/initialize",
        json=agent_data
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    assert response.json()['success'] == True

    print("✅ PASSED")


def test_step_execution():
    """Test /step endpoint"""

    print("\n" + "=" * 60)
    print("TEST 3: Step Execution")
    print("=" * 60)

    # Initialize first
    test_initialize_agent()

    # Send step
    step_data = {
        "agent_id": 0,
        "unity_position": [-5.9, 0.0, 0.0],
        "unity_direction": [1.0, 0.0, 0.0],
        "unity_energy": 9.95,
        "energy_deposited": 0.05
    }

    response = requests.post(
        f"{BASE_URL}/step",
        json=step_data
    )

    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Response:")
    print(f"  Reward: {result['reward']:.4f}")
    print(f"  Position error: {result['metrics']['position_error']:.4f} cm")
    print(f"  Energy error: {result['metrics']['energy_error']:.4f} MeV")
    print(f"  Processing time: {result['processing_time_ms']:.2f} ms")

    assert response.status_code == 200
    assert result['success'] == True
    assert 'reward' in result

    print("✅ PASSED")


def test_trajectory_submission():
    """Test /trajectory/submit endpoint"""

    print("\n" + "=" * 60)
    print("TEST 4: Trajectory Submission")
    print("=" * 60)

    # Generate dummy trajectory
    trajectory = {
        "agent_id": 1,
        "initial_conditions": {
            "particle_type": "e-",
            "initial_energy": 12.0,
            "initial_position": [-6.0, 0.0, 0.0],
            "initial_direction": [1.0, 0.0, 0.0]
        },
        "steps": [
            {
                "step_number": i,
                "position": [-6.0 + i * 0.1, 0.0, 0.0],
                "direction": [1.0, 0.0, 0.0],
                "energy": 12.0 - i * 0.1,
                "energy_deposited": 0.1
            }
            for i in range(50)
        ]
    }

    response = requests.post(
        f"{BASE_URL}/trajectory/submit",
        json=trajectory
    )

    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Response:")
    print(f"  Trajectory ID: {result['trajectory_id']}")
    print(f"  Buffer count: {result['buffer_count']}/{result['buffer_size']}")
    print(f"  Buffer utilization: {result['buffer_utilization'] * 100:.1f}%")

    assert response.status_code == 200
    assert result['success'] == True

    print("✅ PASSED")
    return result['trajectory_id']


def test_buffer_status():
    """Test /trajectory/buffer_status endpoint"""

    print("\n" + "=" * 60)
    print("TEST 5: Buffer Status")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/trajectory/buffer_status")

    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Response:")
    print(f"  Buffer count: {result['buffer_count']}")
    print(f"  Buffer size: {result['buffer_size']}")
    print(f"  Total received: {result['total_received']}")
    print(f"  Total processed: {result['total_processed']}")

    assert response.status_code == 200
    assert result['success'] == True

    print("✅ PASSED")


def test_batch_processing():
    """Test /trajectory/process_batch endpoint"""

    print("\n" + "=" * 60)
    print("TEST 6: Batch Processing")
    print("=" * 60)

    # Submit multiple trajectories first
    print("Submitting 5 trajectories...")
    for i in range(5):
        test_trajectory_submission()
        time.sleep(0.1)

    # Process batch
    print("\nProcessing batch...")
    response = requests.post(f"{BASE_URL}/trajectory/process_batch")

    print(f"Status Code: {response.status_code}")
    result = response.json()

    if result['success']:
        print(f"✅ Batch processed successfully!")
        print(f"  Trajectories processed: {result['trajectories_processed']}")
        print(f"  Processing time: {result['processing_time_seconds']:.2f}s")

        if result['results']:
            first_result = result['results'][0]
            print(f"\nFirst trajectory result:")
            print(f"  Trajectory ID: {first_result['trajectory_id']}")
            print(f"  Agent ID: {first_result['agent_id']}")
            print(f"  Total reward: {first_result['episode_summary']['total_reward']:.3f}")
            print(f"  Mean position error: {first_result['episode_summary']['mean_position_error']:.3f} cm")
    else:
        print(f"❌ Batch processing failed: {result.get('error')}")

    assert response.status_code == 200

    print("✅ PASSED")


def main():
    """Run all tests"""

    print("\n" + "🧪" * 30)
    print("REST API INTEGRATION TESTS")
    print("🧪" * 30)
    print("\n⚠️  Make sure REST server is running!")
    print("   Run: python rest_interface/rest_server.py\n")

    input("Press Enter when server is ready...")

    try:
        test_health_check()
        test_initialize_agent()
        test_step_execution()
        test_trajectory_submission()
        test_buffer_status()
        test_batch_processing()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60 + "\n")

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to server!")
        print("   Make sure REST server is running on http://localhost:5000\n")
        return 1
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