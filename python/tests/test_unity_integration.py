"""
Test Unity Multi-Agent Integration
"""

import sys
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from unity_integration.unity_connector import UnityConnector


def test_unity_connection():
    """Test connection to Unity with multiple agents"""

    print("\n" + "🎮" * 35)
    print("UNITY MULTI-AGENT CONNECTION TEST")
    print("🎮" * 35 + "\n")

    print("📋 INSTRUCTIONS:")
    print("   1. Open Unity")
    print("   2. Load scene with ParticleAgents")
    print("   3. Click PLAY in Unity Editor")
    print("   4. Press Enter here...")
    input()

    # Create connector
    connector = UnityConnector()

    # Connect
    if not connector.connect():
        print("❌ Connection failed!")
        return

    print(f"\n✅ Connected! Found {connector.num_agents} agents")
    print("\n🎮 Running 10 steps with random actions...\n")

    # Run some steps
    for step in range(10):
        # Get step info
        num_decision, num_terminal, total = connector.get_step_info()

        print(f"Step {step}:")
        print(f"  Decision agents: {num_decision}")
        print(f"  Terminal agents: {num_terminal}")
        print(f"  Total: {total}")

        # Get observations
        obs = connector.get_observations()

        if len(obs) > 0:
            print(f"  Observations shape: {obs.shape}")
            print(f"  First agent obs: {obs[0]}")

            # Random actions (3D movement)
            # Use SMALLER actions to avoid exiting phantom too quickly
            actions = np.random.uniform(-0.3, 0.3, size=(len(obs), 3))  # ← Mniejsze ruchy!
            connector.send_actions(actions)
        else:
            print(f"  ⚠️  No agents waiting for actions!")
            print(f"  All agents finished episodes - they will restart automatically")
            # Still need to step to allow Unity to restart episodes
            connector.send_actions(np.array([]))

        print()
        time.sleep(0.5)

    # Close
    connector.close()

    print("\n✅ TEST COMPLETE!")


def test_unity_continuous():
    """Test continuous running with monitoring"""

    print("\n" + "🔬" * 35)
    print("CONTINUOUS MONITORING TEST")
    print("🔬" * 35 + "\n")

    print("This test will run for 30 seconds and monitor agent states")
    print("Press Ctrl+C to stop early\n")

    connector = UnityConnector()

    if not connector.connect():
        print("❌ Connection failed!")
        return

    print(f"✅ Connected! Found {connector.num_agents} agents\n")

    step = 0
    start_time = time.time()

    try:
        while time.time() - start_time < 30:
            num_decision, num_terminal, total = connector.get_step_info()

            if step % 10 == 0:  # Print every 10 steps
                print(f"Step {step}: Decision={num_decision}, Terminal={num_terminal}, Total={total}")

            obs = connector.get_observations()

            if len(obs) > 0:
                # Smaller random actions
                actions = np.random.uniform(-0.2, 0.2, size=(len(obs), 3))
                connector.send_actions(actions)
            else:
                connector.send_actions(np.array([]))

            step += 1
            time.sleep(0.05)  # Fast stepping

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")

    connector.close()

    elapsed = time.time() - start_time
    print(f"\n✅ Test complete!")
    print(f"   Duration: {elapsed:.1f}s")
    print(f"   Total steps: {step}")
    print(f"   Steps/sec: {step/elapsed:.1f}")


if __name__ == "__main__":
    print("\nSelect test:")
    print("1. Basic connection test (10 steps)")
    print("2. Continuous monitoring (30 seconds)")

    choice = input("\nChoice (1/2): ").strip()

    if choice == "1":
        test_unity_connection()
    elif choice == "2":
        test_unity_continuous()
    else:
        print("Invalid choice")