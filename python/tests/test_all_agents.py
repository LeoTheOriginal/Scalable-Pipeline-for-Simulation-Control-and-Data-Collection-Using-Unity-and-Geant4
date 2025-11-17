"""
Test All Agents - Track trajectories of all 16 agents
"""

import sys
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from unity_integration.unity_connector import UnityConnector


def test_all_agents():
    print("\n" + "🔬" * 35)
    print("ALL AGENTS TRAJECTORY TEST")
    print("🔬" * 35 + "\n")

    print("📋 SETUP:")
    print("   1. Unity: Play mode")
    print("   2. Use Gravity = FALSE (CRITICAL!)")
    print("   3. usePhysicsSimulation = true")
    print("   4. Press Enter...")
    input()

    connector = UnityConnector()

    if not connector.connect():
        print("❌ Failed!")
        return

    print(f"✅ Connected! {connector.num_agents} agents\n")

    # Track all agents
    agent_data = {i: {'positions': [], 'velocities': [], 'energies': []}
                  for i in range(16)}

    max_steps = 50
    print("Collecting 50 steps (ZERO actions)...\n")

    for step in range(max_steps):
        obs = connector.get_observations()

        if len(obs) == 0:
            break

        # Record each agent
        for i in range(min(len(obs), 16)):
            agent_data[i]['positions'].append(obs[i][0:3].copy())
            agent_data[i]['velocities'].append(obs[i][3:6].copy())
            agent_data[i]['energies'].append(obs[i][6])

        # ZERO actions (no forces)
        actions = np.zeros((len(obs), 3))
        connector.send_actions(actions)

        if step % 10 == 0:
            vel = obs[0][3:6]
            print(f"Step {step}: Agent 0 velocity=({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f})")

        time.sleep(0.02)

    # Analysis
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    for i in range(16):
        if len(agent_data[i]['positions']) < 2:
            continue

        positions = np.array(agent_data[i]['positions'])
        velocities = np.array(agent_data[i]['velocities'])

        start = positions[0]
        end = positions[-1]
        displacement = end - start
        distance = np.linalg.norm(displacement)

        avg_vel = np.mean(velocities, axis=0)

        # Check direction
        moving_right = displacement[0] > 0.5  # Should move in +X
        falling = abs(displacement[1]) > abs(displacement[0])  # Y > X bad!

        status = "✅" if moving_right and not falling else "⚠️"

        print(f"{status} Agent {i:2d}: "
              f"ΔX={displacement[0]:6.2f}, ΔY={displacement[1]:6.2f}, ΔZ={displacement[2]:6.2f} cm | "
              f"Dist={distance:5.2f} cm | "
              f"VelAvg=({avg_vel[0]:.2f}, {avg_vel[1]:.2f}, {avg_vel[2]:.2f})")

        if falling:
            print(f"       ⚠️  FALLING! Disable gravity!")

    connector.close()


if __name__ == "__main__":
    test_all_agents()