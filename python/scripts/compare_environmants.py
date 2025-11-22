"""
compare_environments.py - Compare Old vs New Environment

Runs side-by-side comparison of:
- Random agent performance
- Episode statistics
- Reward distributions
- Physics consistency

Usage:
    python compare_environments.py --episodes 50
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# Import both environments (old and new)
# Assume old environment is backed up as environment_old.py
try:
    from src.training.environment import Geant4ParticleEnv as AdvancedEnv

    ADVANCED_ENV_AVAILABLE = True
except ImportError:
    ADVANCED_ENV_AVAILABLE = False
    print("⚠️ Advanced environment not available")

try:
    from src.training.environment_old import Geant4ParticleEnv as OldEnv

    OLD_ENV_AVAILABLE = True
except ImportError:
    OLD_ENV_AVAILABLE = False
    print("⚠️ Old environment not available (expected at environment_old.py)")


class EnvironmentComparison:
    """
    Compare two environment implementations.
    """

    def __init__(self, num_episodes: int = 50):
        """
        Initialize comparison.

        Args:
            num_episodes: Number of episodes to test
        """
        self.num_episodes = num_episodes
        self.results = {
            'old': {'rewards': [], 'lengths': [], 'times': []},
            'advanced': {'rewards': [], 'lengths': [], 'times': []}
        }

    def run_comparison(self):
        """Run comparison on both environments"""

        print("\n" + "=" * 80)
        print("ENVIRONMENT COMPARISON")
        print("=" * 80 + "\n")

        # Test old environment
        if OLD_ENV_AVAILABLE:
            print("[Test] Testing OLD environment...")
            self._test_environment(OldEnv, 'old')
        else:
            print("[Test] ⚠️ Old environment not available - skipping")

        # Test new environment
        if ADVANCED_ENV_AVAILABLE:
            print("\n[Test] Testing ADVANCED environment...")
            self._test_environment(AdvancedEnv, 'advanced')
        else:
            print("[Test] ⚠️ Advanced environment not available - skipping")

        # Generate comparison report
        self._generate_report()

    def _test_environment(self, env_class, name: str):
        """Test a single environment with random agent"""

        try:
            env = env_class(verbose=False)
        except Exception as e:
            print(f"❌ Failed to create {name} environment: {e}")
            return

        for episode in range(self.num_episodes):
            start_time = time.time()

            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done and episode_length < 200:
                action = env.action_space.sample()  # Random action
                obs, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward
                episode_length += 1
                done = terminated or truncated

            episode_time = time.time() - start_time

            self.results[name]['rewards'].append(episode_reward)
            self.results[name]['lengths'].append(episode_length)
            self.results[name]['times'].append(episode_time)

            if (episode + 1) % 10 == 0:
                print(f"  Episode {episode + 1}/{self.num_episodes}: "
                      f"R={episode_reward:.1f}, L={episode_length}, T={episode_time:.2f}s")

        env.close()
        print(f"✅ {name.upper()} environment tested\n")

    def _generate_report(self):
        """Generate comparison report"""

        print("\n" + "=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80 + "\n")

        for name in ['old', 'advanced']:
            if len(self.results[name]['rewards']) == 0:
                continue

            rewards = np.array(self.results[name]['rewards'])
            lengths = np.array(self.results[name]['lengths'])
            times = np.array(self.results[name]['times'])

            print(f"{name.upper()} Environment:")
            print(f"  Reward:  {rewards.mean():.2f} ± {rewards.std():.2f} "
                  f"(min: {rewards.min():.2f}, max: {rewards.max():.2f})")
            print(f"  Length:  {lengths.mean():.1f} ± {lengths.std():.1f} steps")
            print(f"  Time:    {times.mean():.2f} ± {times.std():.2f} seconds/episode")
            print()

        # Statistical comparison
        if len(self.results['old']['rewards']) > 0 and len(self.results['advanced']['rewards']) > 0:
            self._statistical_comparison()

        # Generate plots
        self._plot_comparison()

    def _statistical_comparison(self):
        """Perform statistical tests"""

        try:
            from scipy import stats
        except ImportError:
            print("⚠️ scipy not available - skipping statistical tests")
            return

        print("STATISTICAL COMPARISON:")
        print("-" * 80)

        old_rewards = np.array(self.results['old']['rewards'])
        new_rewards = np.array(self.results['advanced']['rewards'])

        # T-test
        t_stat, p_value = stats.ttest_ind(old_rewards, new_rewards)

        print(f"T-test (rewards):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")

        if p_value < 0.05:
            if new_rewards.mean() > old_rewards.mean():
                print(f"  ✅ Advanced environment shows SIGNIFICANT improvement (p < 0.05)")
            else:
                print(f"  ⚠️ Advanced environment shows SIGNIFICANT degradation (p < 0.05)")
        else:
            print(f"  No significant difference (p >= 0.05)")

        print()

    def _plot_comparison(self):
        """Create comparison plots"""

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        colors = {'old': 'blue', 'advanced': 'orange'}

        # Plot 1: Reward distribution
        ax = axes[0, 0]
        for name in ['old', 'advanced']:
            if len(self.results[name]['rewards']) > 0:
                ax.hist(self.results[name]['rewards'], bins=20, alpha=0.6,
                        label=name.upper(), color=colors[name])
        ax.set_xlabel('Episode Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Reward Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Episode length distribution
        ax = axes[0, 1]
        for name in ['old', 'advanced']:
            if len(self.results[name]['lengths']) > 0:
                ax.hist(self.results[name]['lengths'], bins=20, alpha=0.6,
                        label=name.upper(), color=colors[name])
        ax.set_xlabel('Episode Length')
        ax.set_ylabel('Frequency')
        ax.set_title('Episode Length Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Reward over episodes
        ax = axes[1, 0]
        for name in ['old', 'advanced']:
            if len(self.results[name]['rewards']) > 0:
                rewards = self.results[name]['rewards']
                ax.plot(rewards, alpha=0.7, label=name.upper(), color=colors[name])
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Over Episodes')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Box plots
        ax = axes[1, 1]
        data_to_plot = []
        labels = []
        for name in ['old', 'advanced']:
            if len(self.results[name]['rewards']) > 0:
                data_to_plot.append(self.results[name]['rewards'])
                labels.append(name.upper())

        if len(data_to_plot) > 0:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'],
                                    [colors[n] for n in ['old', 'advanced'] if len(self.results[n]['rewards']) > 0]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

        ax.set_ylabel('Reward')
        ax.set_title('Reward Box Plot Comparison')
        ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'Environment Comparison ({self.num_episodes} episodes)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save plot
        output_path = 'environment_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[Plot] Saved: {output_path}")

        plt.show()


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(description='Compare environment implementations')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of test episodes per environment')

    args = parser.parse_args()

    if not OLD_ENV_AVAILABLE and not ADVANCED_ENV_AVAILABLE:
        print("❌ No environments available for comparison")
        print("Make sure you have:")
        print("  - src/training/environment.py (advanced)")
        print("  - src/training/environment_old.py (old version for comparison)")
        return

    comparison = EnvironmentComparison(num_episodes=args.episodes)
    comparison.run_comparison()

    print("\n" + "=" * 80)
    print("✅ COMPARISON COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()