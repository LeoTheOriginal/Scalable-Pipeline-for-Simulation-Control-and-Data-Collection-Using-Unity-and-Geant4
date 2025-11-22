"""
visualize_training.py - Training Visualization and Analysis

Creates comprehensive visualizations of training progress:
- Reward curves over time
- Reward component breakdown
- Physics metrics evolution
- Episode length statistics
- Learning rate schedules

Can be run during training or post-hoc.

Usage:
    python visualize_training.py --logdir data/logs/advanced
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
from typing import Dict, List, Tuple
import glob

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# Try to import TensorBoard event parser
try:
    from tensorboard.backend.event_processing import event_accumulator

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️ TensorBoard not available. Install with: pip install tensorboard")


class TrainingVisualizer:
    """
    Visualizes training progress from TensorBoard logs or CSV files.
    """

    def __init__(self, log_dir: str):
        """
        Initialize visualizer.

        Args:
            log_dir: Directory containing training logs
        """
        self.log_dir = log_dir
        self.data = {}

        if not os.path.exists(log_dir):
            raise ValueError(f"Log directory not found: {log_dir}")

    def load_tensorboard_data(self):
        """Load data from TensorBoard event files"""

        if not TENSORBOARD_AVAILABLE:
            print("❌ Cannot load TensorBoard data - tensorboard not installed")
            return

        print(f"[Viz] Loading TensorBoard data from: {self.log_dir}")

        # Find event files
        event_files = glob.glob(os.path.join(self.log_dir, "**", "events.out.tfevents.*"), recursive=True)

        if len(event_files) == 0:
            print("❌ No TensorBoard event files found")
            return

        print(f"[Viz] Found {len(event_files)} event files")

        # Load latest event file
        event_file = max(event_files, key=os.path.getmtime)
        print(f"[Viz] Loading: {os.path.basename(event_file)}")

        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()

        # Get available tags
        scalar_tags = ea.Tags()['scalars']
        print(f"[Viz] Found {len(scalar_tags)} scalar metrics")

        # Load all scalar data
        for tag in scalar_tags:
            try:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]

                self.data[tag] = {
                    'steps': np.array(steps),
                    'values': np.array(values)
                }
            except Exception as e:
                print(f"⚠️ Failed to load {tag}: {e}")

        print(f"[Viz] ✅ Loaded {len(self.data)} metrics\n")

    def plot_training_progress(self, save_path: str = None):
        """
        Create comprehensive training progress visualization.

        Args:
            save_path: Path to save figure (optional)
        """
        if len(self.data) == 0:
            print("❌ No data loaded. Run load_tensorboard_data() first.")
            return

        print("[Viz] Creating training progress plot...")

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Main reward curve
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_metric(ax1, 'rollout/ep_rew_mean',
                          'Episode Reward (Mean)', 'Reward', smoothing=0.9)

        # Episode length
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_metric(ax2, 'rollout/ep_len_mean',
                          'Episode Length (Mean)', 'Steps', smoothing=0.9)

        # Learning rate
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_metric(ax3, 'train/learning_rate',
                          'Learning Rate', 'LR', smoothing=0)

        # Value loss
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_metric(ax4, 'train/value_loss',
                          'Value Loss', 'Loss', smoothing=0.8)

        # Reward components
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_reward_components(ax5)

        plt.suptitle('Training Progress - Physics-Informed RL', fontsize=16, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Viz] ✅ Saved plot: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_reward_components(self, save_path: str = None):
        """
        Plot detailed breakdown of reward components.

        Args:
            save_path: Path to save figure (optional)
        """
        print("[Viz] Creating reward components plot...")

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()

        components = [
            'position', 'direction', 'energy', 'physics', 'scattering',
            'range', 'thermodynamics', 'progress', 'smoothness', 'precision_bonus'
        ]

        for idx, component in enumerate(components):
            tag = f'reward_components/{component}_mean'

            if tag in self.data:
                steps = self.data[tag]['steps']
                values = self.data[tag]['values']

                axes[idx].plot(steps, values, linewidth=1.5, alpha=0.7)
                axes[idx].set_title(component.replace('_', ' ').title(), fontsize=10, fontweight='bold')
                axes[idx].set_xlabel('Step')
                axes[idx].set_ylabel('Reward')
                axes[idx].grid(True, alpha=0.3)
                axes[idx].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            else:
                axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center',
                               transform=axes[idx].transAxes)
                axes[idx].set_title(component.replace('_', ' ').title(), fontsize=10)

        plt.suptitle('Reward Component Evolution', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Viz] ✅ Saved plot: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_physics_metrics(self, save_path: str = None):
        """
        Plot physics-specific metrics.

        Args:
            save_path: Path to save figure (optional)
        """
        print("[Viz] Creating physics metrics plot...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Energy violations
        if 'physics/energy_violations_mean' in self.data:
            ax = axes[0, 0]
            self._plot_metric(ax, 'physics/energy_violations_mean',
                              'Energy Conservation Violations', 'Energy Gain (MeV)', smoothing=0.8)

        # Position error
        if 'physics/position_error_mean' in self.data:
            ax = axes[0, 1]
            self._plot_metric(ax, 'physics/position_error_mean',
                              'Position Prediction Error', 'Error (cm)', smoothing=0.8)

        # Violation count
        if 'physics/energy_violations_count' in self.data:
            ax = axes[1, 0]
            self._plot_metric(ax, 'physics/energy_violations_count',
                              'Energy Violations Count', 'Count', smoothing=0)

        # Max violation
        if 'physics/energy_violations_max' in self.data:
            ax = axes[1, 1]
            self._plot_metric(ax, 'physics/energy_violations_max',
                              'Maximum Energy Violation', 'Energy (MeV)', smoothing=0)

        plt.suptitle('Physics Consistency Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Viz] ✅ Saved plot: {save_path}")
        else:
            plt.show()

        plt.close()

    def _plot_metric(self, ax, tag: str, title: str, ylabel: str, smoothing: float = 0.9):
        """Helper function to plot a single metric"""

        if tag not in self.data:
            ax.text(0.5, 0.5, f'No data for {tag}', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(title)
            return

        steps = self.data[tag]['steps']
        values = self.data[tag]['values']

        # Plot raw data
        ax.plot(steps, values, alpha=0.3, linewidth=0.5, label='Raw')

        # Plot smoothed data
        if smoothing > 0 and len(values) > 1:
            smoothed = self._exponential_smoothing(values, smoothing)
            ax.plot(steps, smoothed, linewidth=2, label='Smoothed')

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Training Step')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_reward_components(self, ax):
        """Plot stacked reward components"""

        components = [
            'position', 'direction', 'energy', 'physics',
            'progress', 'thermodynamics'
        ]

        # Collect data for each component
        data_dict = {}
        steps_ref = None

        for comp in components:
            tag = f'reward_components/{comp}_mean'
            if tag in self.data:
                if steps_ref is None:
                    steps_ref = self.data[tag]['steps']
                data_dict[comp] = self.data[tag]['values']

        if len(data_dict) == 0:
            ax.text(0.5, 0.5, 'No reward component data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title('Reward Components Breakdown')
            return

        # Plot each component
        for comp, values in data_dict.items():
            ax.plot(steps_ref, values, label=comp.title(), linewidth=1.5, alpha=0.7)

        ax.set_title('Reward Components Over Time', fontsize=11, fontweight='bold')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Reward Contribution')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    @staticmethod
    def _exponential_smoothing(values: np.ndarray, alpha: float = 0.9) -> np.ndarray:
        """Apply exponential smoothing to data"""
        smoothed = np.zeros_like(values)
        smoothed[0] = values[0]

        for i in range(1, len(values)):
            smoothed[i] = alpha * smoothed[i - 1] + (1 - alpha) * values[i]

        return smoothed

    def generate_report(self, output_dir: str):
        """
        Generate comprehensive training report with all plots.

        Args:
            output_dir: Directory to save report files
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n[Report] Generating training report in: {output_dir}")

        # Generate all plots
        self.plot_training_progress(
            save_path=os.path.join(output_dir, 'training_progress.png')
        )

        self.plot_reward_components(
            save_path=os.path.join(output_dir, 'reward_components.png')
        )

        self.plot_physics_metrics(
            save_path=os.path.join(output_dir, 'physics_metrics.png')
        )

        # Generate text summary
        summary_path = os.path.join(output_dir, 'training_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TRAINING SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Log directory: {self.log_dir}\n")
            f.write(f"Metrics loaded: {len(self.data)}\n\n")

            # Final values
            f.write("FINAL VALUES:\n")
            f.write("-" * 80 + "\n")

            for key in ['rollout/ep_rew_mean', 'rollout/ep_len_mean', 'train/learning_rate']:
                if key in self.data:
                    final_value = self.data[key]['values'][-1]
                    f.write(f"{key}: {final_value:.4f}\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"[Report] ✅ Report saved to: {output_dir}")
        print(f"[Report] View plots:")
        print(f"  - training_progress.png")
        print(f"  - reward_components.png")
        print(f"  - physics_metrics.png")
        print(f"  - training_summary.txt")


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(description='Visualize training progress')
    parser.add_argument('--logdir', type=str, required=True,
                        help='Path to TensorBoard log directory')
    parser.add_argument('--output', type=str, default='training_report',
                        help='Output directory for report')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively instead of saving')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("TRAINING VISUALIZATION TOOL")
    print("=" * 80 + "\n")

    # Create visualizer
    viz = TrainingVisualizer(args.logdir)

    # Load data
    viz.load_tensorboard_data()

    if args.show:
        # Show plots interactively
        viz.plot_training_progress()
        viz.plot_reward_components()
        viz.plot_physics_metrics()
    else:
        # Generate full report
        viz.generate_report(args.output)

    print("\n" + "=" * 80)
    print("✅ DONE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()