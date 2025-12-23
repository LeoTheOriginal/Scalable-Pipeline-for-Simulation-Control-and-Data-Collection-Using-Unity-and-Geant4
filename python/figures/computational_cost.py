import matplotlib.pyplot as plt
import numpy as np


def create_computational_cost_comparison(
        geant4_time_ms=500,  # Average time per trajectory in ms
        ml_time_ms=5,  # Average time per trajectory in ms
        output_path='motivation_comparison.png'
):
    """
    Creates a comparison chart of computational costs.

    Args:
        geant4_time_ms: Average Geant4 simulation time in milliseconds
        ml_time_ms: Average ML agent inference time in milliseconds
        output_path: Path to save the figure
    """

    # Data
    methods = ['Geant4\nMonte Carlo', 'ML Agent\n(Trained Model)']
    times = [geant4_time_ms, ml_time_ms]
    colors = ['#e74c3c', '#2ecc71']

    # Calculate speedup
    speedup = geant4_time_ms / ml_time_ms

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left subplot: Computation time comparison
    bars = ax1.bar(methods, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Time per Trajectory (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Computational Cost Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(times) * 1.2)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{time:.1f} ms',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Right subplot: FPS comparison
    fps_geant4 = 1000 / geant4_time_ms
    fps_ml = 1000 / ml_time_ms
    fps_values = [fps_geant4, fps_ml]

    bars2 = ax2.bar(methods, fps_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Trajectories per Second (FPS)', fontsize=12, fontweight='bold')
    ax2.set_title('Throughput Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(fps_values) * 1.2)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, fps in zip(bars2, fps_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{fps:.1f} FPS',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add speedup annotation
    fig.text(0.5, 0.02, f'Speedup: {speedup:.1f}× faster',
             ha='center', fontsize=13, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")

    return fig


# Example usage with placeholder values
# You need to replace these with your actual measurements
create_computational_cost_comparison(
    geant4_time_ms=500,  # TODO: Measure actual time
    ml_time_ms=5,  # TODO: Measure actual time
)