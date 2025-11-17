"""
Analyze Training Data
Generate plots and statistics for thesis
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from data_collection.data_collector import DataCollector


def load_all_data(data_dir: str):
    """Load all training data"""

    print("📂 Loading training data...")

    collector = DataCollector(output_directory=data_dir)
    dataset = collector.load_dataset(file_index=0)

    if not dataset:
        print("❌ Failed to load data!")
        return None

    # Extract arrays
    unity_obs = np.array(dataset['unity_observations'])  # Shape: (N, 1, 10)
    unity_obs = unity_obs.squeeze(axis=1)  # Shape: (N, 10)

    geant4_energies = np.array([
        r['total_energy_deposit'] for r in dataset['geant4_results']
    ])

    parameters = dataset['parameters']
    input_energies = np.array([p['particle_energy'] for p in parameters])

    print(f"✅ Loaded {len(unity_obs)} samples")
    print(f"   Unity observations shape: {unity_obs.shape}")
    print(f"   Geant4 energies shape: {geant4_energies.shape}")

    return {
        'unity_observations': unity_obs,
        'geant4_energies': geant4_energies,
        'input_energies': input_energies,
        'parameters': parameters
    }


def plot_energy_distribution(data, save_path: Path):
    """Plot energy distribution"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Input vs Deposited Energy
    ax = axes[0]
    ax.scatter(data['input_energies'], data['geant4_energies'],
               alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    # Perfect deposition line (y=x)
    max_energy = max(data['input_energies'].max(), data['geant4_energies'].max())
    ax.plot([0, max_energy], [0, max_energy], 'r--',
            label='Perfect deposition (100%)', linewidth=2)

    ax.set_xlabel('Input Energy [MeV]', fontsize=12)
    ax.set_ylabel('Deposited Energy [MeV]', fontsize=12)
    ax.set_title('Energy Deposition in Water Phantom', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Calculate deposition efficiency
    efficiency = data['geant4_energies'] / data['input_energies'] * 100
    mean_eff = np.mean(efficiency)

    ax.text(0.05, 0.95, f'Mean efficiency: {mean_eff:.1f}%',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Histogram of deposited energies
    ax = axes[1]
    ax.hist(data['geant4_energies'], bins=20, alpha=0.7,
            edgecolor='black', linewidth=1)
    ax.axvline(np.mean(data['geant4_energies']), color='red',
               linestyle='--', linewidth=2, label=f'Mean: {np.mean(data["geant4_energies"]):.2f} MeV')
    ax.set_xlabel('Deposited Energy [MeV]', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Deposited Energy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {save_path}")
    plt.close()


def plot_deposition_efficiency(data, save_path: Path):
    """Plot energy deposition efficiency"""

    efficiency = data['geant4_energies'] / data['input_energies'] * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Efficiency vs Input Energy
    ax = axes[0]
    ax.scatter(data['input_energies'], efficiency,
               alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax.axhline(100, color='red', linestyle='--', linewidth=2, label='100% efficiency')
    ax.axhline(np.mean(efficiency), color='green', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(efficiency):.1f}%')

    ax.set_xlabel('Input Energy [MeV]', fontsize=12)
    ax.set_ylabel('Deposition Efficiency [%]', fontsize=12)
    ax.set_title('Energy Deposition Efficiency', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([90, 105])

    # Efficiency histogram
    ax = axes[1]
    ax.hist(efficiency, bins=20, alpha=0.7, edgecolor='black', linewidth=1)
    ax.axvline(np.mean(efficiency), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(efficiency):.1f}%')
    ax.axvline(np.median(efficiency), color='green', linestyle='--',
               linewidth=2, label=f'Median: {np.median(efficiency):.1f}%')

    ax.set_xlabel('Deposition Efficiency [%]', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Efficiency', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {save_path}")
    plt.close()


def plot_unity_observations(data, save_path: Path):
    """Plot Unity observation distributions"""

    unity_obs = data['unity_observations']

    # Feature names
    feature_names = [
        'Pos X', 'Pos Y', 'Pos Z',
        'Vel X', 'Vel Y', 'Vel Z',
        'Energy', 'Dir X', 'Dir Y', 'Dir Z'
    ]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        values = unity_obs[:, i]

        if np.std(values) > 0.001:  # Variable feature
            ax.hist(values, bins=20, alpha=0.7, edgecolor='black', linewidth=1)
            ax.set_title(f'{name}\n(μ={np.mean(values):.3f}, σ={np.std(values):.3f})',
                         fontsize=10, fontweight='bold')
        else:  # Constant feature
            ax.bar([0], [len(values)], width=0.5, alpha=0.7, edgecolor='black')
            ax.set_title(f'{name}\n(constant: {np.mean(values):.3f})',
                         fontsize=10, fontweight='bold')
            ax.set_xticks([0])
            ax.set_xticklabels([f'{np.mean(values):.2f}'])

        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Unity ML-Agents Observations Distribution',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {save_path}")
    plt.close()


def generate_statistics_report(data, save_path: Path):
    """Generate text statistics report"""

    report = []
    report.append("=" * 70)
    report.append("TRAINING DATA STATISTICS REPORT")
    report.append("=" * 70)
    report.append("")

    # Dataset info
    report.append("📊 DATASET INFORMATION:")
    report.append(f"   Total samples: {len(data['unity_observations'])}")
    report.append(f"   Unity observations shape: {data['unity_observations'].shape}")
    report.append(f"   Features per observation: {data['unity_observations'].shape[1]}")
    report.append("")

    # Energy statistics
    report.append("⚡ ENERGY STATISTICS:")
    report.append(f"   Input Energy:")
    report.append(f"      Mean:   {np.mean(data['input_energies']):.3f} MeV")
    report.append(f"      Std:    {np.std(data['input_energies']):.3f} MeV")
    report.append(f"      Min:    {np.min(data['input_energies']):.3f} MeV")
    report.append(f"      Max:    {np.max(data['input_energies']):.3f} MeV")
    report.append("")

    report.append(f"   Deposited Energy:")
    report.append(f"      Mean:   {np.mean(data['geant4_energies']):.3f} MeV")
    report.append(f"      Std:    {np.std(data['geant4_energies']):.3f} MeV")
    report.append(f"      Min:    {np.min(data['geant4_energies']):.3f} MeV")
    report.append(f"      Max:    {np.max(data['geant4_energies']):.3f} MeV")
    report.append("")

    # Deposition efficiency
    efficiency = data['geant4_energies'] / data['input_energies'] * 100
    report.append(f"   Deposition Efficiency:")
    report.append(f"      Mean:   {np.mean(efficiency):.2f}%")
    report.append(f"      Std:    {np.std(efficiency):.2f}%")
    report.append(f"      Min:    {np.min(efficiency):.2f}%")
    report.append(f"      Max:    {np.max(efficiency):.2f}%")
    report.append("")

    # Unity observation statistics
    report.append("🎮 UNITY OBSERVATION STATISTICS:")
    feature_names = [
        'Pos X', 'Pos Y', 'Pos Z',
        'Vel X', 'Vel Y', 'Vel Z',
        'Energy', 'Dir X', 'Dir Y', 'Dir Z'
    ]

    for i, name in enumerate(feature_names):
        values = data['unity_observations'][:, i]
        report.append(f"   {name:10s}: mean={np.mean(values):8.4f}, std={np.std(values):8.4f}, "
                      f"min={np.min(values):8.4f}, max={np.max(values):8.4f}")

    report.append("")
    report.append("=" * 70)

    # Save report
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    # Print report
    print('\n'.join(report))
    print(f"\n📄 Report saved to: {save_path}")


def main():
    """Main analysis routine"""

    print("\n" + "📊" * 35)
    print("TRAINING DATA ANALYSIS")
    print("📊" * 35 + "\n")

    # Paths
    data_dir = r"C:\Thesis\python\training_data"
    output_dir = Path(r"C:\Thesis\python\analysis_results")
    output_dir.mkdir(exist_ok=True)

    # Load data
    data = load_all_data(data_dir)
    if not data:
        return

    print("\n📊 Generating plots and statistics...\n")

    # Generate plots
    plot_energy_distribution(data, output_dir / "energy_distribution.png")
    plot_deposition_efficiency(data, output_dir / "deposition_efficiency.png")
    plot_unity_observations(data, output_dir / "unity_observations.png")

    # Generate text report
    generate_statistics_report(data, output_dir / "statistics_report.txt")

    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
    print("\n📁 Generated files:")
    for file in sorted(output_dir.iterdir()):
        print(f"   📄 {file.name}")

    print("\n" + "=" * 70)
    print("🎉 ANALYSIS COMPLETE!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()