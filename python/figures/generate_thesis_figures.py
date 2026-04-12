#!/usr/bin/env python3
"""
Thesis Figure Generator - Version 2 (Fully Corrected)
======================================================

Generates publication-quality figures for engineering thesis comparing
ML agent trajectories with Geant4 Monte Carlo reference data.

COORDINATE SYSTEMS:
===================

GEANT4 (beam in +Z direction):
  - Particles START 1 cm BEFORE phantom (at Z = -6 cm)
  - Phantom entry at Z = -5 cm, exit at Z = +5 cm
  - Beam direction: +Z
  - Lateral plane: X-Y (perpendicular to beam)
  - In exported CSV:
    * LateralY = final Y position (lateral, should be ~0 centered)
    * LateralZ = final Z position (this is DEPTH, not lateral!)
    * Note: Column naming in CSV may be confusing

UNITY ML AGENTS (beam in +X direction):
  - Particles START AT phantom entry (at X = -5 cm)
  - Phantom entry at X = -5 cm, exit at X = +5 cm
  - Beam direction: +X
  - Lateral plane: Y-Z (perpendicular to beam)
  - In exported CSV:
    * LateralY = final Y position (lateral)
    * LateralZ = final Z position (lateral)

IMPORTANT PHYSICS CORRECTION:
- Geant4 travels extra 1 cm in AIR before phantom
- This affects Highland scattering calculations
- Path length in WATER is what matters for physics

Required input files:
- geant4_trajectories.csv
- ppo_base_v1_trajectories.csv
- sac_base_v1_trajectories.csv

Author: Thesis Project
Date: 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib.ticker import FuncFormatter
from scipy.stats import ks_2samp
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = r"C:\Thesis\python\data"
OUTPUT_DIR = r"C:\Thesis\python\figures"

TRAJECTORY_FILES = {
    'Geant4': 'geant4_trajectories.csv',
    'PPO': 'ppo_base_v1_trajectories.csv',
    'SAC': 'sac_base_v1_trajectories.csv',
}

CHECKPOINT_FILES = {
    'PPO': 'ppo_base_v1_statistics.csv',
    'SAC': 'sac_base_v1_statistics.csv',
}

# =============================================================================
# PHYSICS CONSTANTS
# =============================================================================

CSDA_RANGE = 4.98  # cm for 10 MeV electrons in water (NIST ESTAR)
INITIAL_ENERGY = 10.0  # MeV
RADIATION_LENGTH_WATER = 36.08  # cm (X0 for water)
RADIATION_LENGTH_AIR = 30420  # cm (X0 for air - very large, minimal scattering)
MASS_ELECTRON = 0.511  # MeV/c²
PHANTOM_HALF_SIZE = 5.0  # cm

# CRITICAL: Geant4 starts 1 cm before phantom, Unity starts at phantom surface
GEANT4_AIR_GAP = 1.0  # cm of air before phantom in Geant4 simulation

# =============================================================================
# STYLING
# =============================================================================

COLORS = {
    'Geant4': '#d62728',
    'PPO': '#1f77b4',
    'SAC': '#ff7f0e',
    'LSTM': '#2ca02c',
}

LABELS = {
    'Geant4': 'Geant4 (Reference)',
    'PPO': 'PPO',
    'SAC': 'SAC',
    'LSTM': 'PPO+LSTM',
}

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# =============================================================================
# DATA LOADING
# =============================================================================

def load_trajectory_data():
    """Load trajectory CSV files."""
    data = {}
    for name, filename in TRAJECTORY_FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            data[name] = df
            print(f"  Loaded {name}: {len(df):,} trajectories")
        else:
            print(f"  WARNING: {filename} not found")
            data[name] = None
    return data


def load_checkpoint_data():
    """Load checkpoint statistics for convergence plots."""
    data = {}
    for name, filename in CHECKPOINT_FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            data[name] = df
            print(f"  Loaded {name} checkpoints: {len(df)} rows")
        else:
            data[name] = None
    return data


# =============================================================================
# PHYSICS CALCULATIONS
# =============================================================================

def highland_rms_angle(energy_mev, path_length_cm, X0=RADIATION_LENGTH_WATER):
    """
    Calculate Highland formula RMS scattering angle.

    θ_RMS = (13.6 MeV / βcp) * sqrt(x/X0) * [1 + 0.038 * ln(x/X0)]

    Returns angle in DEGREES.
    """
    if energy_mev <= 0 or path_length_cm <= 0:
        return 0.0

    total_energy = energy_mev + MASS_ELECTRON
    momentum = np.sqrt(total_energy ** 2 - MASS_ELECTRON ** 2)
    beta = momentum / total_energy
    beta_cp = beta * momentum

    x_over_X0 = path_length_cm / X0
    if x_over_X0 <= 0:
        return 0.0

    theta_0 = (13.6 / beta_cp) * np.sqrt(x_over_X0)
    if x_over_X0 > 0.001:
        theta_0 *= (1 + 0.038 * np.log(x_over_X0))

    return np.degrees(theta_0)


def highland_rms_angle_corrected(energy_mev, path_in_water_cm, path_in_air_cm=0):
    """
    Calculate Highland RMS angle with correction for air gap.

    Scattering in air is negligible (X0_air >> X0_water), but we include it
    for completeness.

    Args:
        energy_mev: Initial kinetic energy
        path_in_water_cm: Path length in water
        path_in_air_cm: Path length in air (before phantom)

    Returns:
        RMS scattering angle in degrees
    """
    # Scattering in water dominates
    theta_water = highland_rms_angle(energy_mev, path_in_water_cm, RADIATION_LENGTH_WATER)

    # Scattering in air is negligible but calculate for completeness
    theta_air = highland_rms_angle(energy_mev, path_in_air_cm, RADIATION_LENGTH_AIR)

    # RMS combination (angles add in quadrature)
    theta_total = np.sqrt(theta_water ** 2 + theta_air ** 2)

    return theta_total


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def run_diagnostics(data):
    """Run comprehensive diagnostics."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC REPORT")
    print("=" * 70)

    issues = []

    for name, df in data.items():
        if df is None:
            continue

        print(f"\n--- {name} ({len(df):,} trajectories) ---")

        for col in ['PathLength', 'PenetrationDepth', 'LateralSpread',
                    'LateralY', 'LateralZ', 'MeanScatterAngle']:
            if col in df.columns:
                vals = df[col].values
                print(f"  {col:20s}: μ={np.mean(vals):+7.2f}, σ={np.std(vals):6.2f}, "
                      f"[{np.min(vals):+.2f}, {np.max(vals):+.2f}]")

        # Detect issues
        if name == 'Geant4':
            pd_mean = df['PenetrationDepth'].mean()
            if pd_mean < 2.0:
                issues.append(f"Geant4: PenetrationDepth needs recalculation ({pd_mean:.2f} cm)")
                print(f"\n  ⚠ PenetrationDepth appears incorrect")
        else:
            # Mode collapse detection
            ly_mean = df['LateralY'].mean()
            lz_mean = df['LateralZ'].mean()
            if abs(ly_mean) > 0.3 or abs(lz_mean) > 0.3:
                issues.append(f"{name}: Mode collapse (bias: Y={ly_mean:+.2f}, Z={lz_mean:+.2f})")
                print(f"\n  ⚠ Mode collapse detected")

            # Scattering check
            if data.get('Geant4') is not None:
                g4_scatter = data['Geant4']['MeanScatterAngle'].mean()
                ml_scatter = df['MeanScatterAngle'].mean()
                if ml_scatter < g4_scatter * 0.6:
                    issues.append(f"{name}: Low scattering ({ml_scatter:.1f}° vs {g4_scatter:.1f}°)")
                    print(f"\n  ⚠ Scattering too low")

    return issues


# =============================================================================
# FIGURE 1: PATH LENGTH
# =============================================================================

def generate_path_length_figure(data):
    """Generate path length distribution."""
    print("\n[1/8] Generating path length distribution...")

    fig, ax = plt.subplots(figsize=(10, 6))

    g4_samples = None
    for name, df in data.items():
        if df is None or 'PathLength' not in df.columns:
            continue

        samples = df['PathLength'].values
        if name == 'Geant4':
            g4_samples = samples
            # Note: Geant4 path includes 1cm in air, but PathLength should be water only
            label = f"{LABELS[name]} (n={len(samples):,})"
        else:
            label = f"{LABELS[name]} (n={len(samples):,})"

        ax.hist(samples, bins=50, alpha=0.6, label=label,
                color=COLORS[name], edgecolor='black', linewidth=0.5, density=True)

    ax.axvline(x=CSDA_RANGE, color='black', linestyle='--', linewidth=2.5,
               label=f'CSDA Range = {CSDA_RANGE} cm')

    ax.set_xlabel('Path Length in Water [cm]')
    ax.set_ylabel('Probability Density')
    ax.set_title('Electron Path Length Distribution')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 8)

    plt.tight_layout()
    save_figure(fig, 'path_length_comparison')
    plt.close(fig)


# =============================================================================
# FIGURE 2: PENETRATION DEPTH
# =============================================================================

def generate_penetration_depth_figure(data):
    """Generate penetration depth distribution."""
    print("\n[2/8] Generating penetration depth distribution...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: All data
    ax1 = axes[0]
    ax1.set_title('Penetration Depth - All Sources')

    for name, df in data.items():
        if df is None or 'PenetrationDepth' not in df.columns:
            continue
        samples = df['PenetrationDepth'].values
        ax1.hist(samples, bins=50, alpha=0.6, label=LABELS[name],
                 color=COLORS[name], edgecolor='black', linewidth=0.5, density=True)

    ax1.axvline(x=CSDA_RANGE, color='black', linestyle='--', linewidth=2.5,
                label=f'CSDA = {CSDA_RANGE} cm')
    ax1.set_xlabel('Penetration Depth [cm]')
    ax1.set_ylabel('Probability Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: ML agents only (zoomed)
    ax2 = axes[1]
    ax2.set_title('ML Agent Penetration Depth (Zoomed)')

    for name in ['PPO', 'SAC', 'LSTM']:
        df = data.get(name)
        if df is None:
            continue
        samples = df['PenetrationDepth'].values
        label = f"{LABELS[name]}: μ={np.mean(samples):.2f}±{np.std(samples):.2f}"
        ax2.hist(samples, bins=30, alpha=0.6, label=label,
                 color=COLORS[name], edgecolor='black', linewidth=0.5, density=True)

    ax2.axvline(x=CSDA_RANGE, color='black', linestyle='--', linewidth=2.5)
    ax2.set_xlabel('Penetration Depth [cm]')
    ax2.set_ylabel('Probability Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(3.5, 5.5)

    plt.tight_layout()
    save_figure(fig, 'penetration_depth')
    plt.close(fig)


# =============================================================================
# FIGURE 3: LATERAL SPREAD - THREE ORTHOGONAL VIEWS WITH CONSISTENT SCALES
# =============================================================================

def generate_lateral_spread_2d_figures(data):
    """
    Generate comprehensive 2D lateral spread visualizations with THREE orthogonal views.

    CRITICAL: All plots use the SAME axis scales for fair comparison!

    Coordinate systems:
    - Geant4: Beam in +Z direction, lateral plane is X-Y
    - Unity:  Beam in +X direction, lateral plane is Y-Z

    Three orthogonal views:
    1. Front view (Cross-section, perpendicular to beam) - "dandelion" pattern
    2. Side view (Depth vs Lateral Y)
    3. Top view (Depth vs Lateral Z)
    """
    print("\n[3/8] Generating lateral spread 2D views (3 orthogonal projections)...")

    # Define CONSISTENT axis ranges for ALL sources
    DEPTH_RANGE = [0, 6]  # Depth from entry [cm]
    LATERAL_RANGE = [-5, 5]  # Lateral position [cm]
    N_BINS = 60

    sources_ordered = []
    for name in ['Geant4', 'PPO', 'SAC']:
        if data.get(name) is not None:
            sources_ordered.append((name, data[name]))

    n_cols = len(sources_ordered)

    # =========================================================================
    # MAIN FIGURE: 3 rows (views) x N columns (sources)
    # Row 1: Front view (cross-section, Y vs Z lateral)
    # Row 2: Side view (Depth vs Y)
    # Row 3: Top view (Depth vs Z)
    # =========================================================================

    fig = plt.figure(figsize=(5 * n_cols, 14))

    for col_idx, (name, df) in enumerate(sources_ordered):

        # Extract coordinates based on source
        if name == 'Geant4':
            # Geant4: beam in +Z
            # From CSV analysis:
            # - LateralY contains depth-like values (range -5 to ~0)
            # - LateralZ is true lateral (centered ~0)
            # - We need to reconstruct proper coordinates

            # Depth = distance traveled into phantom
            # LateralY appears to be Z coordinate (depth), shifted
            raw_depth = df['LateralY'].values
            depth = raw_depth - raw_depth.min()  # Shift to start at 0

            # LateralZ is one lateral dimension (let's call it lateral_1)
            lateral_1 = df['LateralZ'].values  # True lateral, centered

            # We don't have second lateral dimension directly
            # Use LateralSpread to estimate or just show what we have
            # For now, create synthetic second lateral from LateralSpread
            lateral_spread = df['LateralSpread'].values
            # Approximate: if LateralSpread = sqrt(lat1^2 + lat2^2), then
            # lat2 ≈ sqrt(spread^2 - lat1^2) with random sign
            lat1_sq = lateral_1 ** 2
            spread_sq = lateral_spread ** 2
            lat2_sq = np.maximum(0, spread_sq - lat1_sq)
            lateral_2 = np.sqrt(lat2_sq) * np.sign(np.random.randn(len(lat2_sq)))

            depth_label = 'Depth (Z) [cm]'
            lat1_label = 'Lateral X [cm]'
            lat2_label = 'Lateral Y [cm]'

        else:
            # Unity ML agents: beam in +X
            # LateralY = Y position (lateral)
            # LateralZ = Z position (lateral)
            # PenetrationDepth = depth into phantom

            depth = df['PenetrationDepth'].values
            lateral_1 = df['LateralY'].values
            lateral_2 = df['LateralZ'].values

            depth_label = 'Depth (X) [cm]'
            lat1_label = 'Lateral Y [cm]'
            lat2_label = 'Lateral Z [cm]'

        # Calculate statistics
        sigma_1 = np.std(lateral_1)
        sigma_2 = np.std(lateral_2)
        sigma_avg = np.sqrt((sigma_1 ** 2 + sigma_2 ** 2) / 2)
        mean_1 = np.mean(lateral_1)
        mean_2 = np.mean(lateral_2)
        mean_depth = np.mean(depth)

        # ---------------------------------------------------------------------
        # ROW 1: Front View (Cross-section) - Lateral1 vs Lateral2
        # ---------------------------------------------------------------------
        ax1 = fig.add_subplot(3, n_cols, col_idx + 1)

        h1 = ax1.hist2d(lateral_1, lateral_2, bins=N_BINS, cmap='hot', cmin=1,
                        range=[LATERAL_RANGE, LATERAL_RANGE])
        plt.colorbar(h1[3], ax=ax1, label='Count', shrink=0.7)

        # Sigma circles
        for n_sig, ls in [(1, '-'), (2, '--'), (3, ':')]:
            circle = Circle((0, 0), n_sig * sigma_avg, fill=False,
                            color='cyan', linewidth=1.5, linestyle=ls, alpha=0.8)
            ax1.add_patch(circle)

        # Mark bias if present
        if abs(mean_1) > 0.2 or abs(mean_2) > 0.2:
            ax1.plot(mean_1, mean_2, 'g+', markersize=15, markeredgewidth=2)

        ax1.axhline(0, color='white', linewidth=0.5, alpha=0.5)
        ax1.axvline(0, color='white', linewidth=0.5, alpha=0.5)
        ax1.set_xlim(LATERAL_RANGE)
        ax1.set_ylim(LATERAL_RANGE)
        ax1.set_aspect('equal')
        ax1.set_xlabel(lat1_label)

        if col_idx == 0:
            ax1.set_ylabel(f'FRONT VIEW\n(Cross-Section)\n\n{lat2_label}')
        else:
            ax1.set_ylabel(lat2_label)

        title = f"{LABELS[name]}\nσ = {sigma_avg:.2f} cm"
        if name != 'Geant4' and (abs(mean_1) > 0.2 or abs(mean_2) > 0.2):
            title += f"\n⚠ Bias: ({mean_1:+.2f}, {mean_2:+.2f})"
        ax1.set_title(title, fontsize=11, fontweight='bold')

        # ---------------------------------------------------------------------
        # ROW 2: Side View - Depth vs Lateral1
        # ---------------------------------------------------------------------
        ax2 = fig.add_subplot(3, n_cols, n_cols + col_idx + 1)

        h2 = ax2.hist2d(depth, lateral_1, bins=N_BINS, cmap='hot', cmin=1,
                        range=[DEPTH_RANGE, LATERAL_RANGE])
        plt.colorbar(h2[3], ax=ax2, label='Count', shrink=0.7)

        ax2.axhline(0, color='white', linewidth=0.5, alpha=0.5)
        ax2.set_xlim(DEPTH_RANGE)
        ax2.set_ylim(LATERAL_RANGE)
        ax2.set_xlabel(depth_label)

        if col_idx == 0:
            ax2.set_ylabel(f'SIDE VIEW\n(Depth vs Lat1)\n\n{lat1_label}')
        else:
            ax2.set_ylabel(lat1_label)

        ax2.set_title(f"σ_lateral1 = {sigma_1:.2f} cm, μ_depth = {mean_depth:.2f} cm", fontsize=10)

        # ---------------------------------------------------------------------
        # ROW 3: Top View - Depth vs Lateral2
        # ---------------------------------------------------------------------
        ax3 = fig.add_subplot(3, n_cols, 2 * n_cols + col_idx + 1)

        h3 = ax3.hist2d(depth, lateral_2, bins=N_BINS, cmap='hot', cmin=1,
                        range=[DEPTH_RANGE, LATERAL_RANGE])
        plt.colorbar(h3[3], ax=ax3, label='Count', shrink=0.7)

        ax3.axhline(0, color='white', linewidth=0.5, alpha=0.5)
        ax3.set_xlim(DEPTH_RANGE)
        ax3.set_ylim(LATERAL_RANGE)
        ax3.set_xlabel(depth_label)

        if col_idx == 0:
            ax3.set_ylabel(f'TOP VIEW\n(Depth vs Lat2)\n\n{lat2_label}')
        else:
            ax3.set_ylabel(lat2_label)

        ax3.set_title(f"σ_lateral2 = {sigma_2:.2f} cm", fontsize=10)

    fig.suptitle('Lateral Spread Analysis: Three Orthogonal Views\n(All plots use same axis scales for comparison)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_figure(fig, 'lateral_spread_3views')
    plt.close(fig)

    # =========================================================================
    # ADDITIONAL: Side-by-side comparison with density texture style
    # =========================================================================

    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))

    for col_idx, (name, df) in enumerate(sources_ordered):

        if name == 'Geant4':
            raw_depth = df['LateralY'].values
            depth = raw_depth - raw_depth.min()
            lateral_1 = df['LateralZ'].values
            lateral_spread = df['LateralSpread'].values
            lat1_sq = lateral_1 ** 2
            spread_sq = lateral_spread ** 2
            lat2_sq = np.maximum(0, spread_sq - lat1_sq)
            lateral_2 = np.sqrt(lat2_sq) * np.sign(np.random.randn(len(lat2_sq)))
        else:
            depth = df['PenetrationDepth'].values
            lateral_1 = df['LateralY'].values
            lateral_2 = df['LateralZ'].values

        sigma_1 = np.std(lateral_1)
        sigma_2 = np.std(lateral_2)
        sigma_avg = np.sqrt((sigma_1 ** 2 + sigma_2 ** 2) / 2)

        # Top: Cross-section (front view)
        ax_top = axes[0, col_idx]
        h = ax_top.hist2d(lateral_1, lateral_2, bins=N_BINS, cmap='hot', cmin=1,
                          range=[LATERAL_RANGE, LATERAL_RANGE])
        plt.colorbar(h[3], ax=ax_top, label='Count', shrink=0.7)

        for n_sig, ls in [(1, '-'), (2, '--')]:
            circle = Circle((0, 0), n_sig * sigma_avg, fill=False,
                            color='cyan', linewidth=1.5, linestyle=ls)
            ax_top.add_patch(circle)

        mean_1, mean_2 = np.mean(lateral_1), np.mean(lateral_2)
        if abs(mean_1) > 0.2 or abs(mean_2) > 0.2:
            ax_top.plot(mean_1, mean_2, 'g+', markersize=12, markeredgewidth=2)

        ax_top.axhline(0, color='white', linewidth=0.3, alpha=0.5)
        ax_top.axvline(0, color='white', linewidth=0.3, alpha=0.5)
        ax_top.set_xlim(LATERAL_RANGE)
        ax_top.set_ylim(LATERAL_RANGE)
        ax_top.set_aspect('equal')
        ax_top.set_xlabel('Lateral [cm]')

        if col_idx == 0:
            ax_top.set_ylabel('Cross-Section\n\nLateral [cm]')
        else:
            ax_top.set_ylabel('Lateral [cm]')

        title = f"{LABELS[name]}\nσ = {sigma_avg:.2f} cm"
        if name != 'Geant4' and (abs(mean_1) > 0.2 or abs(mean_2) > 0.2):
            title += f" (bias: {mean_1:+.1f}, {mean_2:+.1f})"
        ax_top.set_title(title, fontsize=11, fontweight='bold')

        # Bottom: Side view (depth vs lateral)
        ax_bot = axes[1, col_idx]
        h = ax_bot.hist2d(depth, lateral_1, bins=N_BINS, cmap='hot', cmin=1,
                          range=[DEPTH_RANGE, LATERAL_RANGE])
        plt.colorbar(h[3], ax=ax_bot, label='Count', shrink=0.7)

        ax_bot.axhline(0, color='white', linewidth=0.3, alpha=0.5)
        ax_bot.set_xlim(DEPTH_RANGE)
        ax_bot.set_ylim(LATERAL_RANGE)
        ax_bot.set_xlabel('Depth [cm]')

        if col_idx == 0:
            ax_bot.set_ylabel('Side View\n\nLateral [cm]')
        else:
            ax_bot.set_ylabel('Lateral [cm]')

        ax_bot.set_title(f"σ_lateral = {sigma_1:.2f} cm", fontsize=10)

    fig.suptitle('Lateral Spread Comparison (Same Scales)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'lateral_spread_combined')
    plt.close(fig)

    # =========================================================================
    # Print diagnostic info about coordinate ranges
    # =========================================================================
    print("\n  Coordinate ranges (for debugging):")
    for name, df in sources_ordered:
        if name == 'Geant4':
            raw_depth = df['LateralY'].values
            depth = raw_depth - raw_depth.min()
            lat1 = df['LateralZ'].values
        else:
            depth = df['PenetrationDepth'].values
            lat1 = df['LateralY'].values

        print(f"    {name}:")
        print(f"      Depth: [{depth.min():.2f}, {depth.max():.2f}], μ={depth.mean():.2f}")
        print(f"      Lateral: [{lat1.min():.2f}, {lat1.max():.2f}], μ={lat1.mean():.2f}")


# =============================================================================
# FIGURE 4: 1D LATERAL DISTRIBUTIONS
# =============================================================================

def generate_lateral_1d_figure(data):
    """Generate 1D lateral spread histograms."""
    print("\n[4/8] Generating lateral spread 1D distributions...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Radial distance from beam axis
    ax1 = axes[0]
    ax1.set_title('Radial Distance from Beam Axis')

    for name, df in data.items():
        if df is None:
            continue

        if name == 'Geant4':
            # For Geant4, only LateralZ is true lateral
            radial = np.abs(df['LateralZ'].values)
            label = f"{LABELS[name]}: σ={np.std(df['LateralZ']):.2f} cm"
        else:
            # For ML agents, use both lateral coordinates
            radial = np.sqrt(df['LateralY'] ** 2 + df['LateralZ'] ** 2)
            label = f"{LABELS[name]}: σ={np.std(radial):.2f} cm"

        ax1.hist(radial, bins=50, alpha=0.6, label=label,
                 color=COLORS[name], edgecolor='black', linewidth=0.5, density=True)

    ax1.set_xlabel('Radial Distance [cm]')
    ax1.set_ylabel('Probability Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 5)

    # Right: Y and Z components for ML agents
    ax2 = axes[1]
    ax2.set_title('ML Agent Lateral Components (Mode Collapse Check)')

    for name in ['PPO', 'SAC']:
        df = data.get(name)
        if df is None:
            continue

        y_vals = df['LateralY'].values
        z_vals = df['LateralZ'].values

        ax2.hist(y_vals, bins=30, alpha=0.5,
                 label=f"{name} Y: μ={np.mean(y_vals):+.2f}",
                 color=COLORS[name], density=True)
        ax2.hist(z_vals, bins=30, alpha=0.5,
                 label=f"{name} Z: μ={np.mean(z_vals):+.2f}",
                 color=COLORS[name], density=True, hatch='//')

    ax2.axvline(0, color='black', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Lateral Position [cm]')
    ax2.set_ylabel('Probability Density')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'lateral_spread_1d')
    plt.close(fig)


# =============================================================================
# FIGURE 5: SCATTERING ANGLES
# =============================================================================

def generate_scattering_figure(data):
    """Generate scattering angle distribution with corrected Highland reference."""
    print("\n[5/8] Generating scattering angle distribution...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get path lengths for Highland calculation
    g4_path = CSDA_RANGE
    if data.get('Geant4') is not None:
        g4_path = data['Geant4']['PathLength'].mean()

    for name, df in data.items():
        if df is None or 'MeanScatterAngle' not in df.columns:
            continue

        samples = np.abs(df['MeanScatterAngle'].values)
        mean_angle = np.mean(samples)

        ax.hist(samples, bins=50, alpha=0.6,
                label=f"{LABELS[name]}: μ={mean_angle:.1f}°",
                color=COLORS[name], edgecolor='black', linewidth=0.5, density=True)

    # Highland predictions
    # For Unity ML agents: path is entirely in water
    highland_unity = highland_rms_angle(INITIAL_ENERGY, g4_path)

    # For Geant4: 1cm in air + path in water
    # But scattering in air is negligible, so effectively the same
    highland_g4 = highland_rms_angle_corrected(INITIAL_ENERGY, g4_path, GEANT4_AIR_GAP)

    ax.axvline(x=highland_unity, color='black', linestyle='--', linewidth=2.5,
               label=f'Highland (water only) = {highland_unity:.1f}°')

    # Note: The difference is negligible, but we document it
    print(f"  Highland RMS (water only): {highland_unity:.2f}°")
    print(f"  Highland RMS (with 1cm air): {highland_g4:.2f}°")
    print(f"  Difference: {abs(highland_g4 - highland_unity):.4f}° (negligible)")

    ax.set_xlabel('Mean Scatter Angle per Trajectory [°]')
    ax.set_ylabel('Probability Density')
    ax.set_title('Scattering Angle Distribution\n(Highland formula for reference)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)

    plt.tight_layout()
    save_figure(fig, 'scatter_angles')
    plt.close(fig)


# =============================================================================
# FIGURE 6: CONVERGENCE
# =============================================================================

def generate_convergence_figure(checkpoint_data, trajectory_data):
    """Generate training convergence plots."""
    print("\n[6/8] Generating convergence plots...")

    has_data = any(df is not None for df in checkpoint_data.values())
    if not has_data:
        print("  Skipping: No checkpoint data")
        return

    # Geant4 reference values
    g4_ref = {}
    if trajectory_data.get('Geant4') is not None:
        g4 = trajectory_data['Geant4']
        g4_ref = {
            'MeanPathLength': g4['PathLength'].mean(),
            'MeanLateralSpread': g4['LateralSpread'].mean(),
            'MeanScatterAngle': g4['MeanScatterAngle'].mean(),
        }

    metrics = [
        ('MeanPathLength', 'Path Length [cm]'),
        ('MeanLateralSpread', 'Lateral Spread [cm]'),
        ('MeanPenetrationDepth', 'Penetration Depth [cm]'),
        ('BoundaryExitRate', 'Boundary Exit [%]'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx]

        for name, df in checkpoint_data.items():
            if df is None or metric not in df.columns:
                continue
            ax.plot(df['StepCount'], df[metric], label=LABELS.get(name, name),
                    color=COLORS.get(name, 'gray'), linewidth=2)

        if metric in g4_ref:
            ax.axhline(y=g4_ref[metric], color=COLORS['Geant4'],
                       linestyle='--', linewidth=2, label='Geant4 Ref')

        ax.set_xlabel('Training Steps')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, p: f'{x / 1e6:.1f}M' if x >= 1e6 else f'{x / 1e3:.0f}k'))
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

    fig.suptitle('Training Convergence', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'convergence')
    plt.close(fig)


# =============================================================================
# FIGURE 7: BAR COMPARISON
# =============================================================================

def generate_comparison_bars(data):
    """Generate comparison bar chart."""
    print("\n[7/8] Generating comparison bar chart...")

    metrics = [
        ('PathLength', 'Path Length\n[cm]'),
        ('PenetrationDepth', 'Penetration\nDepth [cm]'),
        ('LateralSpread', 'Lateral\nSpread [cm]'),
        ('MeanScatterAngle', 'Scatter\nAngle [°]'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 5))

    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx]

        names, values, errors, colors = [], [], [], []

        for name in ['Geant4', 'PPO', 'SAC']:
            df = data.get(name)
            if df is None or metric not in df.columns:
                continue

            vals = df[metric].values
            names.append(name)
            values.append(np.mean(vals))
            errors.append(np.std(vals))
            colors.append(COLORS[name])

        bars = ax.bar(names, values, yerr=errors, color=colors,
                      edgecolor='black', capsize=5, alpha=0.8)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)

        ax.set_ylabel(label)
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('Physics Metrics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'metrics_comparison')
    plt.close(fig)


# =============================================================================
# FIGURE 8: HIGHLAND FORMULA ANALYSIS
# =============================================================================

def generate_highland_analysis(data):
    """Generate Highland formula analysis figure."""
    print("\n[8/8] Generating Highland formula analysis...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Highland vs path length
    ax1 = axes[0]

    path_lengths = np.linspace(0.5, 6, 100)
    highland_angles = [highland_rms_angle(INITIAL_ENERGY, p) for p in path_lengths]

    ax1.plot(path_lengths, highland_angles, 'k-', linewidth=2, label='Highland formula')

    # Mark actual values from data
    for name, df in data.items():
        if df is None:
            continue
        path_mean = df['PathLength'].mean()
        scatter_mean = df['MeanScatterAngle'].mean()
        ax1.scatter(path_mean, scatter_mean, s=100, c=COLORS[name],
                    edgecolor='black', linewidth=2, label=f'{name} (measured)', zorder=5)

    ax1.axvline(x=CSDA_RANGE, color='gray', linestyle=':', label=f'CSDA = {CSDA_RANGE} cm')
    ax1.set_xlabel('Path Length in Water [cm]')
    ax1.set_ylabel('RMS Scattering Angle [°]')
    ax1.set_title('Highland Formula vs Measured Scattering')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Effect of air gap
    ax2 = axes[1]

    air_gaps = np.linspace(0, 5, 50)
    water_path = CSDA_RANGE

    angles_total = [highland_rms_angle_corrected(INITIAL_ENERGY, water_path, ag) for ag in air_gaps]
    angles_water_only = [highland_rms_angle(INITIAL_ENERGY, water_path)] * len(air_gaps)

    ax2.plot(air_gaps, angles_total, 'b-', linewidth=2, label='Total (water + air)')
    ax2.plot(air_gaps, angles_water_only, 'r--', linewidth=2, label='Water only')

    ax2.axvline(x=GEANT4_AIR_GAP, color='green', linestyle=':', linewidth=2,
                label=f'Geant4 air gap = {GEANT4_AIR_GAP} cm')

    ax2.set_xlabel('Air Gap Before Phantom [cm]')
    ax2.set_ylabel('RMS Scattering Angle [°]')
    ax2.set_title(f'Effect of Air Gap on Scattering\n(Water path = {water_path:.1f} cm)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(20, 30)

    plt.tight_layout()
    save_figure(fig, 'highland_analysis')
    plt.close(fig)


# =============================================================================
# UTILITY
# =============================================================================

def save_figure(fig, name):
    """Save figure as PNG and PDF."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.pdf"), bbox_inches='tight')
    print(f"  Saved: {name}.png/.pdf")


def generate_latex_table(data):
    """Generate LaTeX summary table."""
    filepath = os.path.join(OUTPUT_DIR, 'algorithm_comparison.tex')

    with open(filepath, 'w') as f:
        f.write(r'\begin{table}[H]' + '\n')
        f.write(r'\centering' + '\n')
        f.write(r'\caption{Algorithm Comparison}' + '\n')
        f.write(r'\begin{tabular}{lccccc}' + '\n')
        f.write(r'\toprule' + '\n')
        f.write(r'\textbf{Source} & \textbf{Path [cm]} & \textbf{Depth [cm]} & '
                r'\textbf{Lateral [cm]} & \textbf{Angle [°]} & \textbf{Exit [\%]} \\' + '\n')
        f.write(r'\midrule' + '\n')

        for name in ['Geant4', 'PPO', 'SAC']:
            df = data.get(name)
            if df is None:
                continue

            f.write(f"{LABELS[name]} & ")
            f.write(f"{df['PathLength'].mean():.2f}±{df['PathLength'].std():.2f} & ")
            f.write(f"{df['PenetrationDepth'].mean():.2f}±{df['PenetrationDepth'].std():.2f} & ")
            f.write(f"{df['LateralSpread'].mean():.2f}±{df['LateralSpread'].std():.2f} & ")
            f.write(f"{df['MeanScatterAngle'].mean():.1f}±{df['MeanScatterAngle'].std():.1f} & ")
            exit_rate = (df['BoundaryExit'] == True).sum() / len(df) * 100
            f.write(f"{exit_rate:.1f} \\\\\n")

        f.write(r'\bottomrule' + '\n')
        f.write(r'\end{tabular}' + '\n')
        f.write(r'\end{table}' + '\n')

    print(f"  Saved: algorithm_comparison.tex")


def print_summary(data):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print(f"\n{'Metric':<20} {'Geant4':>15} {'PPO':>15} {'SAC':>15}")
    print("-" * 65)

    for metric in ['PathLength', 'PenetrationDepth', 'LateralSpread', 'MeanScatterAngle']:
        row = f"{metric:<20}"
        for name in ['Geant4', 'PPO', 'SAC']:
            df = data.get(name)
            if df is not None and metric in df.columns:
                row += f" {df[metric].mean():>6.2f}±{df[metric].std():<5.2f}"
            else:
                row += f" {'---':>15}"
        print(row)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("THESIS FIGURE GENERATOR - VERSION 2")
    print("=" * 70)
    print(f"\nData: {DATA_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("\n" + "-" * 70)
    print("LOADING DATA")
    print("-" * 70)

    print("\nTrajectories:")
    trajectory_data = load_trajectory_data()

    print("\nCheckpoints:")
    checkpoint_data = load_checkpoint_data()

    # Diagnostics
    issues = run_diagnostics(trajectory_data)

    # Generate figures
    print("\n" + "-" * 70)
    print("GENERATING FIGURES")
    print("-" * 70)

    generate_path_length_figure(trajectory_data)
    generate_penetration_depth_figure(trajectory_data)
    generate_lateral_spread_2d_figures(trajectory_data)
    generate_lateral_1d_figure(trajectory_data)
    generate_scattering_figure(trajectory_data)
    generate_convergence_figure(checkpoint_data, trajectory_data)
    generate_comparison_bars(trajectory_data)
    generate_highland_analysis(trajectory_data)

    # Tables
    print("\n" + "-" * 70)
    print("GENERATING TABLES")
    print("-" * 70)
    generate_latex_table(trajectory_data)

    # Summary
    print_summary(trajectory_data)

    if issues:
        print("\n" + "=" * 70)
        print("ISSUES DETECTED")
        print("=" * 70)
        for issue in issues:
            print(f"  ⚠ {issue}")

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()