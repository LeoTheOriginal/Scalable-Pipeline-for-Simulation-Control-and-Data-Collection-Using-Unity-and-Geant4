#!/usr/bin/env python3
"""
Performance Comparison Visualization for Engineering Thesis

Generates publication-quality comparison figures showing:
1. Computational cost (time per trajectory)
2. Throughput (trajectories per second)
3. Speedup factor (as annotation)

Reads benchmark data from Unity JSON export or manual input.

Author: Engineering Thesis - Electron Transport RL Surrogate Model
Date: 2024
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import argparse
from pathlib import Path
import os

# --- KONFIGURACJA STATYCZNA ---
DEFAULT_JSON_PATH = r"C:\Thesis\unity\GeantML_Test\Assets\performance_benchmark.json"
DEFAULT_OUTPUT_PATH = r"C:\Thesis\python\figures\motivation_comparison.png"


# ------------------------------

class PerformanceVisualizer:
    """Generate performance comparison visualizations."""

    def __init__(self, config=None):
        """
        Initialize visualizer with optional configuration.
        """
        self.config = config or self._default_config()

        # Set publication-quality defaults
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 13
        plt.rcParams['axes.titlesize'] = 15
        plt.rcParams['xtick.labelsize'] = 11
        plt.rcParams['ytick.labelsize'] = 11
        plt.rcParams['legend.fontsize'] = 11
        plt.rcParams['figure.titlesize'] = 18

    def _default_config(self):
        """Default visualization configuration."""
        return {
            'colors': {
                'geant4': '#e74c3c',  # Red
                'ml_agent': '#2ecc71',  # Green
                'speedup': '#f39c12',  # Orange/Yellow
                'speedup_bg': '#fff9c4'  # Light yellow for text box
            },
            'figure_size': (12, 6),
            'dpi': 300,
            'style': 'seaborn-v0_8-darkgrid'
        }

    def load_benchmark_data(self, json_path):
        """Load benchmark results from Unity JSON export."""
        with open(json_path, 'r') as f:
            data = json.load(f)

        return {
            'geant4_time_ms': data['Geant4']['MeanTimeMs'],
            'geant4_std_ms': data['Geant4']['StdDevMs'],
            'ml_time_ms': data['MLAgent']['MeanTimeMs'],
            'ml_std_ms': data['MLAgent']['StdDevMs'],
            'speedup': data['SpeedupFactor'],
        }

    def create_comparison_figure(
            self,
            geant4_time_ms,
            ml_time_ms,
            geant4_std_ms=0,
            ml_std_ms=0,
            speedup=None,
            output_path='motivation_comparison.png',
            show_error_bars=True,
            title_suffix=""
    ):
        """
        Create comprehensive performance comparison figure (2 Subplots + Speedup Banner).
        """
        if speedup is None:
            speedup = geant4_time_ms / ml_time_ms

        # Calculate FPS
        geant4_fps = 1000.0 / geant4_time_ms
        ml_fps = 1000.0 / ml_time_ms

        # Create figure with TWO subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config['figure_size'])

        methods = ['Geant4\nMonte Carlo', 'ML Agent\n(Trained Model)']
        times = [geant4_time_ms, ml_time_ms]
        times_std = [geant4_std_ms, ml_std_ms]
        fps_values = [geant4_fps, ml_fps]
        colors = [self.config['colors']['geant4'], self.config['colors']['ml_agent']]

        # ================================================================
        # SUBPLOT 1: Computational Cost
        # ================================================================
        bars1 = ax1.bar(
            methods, times,
            color=colors,
            alpha=0.85,
            edgecolor='black',
            linewidth=1.5,
            yerr=times_std if show_error_bars else None,
            capsize=6,
            error_kw={'linewidth': 1.5, 'ecolor': 'black', 'alpha': 0.8}
        )

        ax1.set_ylabel('Time per Trajectory (ms)', fontweight='bold')
        ax1.set_title('Computational Cost Comparison', fontweight='bold', pad=15)

        max_height_with_err = 0
        for t, s in zip(times, times_std):
            current_top = t + (s if show_error_bars else 0)
            if current_top > max_height_with_err:
                max_height_with_err = current_top

        ax1.set_ylim(0, max_height_with_err * 1.4)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        for bar, time, std in zip(bars1, times, times_std):
            height = bar.get_height()
            label = f'{time:.4f} ms'
            if show_error_bars and std > 0:
                label += f'\n±{std:.4f}'

            y_pos = height + (std if show_error_bars else 0)

            ax1.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2., y_pos),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=11, fontweight='bold'
            )

        # ================================================================
        # SUBPLOT 2: Throughput (FPS)
        # ================================================================
        bars2 = ax2.bar(
            methods, fps_values,
            color=colors,
            alpha=0.85,
            edgecolor='black',
            linewidth=1.5
        )

        ax2.set_ylabel('Trajectories per Second', fontweight='bold')
        ax2.set_title('Throughput Comparison', fontweight='bold', pad=15)
        ax2.set_ylim(0, max(fps_values) * 1.3)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        for bar, fps in zip(bars2, fps_values):
            height = bar.get_height()
            if fps > 1000000:
                fps_label = f'{fps / 1000000:.2f}M FPS'
            elif fps > 1000:
                fps_label = f'{fps / 1000:.1f}k FPS'
            else:
                fps_label = f'{fps:.1f} FPS'

            ax2.annotate(
                fps_label,
                xy=(bar.get_x() + bar.get_width() / 2., height),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=11, fontweight='bold'
            )

        # ================================================================
        # SPEEDUP BANNER (BOTTOM)
        # ================================================================

        speedup_text = f"Speedup: {speedup:.1f}× faster"

        # Opuściliśmy tekst niżej (0.05), bo dodamy padding przy zapisie
        fig.text(
            0.5, 0.05,
            speedup_text,
            ha='center', va='center',
            fontsize=16, fontweight='bold',
            color='black',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor=self.config['colors']['speedup_bg'],
                edgecolor='black',
                linewidth=1.5
            )
        )

        # ================================================================
        # FORMATTING & SAVING
        # ================================================================

        fig.suptitle(f'Geant4 vs ML Agent Performance{title_suffix}',
                     fontsize=18, fontweight='bold', y=0.96)

        # Zmniejszono nieco bottom margin (z 0.30 na 0.24), bo tekst jest niżej
        plt.subplots_adjust(top=0.82, bottom=0.24, left=0.08, right=0.92, wspace=0.25)

        # KLUCZOWA ZMIANA: pad_inches=0.2 dodaje margines dookoła przyciętego obrazka
        plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight', pad_inches=0.2)
        print(f"✓ Figure saved to: {output_path}")

        pdf_path = output_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, dpi=self.config['dpi'], bbox_inches='tight', pad_inches=0.2, format='pdf')
        print(f"✓ PDF version saved to: {pdf_path}")

        return fig


def main():
    """Command-line interface for generating figures."""
    parser = argparse.ArgumentParser(
        description='Generate performance comparison figures for thesis'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help='Path to performance_benchmark.json from Unity'
    )
    parser.add_argument(
        '--geant4-time', '-g',
        type=float,
        help='Geant4 time in ms (manual input)'
    )
    parser.add_argument(
        '--ml-time', '-m',
        type=float,
        help='ML agent time in ms (manual input)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help='Output file path'
    )
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Generate simplified version'
    )

    args = parser.parse_args()
    visualizer = PerformanceVisualizer()

    # Logika wyboru pliku wejściowego
    input_path = args.input
    if input_path is None and os.path.exists(DEFAULT_JSON_PATH):
        input_path = DEFAULT_JSON_PATH

    # --- GLOWNA LOGIKA ---
    if input_path and os.path.exists(input_path):
        print(f"Loading benchmark data from: {input_path}")
        data = visualizer.load_benchmark_data(input_path)

        visualizer.create_comparison_figure(
            data['geant4_time_ms'],
            data['ml_time_ms'],
            data['geant4_std_ms'],
            data['ml_std_ms'],
            data['speedup'],
            args.output,
            show_error_bars=True,
            title_suffix=""
        )

    elif args.geant4_time and args.ml_time:
        print(f"Using manual input: Geant4={args.geant4_time}ms, ML={args.ml_time}ms")
        visualizer.create_comparison_figure(
            args.geant4_time,
            args.ml_time,
            output_path=args.output
        )

    else:
        print(f"WARNING: Default file not found at: {DEFAULT_JSON_PATH}")
        print("Generating example figure...")
        visualizer.create_comparison_figure(
            geant4_time_ms=450.0,
            ml_time_ms=4.5,
            geant4_std_ms=45.0,
            ml_std_ms=0.8,
            output_path=args.output,
            title_suffix=" (Example Data)"
        )


if __name__ == '__main__':
    main()