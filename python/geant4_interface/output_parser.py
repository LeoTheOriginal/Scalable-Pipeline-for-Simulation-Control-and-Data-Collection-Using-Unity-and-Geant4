"""
Geant4 Output Parser
Parse CSV output files from WaterPhantomSim
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class Geant4OutputParser:
    """Parse Geant4 CSV output files"""

    def __init__(self):
        """Initialize parser"""
        pass

    def parse_event_file(self, filepath: str) -> Dict:
        """
        Parse single event CSV file

        Args:
            filepath: Path to event CSV file

        Returns:
            dict: Parsed event data with summary and steps
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Parse summary section
            summary = {}
            step_data_start_idx = -1

            for idx, line in enumerate(lines):
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Find step data section
                if line.startswith('# Step Data'):
                    step_data_start_idx = idx
                    continue

                # Parse summary data (before step data)
                if step_data_start_idx == -1 and ',' in line and not line.startswith('#'):
                    parts = line.split(',')
                    if len(parts) == 2:
                        key, value = parts
                        try:
                            summary[key] = float(value)
                        except ValueError:
                            summary[key] = value

            # Parse step data using pandas
            steps = []
            if step_data_start_idx != -1:
                # Find where actual data starts (after header)
                data_start = step_data_start_idx + 2  # Skip "# Step Data" and header line
                step_lines = [l for l in lines[data_start:] if l.strip() and not l.startswith('#')]

                if step_lines:
                    # Create temporary CSV-like string
                    header = "StepID,PosX_cm,PosY_cm,PosZ_cm,KineticEnergy_MeV,EnergyDeposited_MeV,ScatterAngle_deg,Acceleration,ProcessName"
                    csv_data = header + '\n' + ''.join(step_lines)

                    # Parse with pandas
                    from io import StringIO
                    df = pd.read_csv(StringIO(csv_data))

                    # Convert to list of dicts with proper structure for trajectory
                    for _, row in df.iterrows():
                        step_dict = {
                            'step_number': int(row['StepID']),
                            'position': [
                                float(row['PosX_cm']),
                                float(row['PosY_cm']),
                                float(row['PosZ_cm'])
                            ],
                            'direction': [1, 0, 0],  # Not available in CSV, use default
                            'energy': float(row['KineticEnergy_MeV']),
                            'energy_deposit': float(row['EnergyDeposited_MeV']),
                            'step_length': 0.0,  # Not available in CSV
                            'process': str(row['ProcessName'])
                        }
                        steps.append(step_dict)

            result = {
                'event_id': int(summary.get('EventID', 0)),
                'total_energy_deposit': summary.get('TotalEnergyDeposit', 0.0),
                'num_steps': int(summary.get('NumberOfSteps', 0)),
                'steps': steps,
                'success': True
            }

            logger.info(f"Parsed event {result['event_id']}: "
                        f"{result['total_energy_deposit']:.3f} MeV deposited "
                        f"in {result['num_steps']} steps")

            return result

        except Exception as e:
            logger.error(f"Error parsing file {filepath}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'filepath': filepath
            }

    def parse_batch_results(self, output_dir: str) -> List[Dict]:
        """
        Parse all event files in directory (including subdirectories)

        Args:
            output_dir: Directory containing event CSV files

        Returns:
            list: List of parsed events
        """
        output_path = Path(output_dir)
        if not output_path.exists():
            logger.error(f"Output directory does not exist: {output_dir}")
            return []

        # Search for event files in output_dir and subdirectories
        event_files = []

        # Search directly in output_dir
        direct_files = list(output_path.glob('event_*.csv'))
        event_files.extend(direct_files)

        # Search in run_* subdirectories (Geant4 creates timestamped folders)
        for run_dir in output_path.glob('run_*'):
            if run_dir.is_dir():
                run_files = list(run_dir.glob('event_*.csv'))
                event_files.extend(run_files)

        # Also search recursively for event_NNNNNN.csv pattern
        recursive_files = list(output_path.rglob('event_[0-9]*.csv'))
        event_files.extend(recursive_files)

        # Remove duplicates and sort
        event_files = list(set(event_files))
        event_files.sort()

        logger.info(f"Found {len(event_files)} event files in {output_dir}")

        if len(event_files) == 0:
            logger.warning(f"No event files found! Searched in:")
            logger.warning(f"  - {output_dir}/event_*.csv")
            logger.warning(f"  - {output_dir}/run_*/event_*.csv")
            logger.warning(f"  - {output_dir}/**/event_[0-9]*.csv (recursive)")

        results = []
        for filepath in event_files:
            logger.info(f"Parsing: {filepath}")
            result = self.parse_event_file(str(filepath))
            if result['success']:
                results.append(result)

        logger.info(f"Successfully parsed {len(results)}/{len(event_files)} events")
        return results

    def compute_statistics(self, results: List[Dict]) -> Dict:
        """
        Compute statistics from multiple events

        Args:
            results: List of parsed event results

        Returns:
            dict: Statistics summary
        """
        if not results:
            return {}

        energy_deposits = [r['total_energy_deposit'] for r in results]
        num_steps = [r['num_steps'] for r in results]

        stats = {
            'num_events': len(results),
            'energy_deposit': {
                'mean': np.mean(energy_deposits),
                'std': np.std(energy_deposits),
                'min': np.min(energy_deposits),
                'max': np.max(energy_deposits)
            },
            'num_steps': {
                'mean': np.mean(num_steps),
                'std': np.std(num_steps),
                'min': int(np.min(num_steps)),
                'max': int(np.max(num_steps))
            }
        }

        logger.info(f"Statistics: Energy deposit = {stats['energy_deposit']['mean']:.3f} ± "
                    f"{stats['energy_deposit']['std']:.3f} MeV")

        return stats


if __name__ == "__main__":
    # Quick test
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        parser = Geant4OutputParser()
        result = parser.parse_event_file(test_file)
        print(f"\n✅ Parsed successfully!")
        print(f"Event ID: {result['event_id']}")
        print(f"Total Energy: {result['total_energy_deposit']:.3f} MeV")
        print(f"Steps: {result['num_steps']}")
        if result['steps']:
            print(f"\nFirst 3 steps:")
            for step in result['steps'][:3]:
                print(f"  Step {step['step_number']}: pos={step['position']}, E={step['energy']:.3f} MeV")
    else:
        print("Usage: python output_parser.py <csv_file>")