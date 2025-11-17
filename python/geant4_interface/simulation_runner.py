"""
Geant4 Simulation Runner - FIXED VERSION
Handles execution and control of Geant4 physics simulations
Added: Direction normalization and comprehensive logging
"""

import subprocess
import os
import logging
from typing import Dict, Any, Optional, List
import json
from pathlib import Path
import numpy as np


logger = logging.getLogger(__name__)


class Geant4SimulationRunner:
    """
    Manages Geant4 simulation execution and parameter control
    """

    def __init__(self,
                 geant4_executable: Optional[str] = None,
                 output_directory: str = "./geant4_output",
                 timeout: int = 120,
                 keep_temp_files: bool = True):
        """
        Initialize Geant4 simulation runner

        Args:
            geant4_executable: Path to Geant4 executable
            output_directory: Directory for output files
            timeout: Timeout in seconds for each simulation (default: 120)
            keep_temp_files: Keep temporary files after simulation (default: True)
        """
        self.geant4_executable = geant4_executable
        self.output_directory = output_directory
        self.timeout = timeout
        self.keep_temp_files = keep_temp_files

        # Create output directory if it doesn't exist
        Path(output_directory).mkdir(parents=True, exist_ok=True)

        logger.info(f"Geant4 runner initialized")
        if geant4_executable:
            logger.info(f"  Executable: {geant4_executable}")
        logger.info(f"  Output dir: {output_directory}")
        logger.info(f"  Timeout: {timeout}s")

    def _normalize_direction(self, direction: List[float]) -> List[float]:
        """
        Normalize direction vector to unit length

        Args:
            direction: Direction vector [x, y, z]

        Returns:
            Normalized direction vector
        """
        direction_array = np.array(direction)
        magnitude = np.linalg.norm(direction_array)

        if magnitude < 1e-10:
            logger.warning("Direction vector has zero magnitude, using default [1, 0, 0]")
            return [1.0, 0.0, 0.0]

        normalized = direction_array / magnitude
        return normalized.tolist()

    def run_simulation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Geant4 simulation with given parameters

        Args:
            parameters: Simulation parameters
                - particle_type: str (e.g., 'e-', 'gamma')
                - particle_energy: float (MeV)
                - particle_position: [x, y, z] (cm)
                - particle_direction: [dx, dy, dz] (will be normalized)
                - num_events: int (default: 1)

        Returns:
            dict: Simulation results with energy deposit and trajectory
        """
        if not self.geant4_executable or not Path(self.geant4_executable).exists():
            raise FileNotFoundError(f"Geant4 executable not found: {self.geant4_executable}")

        # Normalize direction vector
        if 'particle_direction' in parameters:
            parameters['particle_direction'] = self._normalize_direction(
                parameters['particle_direction']
            )

        # Create temporary directory for this run
        import time
        run_id = f"run_{int(time.time() * 1000)}"
        temp_dir = Path(self.output_directory) / run_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Generate macro file
            macro_content = self._generate_macro(parameters)
            macro_file = temp_dir / "run.mac"
            with open(macro_file, 'w') as f:
                f.write(macro_content)

            # Set output directory environment variable
            env = os.environ.copy()
            env['G4_OUTPUT_DIR'] = str(temp_dir)

            # Log simulation parameters
            logger.info(f"Running Geant4 simulation:")
            logger.info(f"  Executable: {self.geant4_executable}")
            logger.info(f"  Particle: {parameters.get('particle_type', 'e-')}")
            logger.info(f"  Energy: {parameters.get('particle_energy', 10.0)} MeV")
            logger.info(f"  Position: {parameters.get('particle_position', [-6, 0, 0])}")
            logger.info(f"  Direction (normalized): {parameters.get('particle_direction', [1, 0, 0])}")
            logger.info(f"  Events: {parameters.get('num_events', 1)}")
            logger.info(f"  Output directory: {temp_dir}")
            logger.info(f"  Macro file: {macro_file}")

            # Run Geant4
            result = subprocess.run(
                [self.geant4_executable, str(macro_file)],
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            # ALWAYS log Geant4 output
            logger.info("=" * 60)
            logger.info("GEANT4 STDOUT:")
            logger.info("=" * 60)
            if result.stdout:
                for line in result.stdout.split('\n')[:50]:  # First 50 lines
                    logger.info(line)
            else:
                logger.info("(no output)")
            logger.info("=" * 60)

            if result.stderr:
                logger.warning("=" * 60)
                logger.warning("GEANT4 STDERR:")
                logger.warning("=" * 60)
                for line in result.stderr.split('\n'):
                    logger.warning(line)
                logger.warning("=" * 60)

            if result.returncode != 0:
                logger.error(f"Geant4 execution failed with return code {result.returncode}")
                return {
                    'success': False,
                    'error': f"Return code {result.returncode}",
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }

            # Check what files were created
            logger.info(f"Checking output directory: {temp_dir}")
            all_files = list(temp_dir.rglob('*'))
            logger.info(f"Files created: {len([f for f in all_files if f.is_file()])}")
            for f in all_files:
                if f.is_file():
                    logger.info(f"  - {f.relative_to(temp_dir)} ({f.stat().st_size} bytes)")

            # Find CSV files
            csv_files = list(temp_dir.rglob('*.csv'))
            logger.info(f"CSV files found: {len(csv_files)}")

            if len(csv_files) == 0:
                logger.error("No CSV files found!")
                logger.error("This means:")
                logger.error("  - Geant4 didn't write any output files")
                logger.error("  - Or particle didn't enter phantom volume")
                logger.error("  - Or G4_OUTPUT_DIR environment variable wasn't set")

                return {
                    'success': False,
                    'error': 'No CSV output files generated',
                    'output_directory': str(temp_dir),
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'parameters': parameters
                }

            # Parse output CSV files
            from .output_parser import Geant4OutputParser
            parser = Geant4OutputParser()
            events = parser.parse_batch_results(str(temp_dir))

            if not events:
                logger.error("CSV files found but parsing failed")
                return {
                    'success': False,
                    'error': 'Failed to parse output files',
                    'output_directory': str(temp_dir),
                    'csv_files_found': [str(f) for f in csv_files],
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }

            # Aggregate results
            energy_deposits = [e['total_energy_deposit'] for e in events]
            result_data = {
                'success': True,
                'total_energy_deposit': np.mean(energy_deposits),
                'energy_deposit_std': np.std(energy_deposits) if len(energy_deposits) > 1 else 0.0,
                'num_events': len(events),
                'events': events,
                'parameters': parameters,
                'output_directory': str(temp_dir)
            }

            logger.info(f"✅ Simulation complete!")
            logger.info(f"   Energy deposited: {result_data['total_energy_deposit']:.3f} MeV")
            logger.info(f"   Events processed: {result_data['num_events']}")

            return result_data

        except subprocess.TimeoutExpired:
            logger.error(f"Simulation timeout after {self.timeout}s")
            return {
                'success': False,
                'error': 'Timeout'
            }
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }

    def _generate_macro(self, parameters: Dict[str, Any]) -> str:
        """Generate Geant4 macro file from parameters"""

        # Default values
        particle_type = parameters.get('particle_type', 'e-')
        particle_energy = parameters.get('particle_energy', 10.0)
        pos = parameters.get('particle_position', [-6, 0, 0])
        direction = parameters.get('particle_direction', [1, 0, 0])
        num_events = parameters.get('num_events', 1)

        macro = f"""# Auto-generated macro
/run/initialize

# Particle configuration
/gun/particle {particle_type}
/gun/energy {particle_energy} MeV
/gun/position {pos[0]} {pos[1]} {pos[2]} cm
/gun/direction {direction[0]} {direction[1]} {direction[2]}

# Run simulation
/run/beamOn {num_events}
"""

        return macro

    def run_batch_simulations(self,
                              parameter_sets: List[Dict[str, Any]],
                              max_parallel: int = 1) -> List[Dict[str, Any]]:
        """
        Run multiple simulations with different parameters

        Args:
            parameter_sets: List of parameter dictionaries
            max_parallel: Maximum number of parallel simulations

        Returns:
            list: List of simulation results
        """
        results = []

        if max_parallel == 1:
            # Sequential execution
            for params in parameter_sets:
                result = self.run_simulation(params)
                results.append(result)
        else:
            # Parallel execution (basic implementation)
            # TODO: Implement proper parallel execution with multiprocessing
            logger.warning("Parallel execution not yet implemented, running sequentially")
            for params in parameter_sets:
                result = self.run_simulation(params)
                results.append(result)

        return results

    def cleanup(self, keep_outputs: bool = True):
        """
        Clean up temporary files

        Args:
            keep_outputs: If True, keep output files
        """
        logger.info("Cleaning up simulation files")

        if not keep_outputs:
            # Remove output files
            output_path = Path(self.output_directory)
            if output_path.exists():
                import shutil
                shutil.rmtree(output_path)

    def get_simulation_status(self, simulation_id: int) -> Dict[str, Any]:
        """
        Get status of a simulation

        Args:
            simulation_id: Simulation identifier

        Returns:
            dict: Simulation status information
        """
        output_file = self.output_directory / f"output_{simulation_id}.root"

        return {
            'simulation_id': simulation_id,
            'output_exists': output_file.exists(),
            'output_path': str(output_file) if output_file.exists() else None
        }