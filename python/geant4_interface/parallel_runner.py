"""
Parallel Geant4 Runner
Runs multiple Geant4 simulations in parallel using multiprocessing
"""

import multiprocessing as mp
from multiprocessing import Pool
from typing import List, Dict, Any
import logging
from pathlib import Path
import time
import os
from tqdm import tqdm

from .simulation_runner import Geant4SimulationRunner

logger = logging.getLogger(__name__)


class ParallelGeant4Runner:
    """
    Manages parallel execution of Geant4 simulations
    """

    def __init__(self,
                 geant4_executable: str,
                 output_directory: str = "./parallel_geant4_runs",
                 num_workers: int = None,
                 timeout: int = 120):
        """
        Initialize parallel runner

        Args:
            geant4_executable: Path to Geant4 executable
            output_directory: Base directory for outputs
            num_workers: Number of parallel workers (None = CPU count)
            timeout: Timeout per simulation in seconds
        """
        self.geant4_executable = geant4_executable
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        # Determine number of workers
        if num_workers is None:
            self.num_workers = mp.cpu_count()
        else:
            self.num_workers = min(num_workers, mp.cpu_count())

        self.timeout = timeout

        logger.info(f"Parallel Geant4 runner initialized")
        logger.info(f"  Workers: {self.num_workers}")
        logger.info(f"  Executable: {self.geant4_executable}")
        logger.info(f"  Output dir: {self.output_directory}")

    def run_batch(self,
                  parameters_list: List[Dict[str, Any]],
                  show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Run batch of simulations in parallel

        Args:
            parameters_list: List of parameter dictionaries
            show_progress: Show progress bar

        Returns:
            List of result dictionaries
        """
        num_simulations = len(parameters_list)

        logger.info(f"Running batch of {num_simulations} simulations")
        logger.info(f"  Parallel workers: {self.num_workers}")
        logger.info(f"  Expected time: ~{num_simulations * 2 / self.num_workers:.1f}s")

        start_time = time.time()

        # Create worker arguments
        worker_args = []
        for i, params in enumerate(parameters_list):
            # Create unique output directory for each simulation
            output_dir = self.output_directory / f"sim_{i:06d}"

            worker_args.append({
                'executable': self.geant4_executable,
                'params': params,
                'output_dir': str(output_dir),
                'timeout': self.timeout,
                'sim_id': i
            })

        # Run simulations in parallel
        try:
            with Pool(processes=self.num_workers) as pool:
                if show_progress:
                    results = []
                    with tqdm(total=num_simulations, desc="Simulations", unit="sim") as pbar:
                        for result in pool.imap(_run_single_simulation, worker_args):
                            results.append(result)
                            pbar.update(1)
                else:
                    results = pool.map(_run_single_simulation, worker_args)

        except KeyboardInterrupt:
            logger.warning("Batch interrupted by user")
            pool.terminate()
            pool.join()
            raise

        elapsed_time = time.time() - start_time

        # Count successes and failures
        successful = sum(1 for r in results if r['success'])
        failed = num_simulations - successful

        logger.info(f"Batch complete!")
        logger.info(f"  Total time: {elapsed_time:.2f}s")
        logger.info(f"  Time per sim: {elapsed_time / num_simulations:.2f}s")
        logger.info(f"  Successful: {successful}/{num_simulations}")
        logger.info(f"  Failed: {failed}/{num_simulations}")
        logger.info(f"  Speedup: {num_simulations * 2 / elapsed_time:.1f}x")

        return results

    def run_streaming_batch(self,
                            parameters_list: List[Dict[str, Any]],
                            callback=None):
        """
        Run batch with streaming results (process as they complete)

        Args:
            parameters_list: List of parameter dictionaries
            callback: Function to call with each result

        Yields:
            Results as they complete
        """
        num_simulations = len(parameters_list)

        logger.info(f"Running streaming batch of {num_simulations} simulations")

        # Create worker arguments
        worker_args = []
        for i, params in enumerate(parameters_list):
            output_dir = self.output_directory / f"sim_{i:06d}"
            worker_args.append({
                'executable': self.geant4_executable,
                'params': params,
                'output_dir': str(output_dir),
                'timeout': self.timeout,
                'sim_id': i
            })

        # Process results as they complete
        with Pool(processes=self.num_workers) as pool:
            for result in pool.imap_unordered(_run_single_simulation, worker_args):
                if callback:
                    callback(result)
                yield result


def _run_single_simulation(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function to run a single simulation
    This runs in a separate process

    Args:
        args: Dictionary with:
            - executable: Path to Geant4 executable
            - params: Simulation parameters
            - output_dir: Output directory
            - timeout: Timeout in seconds
            - sim_id: Simulation ID

    Returns:
        Result dictionary
    """
    executable = args['executable']
    params = args['params']
    output_dir = args['output_dir']
    timeout = args['timeout']
    sim_id = args['sim_id']

    # Create runner for this worker
    runner = Geant4SimulationRunner(
        geant4_executable=executable,
        output_directory=output_dir,
        timeout=timeout
    )

    # Run simulation
    try:
        result = runner.run_simulation(params)
        result['sim_id'] = sim_id
        return result

    except Exception as e:
        logger.error(f"Simulation {sim_id} failed: {e}")
        return {
            'success': False,
            'sim_id': sim_id,
            'error': str(e),
            'parameters': params
        }


class BatchStatistics:
    """Calculate and display batch statistics"""

    def __init__(self):
        self.results = []

    def add_result(self, result: Dict[str, Any]):
        """Add a result to statistics"""
        self.results.append(result)

    def print_summary(self):
        """Print summary statistics"""
        if not self.results:
            print("No results to summarize")
            return

        total = len(self.results)
        successful = sum(1 for r in self.results if r['success'])
        failed = total - successful

        if successful > 0:
            energies = [r['total_energy_deposit'] for r in self.results if r['success']]
            import numpy as np

            print("\n" + "=" * 70)
            print("BATCH STATISTICS")
            print("=" * 70)
            print(f"Total simulations: {total}")
            print(f"Successful: {successful} ({successful / total * 100:.1f}%)")
            print(f"Failed: {failed} ({failed / total * 100:.1f}%)")
            print()
            print("Energy Deposition:")
            print(f"  Mean:   {np.mean(energies):.3f} MeV")
            print(f"  Std:    {np.std(energies):.3f} MeV")
            print(f"  Min:    {np.min(energies):.3f} MeV")
            print(f"  Max:    {np.max(energies):.3f} MeV")
            print("=" * 70 + "\n")