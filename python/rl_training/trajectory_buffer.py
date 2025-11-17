"""
Trajectory Buffer - FIXED VERSION
Added: Direction correction to ensure particles hit phantom
"""

import numpy as np
from typing import List, Dict, Any, Callable
from pathlib import Path
import logging
from collections import deque
import threading
import time

from .trajectory_data import ParticleTrajectory, TrajectoryPair
from geant4_interface.parallel_runner import ParallelGeant4Runner

logger = logging.getLogger(__name__)


class TrajectoryBuffer:
    """
    Buffer for collecting and processing particle trajectories
    FIXED: Corrects direction vectors to ensure phantom hits
    """

    def __init__(self,
                 geant4_executable: str,
                 buffer_size: int = 1000,
                 num_workers: int = 8,
                 auto_process: bool = True,
                 correct_direction: bool = True,  # NEW: Enable direction correction
                 phantom_center: np.ndarray = np.array([0.0, 0.0, 0.0])):  # NEW
        """
        Initialize trajectory buffer

        Args:
            geant4_executable: Path to Geant4 executable
            buffer_size: Number of trajectories to buffer before processing
            num_workers: Number of parallel Geant4 workers
            auto_process: Automatically process when buffer is full
            correct_direction: If True, correct particle direction toward phantom
            phantom_center: Center position of water phantom [x, y, z] in cm
        """
        self.buffer_size = buffer_size
        self.auto_process = auto_process
        self.correct_direction = correct_direction
        self.phantom_center = phantom_center

        # Buffers
        self.unity_trajectories = deque()  # Trajectories from Unity
        self.trajectory_pairs = []  # Matched Unity-Geant4 pairs

        # Parallel Geant4 runner
        self.geant4_runner = ParallelGeant4Runner(
            geant4_executable=geant4_executable,
            output_directory="./trajectory_buffer_geant4",
            num_workers=num_workers
        )

        # Statistics
        self.total_collected = 0
        self.total_processed = 0
        self.next_trajectory_id = 0

        # Thread-safe lock
        self.lock = threading.Lock()

        logger.info(f"Trajectory buffer initialized")
        logger.info(f"  Buffer size: {buffer_size}")
        logger.info(f"  Workers: {num_workers}")
        logger.info(f"  Auto-process: {auto_process}")
        logger.info(f"  Direction correction: {correct_direction}")
        if correct_direction:
            logger.info(f"  Phantom center: {phantom_center}")

    def add_unity_trajectory(self, trajectory: ParticleTrajectory) -> int:
        """
        Add Unity trajectory to buffer

        Args:
            trajectory: Unity particle trajectory

        Returns:
            Trajectory ID
        """
        with self.lock:
            # Assign ID
            trajectory.trajectory_id = self.next_trajectory_id
            self.next_trajectory_id += 1

            # Add to buffer
            self.unity_trajectories.append(trajectory)
            self.total_collected += 1

            trajectory_id = trajectory.trajectory_id

        # Auto-process if buffer is full
        if self.auto_process and len(self.unity_trajectories) >= self.buffer_size:
            self.process_buffer()

        return trajectory_id

    def _correct_direction_to_phantom(self, position: np.ndarray) -> np.ndarray:
        """
        Calculate direction vector from position toward phantom center

        Args:
            position: Starting position [x, y, z]

        Returns:
            Normalized direction vector toward phantom
        """
        direction = self.phantom_center - position
        magnitude = np.linalg.norm(direction)

        if magnitude < 1e-10:
            logger.warning(f"Position {position} is at phantom center, using [1,0,0]")
            return np.array([1.0, 0.0, 0.0])

        return direction / magnitude

    def process_buffer(self) -> List[TrajectoryPair]:
        """
        Process buffered trajectories:
        1. Extract parameters from Unity trajectories
        2. Optionally correct direction toward phantom
        3. Run Geant4 simulations in parallel
        4. Create trajectory pairs
        5. Calculate rewards

        Returns:
            List of trajectory pairs with rewards
        """
        with self.lock:
            if len(self.unity_trajectories) == 0:
                logger.warning("Buffer is empty, nothing to process")
                return []

            # Get trajectories from buffer
            trajectories_to_process = list(self.unity_trajectories)
            self.unity_trajectories.clear()

            num_trajectories = len(trajectories_to_process)

        logger.info(f"Processing buffer: {num_trajectories} trajectories")
        start_time = time.time()

        # Prepare Geant4 parameters
        geant4_params = []
        for traj in trajectories_to_process:
            # Get direction - either corrected or original
            if self.correct_direction:
                # Calculate direction toward phantom
                corrected_direction = self._correct_direction_to_phantom(traj.initial_position)
                direction_to_use = corrected_direction

                logger.debug(f"Trajectory {traj.trajectory_id}:")
                logger.debug(f"  Position: {traj.initial_position}")
                logger.debug(f"  Original direction: {traj.initial_direction}")
                logger.debug(f"  Corrected direction: {corrected_direction}")
            else:
                # Use original Unity direction
                direction_to_use = traj.initial_direction

            params = {
                'particle_type': traj.particle_type,
                'particle_energy': traj.initial_energy,
                'particle_position': traj.initial_position.tolist(),
                'particle_direction': direction_to_use.tolist(),  # Corrected or original
                'num_events': 1,
                # Store original direction for comparison
                '_original_direction': traj.initial_direction.tolist(),
                '_direction_corrected': self.correct_direction
            }
            geant4_params.append(params)

        # Log sample parameters
        if geant4_params:
            sample = geant4_params[0]
            logger.info(f"Sample Geant4 parameters (first trajectory):")
            logger.info(f"  Position: {sample['particle_position']}")
            logger.info(f"  Direction: {sample['particle_direction']}")
            if self.correct_direction:
                logger.info(f"  Original direction: {sample['_original_direction']}")
                logger.info(f"  → Corrected to point at phantom")

        # Run Geant4 batch
        logger.info(f"Running Geant4 batch: {num_trajectories} simulations...")
        geant4_results = self.geant4_runner.run_batch(
            geant4_params,
            show_progress=True
        )

        # Create trajectory pairs
        pairs = []
        for unity_traj, geant4_result in zip(trajectories_to_process, geant4_results):
            # Convert Geant4 result to trajectory
            geant4_traj = ParticleTrajectory.from_geant4_result(
                trajectory_id=unity_traj.trajectory_id,
                geant4_result=geant4_result
            )

            # Create pair
            pair = TrajectoryPair(
                pair_id=unity_traj.trajectory_id,
                unity_trajectory=unity_traj,
                geant4_trajectory=geant4_traj
            )

            # Calculate metrics and reward
            pair.calculate_metrics()
            pair.reward = self._calculate_reward(pair)

            pairs.append(pair)

        # Update statistics
        with self.lock:
            self.trajectory_pairs.extend(pairs)
            self.total_processed += len(pairs)

        elapsed = time.time() - start_time

        # Statistics
        successful_pairs = [p for p in pairs if p.geant4_trajectory.num_steps > 0]
        failed_pairs = [p for p in pairs if p.geant4_trajectory.num_steps == 0]

        logger.info(f"Buffer processed in {elapsed:.2f}s")
        logger.info(f"  Successful: {len(successful_pairs)}/{len(pairs)}")
        logger.info(f"  Failed (no steps): {len(failed_pairs)}/{len(pairs)}")

        if successful_pairs:
            logger.info(f"  Average distance: {np.mean([p.position_distance for p in successful_pairs]):.3f} cm")
            logger.info(f"  Average reward: {np.mean([p.reward for p in successful_pairs]):.3f}")
            logger.info(f"  Average energy deposit: {np.mean([p.geant4_trajectory.total_energy_deposited for p in successful_pairs]):.3f} MeV")

        if failed_pairs:
            logger.warning(f"⚠️  {len(failed_pairs)} trajectories had no steps!")
            logger.warning(f"   This means particles didn't enter phantom")
            if not self.correct_direction:
                logger.warning(f"   💡 Enable direction correction: correct_direction=True")

        return pairs

    def _calculate_reward(self, pair: TrajectoryPair) -> float:
        """
        Calculate reward for trajectory pair

        Reward components:
        - Distance penalty: -position_distance
        - Energy penalty: -energy_difference
        - Bonus for completion

        Args:
            pair: Trajectory pair

        Returns:
            Reward value
        """
        # If Geant4 trajectory failed, heavy penalty
        if pair.geant4_trajectory.num_steps == 0:
            return -100.0  # Large penalty for missing phantom

        # Distance penalty (main component)
        distance_penalty = -pair.position_distance

        # Energy penalty
        energy_penalty = -0.1 * pair.energy_difference

        # Completion bonus
        completion_bonus = 1.0 if pair.unity_trajectory.completed else 0.0

        # Total reward
        reward = distance_penalty + energy_penalty + completion_bonus

        return reward

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self.lock:
            return {
                'buffer_size': len(self.unity_trajectories),
                'total_collected': self.total_collected,
                'total_processed': self.total_processed,
                'total_pairs': len(self.trajectory_pairs),
                'next_id': self.next_trajectory_id,
                'direction_correction_enabled': self.correct_direction
            }

    def clear(self):
        """Clear all buffers"""
        with self.lock:
            self.unity_trajectories.clear()
            self.trajectory_pairs.clear()
            logger.info("Buffer cleared")