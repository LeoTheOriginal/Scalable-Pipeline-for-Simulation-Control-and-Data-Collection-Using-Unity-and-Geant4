"""
Trajectory Aggregator
Buffers multiple agent trajectories before sending to Geant4

Meeting requirement: "buforujemy wiele historii cząstki (>1000)"
"""

import numpy as np
from typing import List, Dict, Any, Optional
from collections import deque
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class TrajectoryAggregator:
    """
    Aggregates trajectories from multiple Unity agents
    Sends batches to Geant4 for parallel processing

    Flow:
    1. Unity agents generate trajectories
    2. Buffer accumulates >1000 trajectories
    3. Send batch to parallel Geant4 runners
    4. Return rewards to agents
    """

    def __init__(self,
                 buffer_size: int = 1000,
                 geant4_executable: str = None,
                 num_workers: int = 8,
                 auto_process: bool = True):
        """
        Initialize trajectory aggregator

        Args:
            buffer_size: Number of trajectories to buffer before processing
            geant4_executable: Path to Geant4 executable
            num_workers: Number of parallel Geant4 workers
            auto_process: Automatically process when buffer is full
        """
        self.buffer_size = buffer_size
        self.auto_process = auto_process

        # Trajectory buffer
        self.pending_trajectories = deque()  # Trajectories waiting to be sent
        self.processing_trajectories = {}  # Currently being processed

        # Geant4 parallel runner
        if geant4_executable:
            from geant4_interface.parallel_runner import ParallelGeant4Runner
            self.geant4_runner = ParallelGeant4Runner(
                geant4_executable=geant4_executable,
                output_directory="./trajectory_buffer_geant4",
                num_workers=num_workers
            )
        else:
            self.geant4_runner = None
            logger.warning("No Geant4 executable provided - running in mock mode")

        # Statistics
        self.total_trajectories_received = 0
        self.total_trajectories_processed = 0
        self.next_trajectory_id = 0

        logger.info(f"TrajectoryAggregator initialized:")
        logger.info(f"  Buffer size: {buffer_size}")
        logger.info(f"  Workers: {num_workers}")
        logger.info(f"  Auto-process: {auto_process}")

    def add_trajectory(self,
                       agent_id: int,
                       initial_conditions: Dict[str, Any],
                       steps: List[Dict[str, Any]]) -> int:
        """
        Add completed trajectory from Unity agent

        Args:
            agent_id: Unity agent ID
            initial_conditions: {
                'particle_type': str,
                'initial_energy': float,
                'initial_position': [x, y, z],
                'initial_direction': [dx, dy, dz]
            }
            steps: List of step dictionaries with Unity state

        Returns:
            trajectory_id: Unique ID assigned to this trajectory
        """
        trajectory_id = self.next_trajectory_id
        self.next_trajectory_id += 1

        trajectory = {
            'trajectory_id': trajectory_id,
            'agent_id': agent_id,
            'initial_conditions': initial_conditions,
            'steps': steps,
            'received_at': time.time()
        }

        self.pending_trajectories.append(trajectory)
        self.total_trajectories_received += 1

        logger.debug(f"Added trajectory {trajectory_id} from agent {agent_id} "
                     f"({len(steps)} steps)")

        # Auto-process if buffer is full
        if self.auto_process and len(self.pending_trajectories) >= self.buffer_size:
            logger.info(f"Buffer full ({len(self.pending_trajectories)} trajectories), "
                        f"triggering processing")
            self.process_buffer()

        return trajectory_id

    def process_buffer(self) -> List[Dict[str, Any]]:
        """
        Process buffered trajectories with parallel Geant4

        Meeting requirement:
        - "buforujemy wiele historii cząstki (>1000)"
        - "odpalamy na kilku agentach równolegle"

        Returns:
            List of processed trajectory results with rewards
        """
        if len(self.pending_trajectories) == 0:
            logger.warning("Buffer empty, nothing to process")
            return []

        # Extract trajectories from buffer
        trajectories_to_process = []
        while self.pending_trajectories and len(trajectories_to_process) < self.buffer_size:
            trajectories_to_process.append(self.pending_trajectories.popleft())

        num_trajectories = len(trajectories_to_process)
        logger.info(f"Processing batch: {num_trajectories} trajectories")

        start_time = time.time()

        # Prepare Geant4 parameters for each trajectory
        geant4_params = []
        for traj in trajectories_to_process:
            params = {
                'particle_type': traj['initial_conditions']['particle_type'],
                'particle_energy': traj['initial_conditions']['initial_energy'],
                'particle_position': traj['initial_conditions']['initial_position'],
                'particle_direction': traj['initial_conditions']['initial_direction'],
                'num_events': 1
            }
            geant4_params.append(params)

        # Run parallel Geant4 batch
        if self.geant4_runner:
            logger.info(f"Running Geant4 batch ({num_trajectories} simulations)...")
            geant4_results = self.geant4_runner.run_batch(
                geant4_params,
                show_progress=True
            )
        else:
            # Mock mode for testing
            logger.warning("Running in MOCK mode (no Geant4)")
            geant4_results = self._mock_geant4_results(num_trajectories)

        # Match Unity trajectories with Geant4 results
        processed_trajectories = []

        from rl_training.reward_calculator import StepRewardCalculator
        reward_calculator = StepRewardCalculator()

        for unity_traj, geant4_result in zip(trajectories_to_process, geant4_results):

            if not geant4_result['success']:
                logger.error(f"Geant4 failed for trajectory {unity_traj['trajectory_id']}")
                continue

            # Calculate per-step rewards
            step_rewards = []

            # Get Geant4 steps
            if 'events' in geant4_result and geant4_result['events']:
                geant4_steps = geant4_result['events'][0].get('steps', [])
            else:
                geant4_steps = []

            # Compare Unity steps with Geant4 steps
            for i, unity_step in enumerate(unity_traj['steps']):

                # Get corresponding Geant4 step (if exists)
                if i < len(geant4_steps):
                    geant4_step = geant4_steps[i]

                    # Prepare states for reward calculation
                    unity_state = {
                        'position': np.array(unity_step['position']),
                        'energy': unity_step['energy'],
                        'direction': np.array(unity_step['direction']),
                        'energy_deposited': unity_step.get('energy_deposited', 0.0)
                    }

                    geant4_state = {
                        'position': np.array(geant4_step['position']),
                        'energy': geant4_step['energy'],
                        'direction': np.array(geant4_step.get('direction', [1, 0, 0])),
                        'energy_deposited': geant4_step.get('energy_deposit', 0.0),
                        'particle_stopped': False  # Will be True on last step
                    }

                    # Last step
                    if i == len(geant4_steps) - 1:
                        geant4_state['particle_stopped'] = True

                    # Calculate reward
                    reward_data = reward_calculator.calculate_step_reward(
                        unity_state,
                        geant4_state
                    )

                    step_rewards.append(reward_data)
                else:
                    # Unity has more steps than Geant4 (particle stopped earlier in Geant4)
                    # Give penalty for continuing after Geant4 stopped
                    step_rewards.append({
                        'reward': -1.0,
                        'position_error': 999.0,
                        'should_terminate': True,
                        'termination_reason': 'unity_exceeded_geant4_steps'
                    })

            # Calculate episode summary
            episode_summary = reward_calculator.calculate_episode_summary(step_rewards)

            processed_traj = {
                'trajectory_id': unity_traj['trajectory_id'],
                'agent_id': unity_traj['agent_id'],
                'step_rewards': step_rewards,
                'episode_summary': episode_summary,
                'geant4_result': geant4_result,
                'processing_time': time.time() - start_time
            }

            processed_trajectories.append(processed_traj)
            self.total_trajectories_processed += 1

        elapsed = time.time() - start_time

        logger.info(f"✅ Batch processed in {elapsed:.2f}s")
        logger.info(f"   Trajectories: {len(processed_trajectories)}/{num_trajectories}")
        logger.info(f"   Time per trajectory: {elapsed / num_trajectories:.3f}s")

        if processed_trajectories:
            avg_reward = np.mean([t['episode_summary']['total_reward']
                                  for t in processed_trajectories])
            avg_pos_error = np.mean([t['episode_summary']['mean_position_error']
                                     for t in processed_trajectories])
            logger.info(f"   Avg total reward: {avg_reward:.3f}")
            logger.info(f"   Avg position error: {avg_pos_error:.3f} cm")

        return processed_trajectories

    def _mock_geant4_results(self, n_trajectories: int) -> List[Dict]:
        """
        Generate mock Geant4 results for testing
        """
        results = []
        for i in range(n_trajectories):
            results.append({
                'success': True,
                'total_energy_deposit': np.random.uniform(1.0, 10.0),
                'num_events': 1,
                'events': [{
                    'event_id': 0,
                    'total_energy_deposit': np.random.uniform(1.0, 10.0),
                    'num_steps': np.random.randint(10, 50),
                    'steps': [
                        {
                            'step_number': j,
                            'position': [
                                -6.0 + j * 0.1,
                                np.random.uniform(-0.1, 0.1),
                                np.random.uniform(-0.1, 0.1)
                            ],
                            'direction': [1, 0, 0],
                            'energy': 10.0 - j * 0.1,
                            'energy_deposit': 0.1
                        }
                        for j in range(np.random.randint(10, 50))
                    ]
                }]
            })
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregator statistics"""
        return {
            'pending_trajectories': len(self.pending_trajectories),
            'total_received': self.total_trajectories_received,
            'total_processed': self.total_trajectories_processed,
            'buffer_size': self.buffer_size,
            'buffer_utilization': len(self.pending_trajectories) / self.buffer_size
        }