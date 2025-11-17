"""
Step Reward Calculator
Calculates immediate rewards by comparing Unity vs Geant4 steps
"""

import numpy as np
from typing import Dict, Any


class StepRewardCalculator:
    """
    Calculate per-step rewards based on Unity-Geant4 comparison

    Meeting requirements:
    - "uczenie agenta per every step" - reward after each step
    - "kończymy epizod jeśli odległość większa" - distance-based termination
    """

    def __init__(self,
                 position_weight: float = 1.0,
                 energy_weight: float = 0.5,
                 direction_weight: float = 0.3,
                 max_position_error: float = 2.0):  # cm
        """
        Initialize reward calculator

        Args:
            position_weight: Weight for position error
            energy_weight: Weight for energy error
            direction_weight: Weight for direction error
            max_position_error: Maximum allowed position error (episode termination)
        """
        self.position_weight = position_weight
        self.energy_weight = energy_weight
        self.direction_weight = direction_weight
        self.max_position_error = max_position_error

    def calculate_step_reward(self,
                             unity_state: Dict[str, Any],
                             geant4_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate immediate step reward

        Args:
            unity_state: {
                'position': np.array([x, y, z]),
                'energy': float,
                'direction': np.array([dx, dy, dz]),
                'energy_deposited': float
            }
            geant4_state: {
                'position': np.array([x, y, z]),
                'energy': float,
                'direction': np.array([dx, dy, dz]),
                'energy_deposited': float,
                'particle_stopped': bool
            }

        Returns:
            {
                'reward': float,
                'position_error': float,
                'energy_error': float,
                'direction_error': float,
                'should_terminate': bool,
                'termination_reason': str
            }
        """
        # Extract states
        unity_pos = np.array(unity_state['position'])
        geant4_pos = np.array(geant4_state['position'])

        unity_energy = unity_state['energy']
        geant4_energy = geant4_state['energy']

        unity_dir = np.array(unity_state['direction'])
        geant4_dir = np.array(geant4_state['direction'])

        # --- Position Error ---
        position_error = float(np.linalg.norm(unity_pos - geant4_pos))
        position_penalty = -self.position_weight * position_error

        # --- Energy Error ---
        energy_error = float(abs(unity_energy - geant4_energy))
        energy_penalty = -self.energy_weight * energy_error

        # --- Direction Error ---
        # Normalize directions
        unity_dir_norm = unity_dir / (np.linalg.norm(unity_dir) + 1e-8)
        geant4_dir_norm = geant4_dir / (np.linalg.norm(geant4_dir) + 1e-8)

        # Angle between directions (radians)
        dot_product = np.clip(np.dot(unity_dir_norm, geant4_dir_norm), -1.0, 1.0)
        direction_error = float(np.arccos(dot_product))
        direction_penalty = -self.direction_weight * direction_error

        # --- Total Reward ---
        step_reward = position_penalty + energy_penalty + direction_penalty

        # Small living penalty (encourages efficiency)
        step_reward -= 0.001

        # --- Check Termination Conditions ---
        should_terminate = False
        termination_reason = ""

        # 1. Distance-based termination (from meeting notes)
        if position_error > self.max_position_error:
            should_terminate = True
            termination_reason = f"position_error_too_large_{position_error:.2f}cm"
            step_reward -= 5.0  # Large penalty

        # 2. Geant4 says particle stopped
        if geant4_state.get('particle_stopped', False):
            should_terminate = True
            termination_reason = "geant4_particle_stopped"
            step_reward += 2.0  # Bonus for completing trajectory

        # 3. Energy depleted
        if unity_energy < 0.01:
            should_terminate = True
            termination_reason = "energy_depleted"
            step_reward += 1.0  # Small bonus

        return {
            'reward': float(step_reward),
            'position_error': position_error,
            'energy_error': energy_error,
            'direction_error': direction_error,
            'should_terminate': should_terminate,
            'termination_reason': termination_reason,
            'metrics': {
                'position_penalty': position_penalty,
                'energy_penalty': energy_penalty,
                'direction_penalty': direction_penalty
            }
        }

    def calculate_episode_summary(self, step_rewards: list) -> Dict[str, float]:
        """
        Calculate episode-level statistics

        Args:
            step_rewards: List of step reward dictionaries

        Returns:
            Episode statistics
        """
        if not step_rewards:
            return {}

        total_reward = sum(r['reward'] for r in step_rewards)

        position_errors = [r['position_error'] for r in step_rewards]
        energy_errors = [r['energy_error'] for r in step_rewards]

        return {
            'total_reward': total_reward,
            'num_steps': len(step_rewards),
            'mean_position_error': np.mean(position_errors),
            'max_position_error': np.max(position_errors),
            'mean_energy_error': np.mean(energy_errors),
            'final_position_error': position_errors[-1],
            'final_energy_error': energy_errors[-1]
        }