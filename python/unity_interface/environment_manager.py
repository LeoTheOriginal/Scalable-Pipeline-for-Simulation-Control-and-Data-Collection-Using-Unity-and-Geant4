"""
Unity Environment Manager
Manages Unity ML-Agents environment lifecycle and interactions
"""

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np
from typing import Optional, Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnityEnvironmentManager:
    """
    Manages Unity ML-Agents environment for simulation control
    """

    def __init__(self, environment_path: Optional[str] = None,
                 worker_id: int = 0,
                 base_port: int = 5005,
                 seed: int = 42):
        """
        Initialize Unity Environment Manager

        Args:
            environment_path: Path to Unity executable (None for Unity Editor mode)
            worker_id: Unique ID for parallel environments
            base_port: Base communication port
            seed: Random seed for reproducibility
        """
        self.environment_path = environment_path
        self.worker_id = worker_id
        self.base_port = base_port
        self.seed = seed
        self.env: Optional[UnityEnvironment] = None
        self.behavior_name: Optional[str] = None

    def initialize(self) -> bool:
        """
        Initialize the Unity environment

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Initializing Unity environment (worker_id: {self.worker_id})")

            self.env = UnityEnvironment(
                file_name=self.environment_path,
                worker_id=self.worker_id,
                base_port=self.base_port,
                seed=self.seed
            )

            # Reset environment to get behavior names
            self.env.reset()

            # Get behavior name (assumes single behavior)
            self.behavior_name = list(self.env.behavior_specs.keys())[0]
            logger.info(f"Environment initialized with behavior: {self.behavior_name}")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize environment: {e}")
            return False

    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment

        Returns:
            dict: Initial observations and info
        """
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call initialize() first.")

        self.env.reset()
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        return {
            'observations': decision_steps.obs,
            'rewards': decision_steps.reward,
            'agents': decision_steps.agent_id,
            'step_count': 0
        }

    def step(self, actions: np.ndarray) -> Dict[str, Any]:
        """
        Execute actions in the environment

        Args:
            actions: Actions to execute

        Returns:
            dict: Observations, rewards, done flags, and info
        """
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call initialize() first.")

        # Fix shape: ensure actions are (num_agents, action_size)
        if len(actions.shape) == 1:
            actions = actions.reshape(1, -1)  # Add batch dimension

        # Set actions for agents
        action_tuple = ActionTuple(continuous=actions)
        self.env.set_actions(self.behavior_name, action_tuple)

        # Step the environment
        self.env.step()

        # Get results
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        return {
            'observations': decision_steps.obs if len(decision_steps) > 0 else terminal_steps.obs,
            'rewards': decision_steps.reward if len(decision_steps) > 0 else terminal_steps.reward,
            'done': len(terminal_steps) > 0,
            'agents': decision_steps.agent_id if len(decision_steps) > 0 else terminal_steps.agent_id
        }

    def get_observation_space(self) -> Dict[str, Any]:
        """
        Get observation space information

        Returns:
            dict: Observation space specifications
        """
        if self.env is None or self.behavior_name is None:
            raise RuntimeError("Environment not initialized.")

        spec = self.env.behavior_specs[self.behavior_name]

        return {
            'observation_shapes': [obs_spec.shape for obs_spec in spec.observation_specs],
            'action_size': spec.action_spec.continuous_size,
            'action_type': 'continuous' if spec.action_spec.continuous_size > 0 else 'discrete'
        }

    def close(self):
        """
        Close the environment and clean up resources
        """
        if self.env is not None:
            logger.info("Closing Unity environment")
            self.env.close()
            self.env = None

    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()