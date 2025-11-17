"""
Unity Connector
Handles connection and communication with Unity ML-Agents
"""

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class UnityConnector:
    """
    Manages Unity ML-Agents environment connection
    """

    def __init__(self,
                 worker_id: int = 0,
                 base_port: int = 5004,
                 time_scale: float = 20.0):
        """
        Initialize Unity connector

        Args:
            worker_id: Worker ID for parallel environments
            base_port: Base port for communication
            time_scale: Unity time scale (higher = faster)
        """
        self.worker_id = worker_id
        self.base_port = base_port
        self.time_scale = time_scale

        self.env = None
        self.behavior_name = None
        self.num_agents = 0

        logger.info(f"Unity connector initialized (worker {worker_id}, port {base_port})")

    def connect(self) -> bool:
        """
        Connect to Unity environment

        Returns:
            True if successful
        """
        try:
            logger.info("Connecting to Unity...")

            # Create environment
            self.env = UnityEnvironment(
                file_name=None,  # None = connect to Unity Editor
                worker_id=self.worker_id,
                base_port=self.base_port
            )

            # Reset environment
            self.env.reset()

            # Get behavior name
            behavior_names = list(self.env.behavior_specs.keys())
            if not behavior_names:
                logger.error("No behaviors found in Unity environment!")
                return False

            self.behavior_name = behavior_names[0]
            behavior_spec = self.env.behavior_specs[self.behavior_name]

            # Get number of agents
            decision_steps, _ = self.env.get_steps(self.behavior_name)
            self.num_agents = len(decision_steps)

            logger.info(f"✅ Connected to Unity!")
            logger.info(f"   Behavior: {self.behavior_name}")
            logger.info(f"   Agents: {self.num_agents}")
            logger.info(f"   Observation shape: {behavior_spec.observation_specs[0].shape}")
            logger.info(f"   Action shape: {behavior_spec.action_spec.continuous_size}")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to Unity: {e}")
            return False

    def get_observations(self) -> np.ndarray:
        """
        Get observations from all agents (ONLY agents waiting for actions)

        Returns:
            Observations array (num_agents, obs_size)
        """
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        if len(decision_steps) > 0:
            return decision_steps.obs[0]  # First observation type
        else:
            return np.array([])

    def get_step_info(self) -> Tuple[int, int, int]:
        """
        Get detailed step information

        Returns:
            (num_decision, num_terminal, total_agents)
        """
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        num_decision = len(decision_steps)
        num_terminal = len(terminal_steps)
        total = num_decision + num_terminal

        return num_decision, num_terminal, total

    def get_all_observations(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get observations from ALL agents (decision + terminal)

        Returns:
            (decision_obs, terminal_obs)
        """
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        decision_obs = decision_steps.obs[0] if len(decision_steps) > 0 else np.array([])
        terminal_obs = terminal_steps.obs[0] if len(terminal_steps) > 0 else np.array([])

        return decision_obs, terminal_obs

    def send_actions(self, actions: np.ndarray):
        """
        Send actions to agents (ONLY to agents in decision_steps)

        Args:
            actions: Actions array (num_agents, action_size)
        """
        decision_steps, _ = self.env.get_steps(self.behavior_name)

        if len(decision_steps) > 0:
            action_tuple = ActionTuple(continuous=actions)
            self.env.set_actions(self.behavior_name, action_tuple)

        self.env.step()

    def close(self):
        """Close connection"""
        if self.env is not None:
            self.env.close()
            logger.info("Unity connection closed")