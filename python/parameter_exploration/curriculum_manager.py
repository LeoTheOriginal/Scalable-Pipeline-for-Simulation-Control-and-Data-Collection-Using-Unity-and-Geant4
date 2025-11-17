"""
Curriculum Learning Manager
Gradually increases training difficulty
"""

from typing import Dict, List
import numpy as np

class CurriculumStage:
    """Single curriculum stage"""

    def __init__(self,
                 stage_id: int,
                 energy_range: tuple,
                 max_steps: int,
                 reward_threshold: float):
        self.stage_id = stage_id
        self.energy_range = energy_range
        self.max_steps = max_steps
        self.reward_threshold = reward_threshold


class CurriculumManager:
    """
    Manages curriculum learning progression
    """

    def __init__(self):
        self.stages = self._define_stages()
        self.current_stage = 0
        self.stage_episode_count = 0
        self.stage_reward_history = []

    def _define_stages(self) -> List[CurriculumStage]:
        """
        Define curriculum stages

        Stage 1: Low energy, short trajectories (easy)
        Stage 2: Medium energy, medium trajectories
        Stage 3: High energy, long trajectories (hard)
        Stage 4: Full range
        """
        return [
            CurriculumStage(
                stage_id=1,
                energy_range=(1.0, 5.0),
                max_steps=100,
                reward_threshold=0.5
            ),
            CurriculumStage(
                stage_id=2,
                energy_range=(5.0, 10.0),
                max_steps=200,
                reward_threshold=0.6
            ),
            CurriculumStage(
                stage_id=3,
                energy_range=(10.0, 15.0),
                max_steps=500,
                reward_threshold=0.7
            ),
            CurriculumStage(
                stage_id=4,
                energy_range=(1.0, 20.0),
                max_steps=1000,
                reward_threshold=0.8
            )
        ]

    def get_current_stage(self) -> CurriculumStage:
        """Get current training stage"""
        return self.stages[self.current_stage]

    def record_episode(self, episode_reward: float):
        """Record episode result and check for progression"""
        self.stage_reward_history.append(episode_reward)
        self.stage_episode_count += 1

        # Check progression every 100 episodes
        if self.stage_episode_count >= 100:
            mean_reward = np.mean(self.stage_reward_history[-100:])

            current_stage = self.stages[self.current_stage]

            if mean_reward >= current_stage.reward_threshold:
                if self.current_stage < len(self.stages) - 1:
                    self.current_stage += 1
                    print(f"✅ Progressed to Stage {self.current_stage + 1}")
                    print(f"   New energy range: {self.stages[self.current_stage].energy_range}")

                    self.stage_episode_count = 0
                    self.stage_reward_history = []