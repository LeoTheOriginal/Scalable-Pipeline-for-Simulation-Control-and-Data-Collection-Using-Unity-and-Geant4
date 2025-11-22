"""
train_advanced.py - Advanced Training Script with Custom Callbacks

Features:
- Custom callback for reward component logging
- TensorBoard integration for detailed metrics
- Curriculum learning support
- Model checkpointing with best model tracking
- Episode statistics logging
- Training time estimation

Author: Dawid (Warsaw University of Technology)
"""

import sys
import os
import time
from typing import Dict, Any
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.logger import TensorBoardOutputFormat
import json

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from src.training.environment import Geant4ParticleEnv


# ============================================================================
# CUSTOM CALLBACKS
# ============================================================================

class RewardComponentLogger(BaseCallback):
    """
    Callback for logging individual reward components.

    This helps diagnose which aspects of the physics model
    the agent is struggling with.
    """

    def __init__(self, log_freq: int = 100, verbose: int = 0):
        super(RewardComponentLogger, self).__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_components = {
            'position': [],
            'direction': [],
            'energy': [],
            'physics': [],
            'scattering': [],
            'range': [],
            'thermodynamics': [],
            'progress': [],
            'smoothness': [],
            'precision_bonus': []
        }
        self.episode_lengths = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        """Called after each environment step"""

        # Get info from the last step
        infos = self.locals.get('infos', [])

        for info in infos:
            if 'reward_components' in info:
                components = info['reward_components']

                # Accumulate components
                for key in self.episode_components.keys():
                    if key in components:
                        self.episode_components[key].append(components[key])

        # Check if episode finished
        dones = self.locals.get('dones', [])

        if any(dones):
            self.episode_count += 1

            # Log every N episodes
            if self.episode_count % self.log_freq == 0:
                self._log_episode_stats()

        return True

    def _log_episode_stats(self):
        """Log accumulated statistics to TensorBoard"""

        # Calculate means
        stats = {}
        for key, values in self.episode_components.items():
            if len(values) > 0:
                stats[f'reward_components/{key}_mean'] = np.mean(values)
                stats[f'reward_components/{key}_std'] = np.std(values)

        # Log to TensorBoard
        for key, value in stats.items():
            self.logger.record(key, value)

        # Reset accumulators
        for key in self.episode_components.keys():
            self.episode_components[key] = []

        if self.verbose > 0:
            print(f"\n[Callback] Episode {self.episode_count} - Reward Component Stats:")
            for key, value in stats.items():
                if 'mean' in key:
                    print(f"  {key}: {value:.2f}")


class PhysicsMetricsCallback(BaseCallback):
    """
    Callback for logging physics-specific metrics.

    Tracks:
    - Energy conservation violations
    - Scattering angle distributions
    - Trajectory depths achieved
    - Position accuracy over time
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super(PhysicsMetricsCallback, self).__init__(verbose)
        self.log_freq = log_freq
        self.step_count = 0

        # Accumulators
        self.energy_violations = []
        self.position_errors = []
        self.depth_reached = []

    def _on_step(self) -> bool:
        self.step_count += 1

        infos = self.locals.get('infos', [])

        for info in infos:
            # Energy conservation
            if 'ground_truth_energy' in info and 'predicted_energy' in info:
                gt_energy = info['ground_truth_energy']
                pred_energy = info['predicted_energy']

                # Check for unphysical energy gain
                if pred_energy > gt_energy:
                    self.energy_violations.append(pred_energy - gt_energy)

            # Position accuracy (if available in info)
            if 'position_error' in info:
                self.position_errors.append(info['position_error'])

        # Log periodically
        if self.step_count % self.log_freq == 0:
            self._log_physics_metrics()

        return True

    def _log_physics_metrics(self):
        """Log physics metrics"""

        if len(self.energy_violations) > 0:
            self.logger.record('physics/energy_violations_mean', np.mean(self.energy_violations))
            self.logger.record('physics/energy_violations_max', np.max(self.energy_violations))
            self.logger.record('physics/energy_violations_count', len(self.energy_violations))

        if len(self.position_errors) > 0:
            self.logger.record('physics/position_error_mean', np.mean(self.position_errors))
            self.logger.record('physics/position_error_std', np.std(self.position_errors))

        # Reset
        self.energy_violations = []
        self.position_errors = []

        if self.verbose > 0:
            print(f"\n[Physics] Step {self.step_count} - Metrics logged")


class BestModelCallback(BaseCallback):
    """
    Callback that saves the best model based on mean reward.

    Keeps track of the best performing model across training.
    """

    def __init__(self, save_path: str, check_freq: int = 10000, verbose: int = 1):
        super(BestModelCallback, self).__init__(verbose)
        self.save_path = save_path
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf
        self.step_count = 0

        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        self.step_count += 1

        if self.step_count % self.check_freq == 0:
            # Get recent episode rewards
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
                mean_reward = np.mean(rewards)

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward

                    # Save best model
                    save_file = os.path.join(self.save_path, 'best_model')
                    self.model.save(save_file)

                    if self.verbose > 0:
                        print(f"\n[BestModel] New best model! Mean reward: {mean_reward:.2f}")
                        print(f"[BestModel] Saved to: {save_file}")

                    # Save metadata
                    metadata = {
                        'mean_reward': float(mean_reward),
                        'timestep': int(self.step_count),
                        'timestamp': time.time()
                    }

                    with open(os.path.join(self.save_path, 'best_model_info.json'), 'w') as f:
                        json.dump(metadata, f, indent=2)

        return True


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Training hyperparameters"""

    # PPO Hyperparameters
    LEARNING_RATE = 3e-4
    N_STEPS = 2048  # Steps per update
    BATCH_SIZE = 64  # Minibatch size
    N_EPOCHS = 10  # Optimization epochs per update
    GAMMA = 0.99  # Discount factor
    GAE_LAMBDA = 0.95  # GAE parameter
    CLIP_RANGE = 0.2  # PPO clip parameter
    ENT_COEF = 0.01  # Entropy coefficient
    VF_COEF = 0.5  # Value function coefficient
    MAX_GRAD_NORM = 0.5  # Gradient clipping

    # Training settings
    TOTAL_TIMESTEPS = 50_000
    CHECKPOINT_FREQ = 25_000
    LOG_INTERVAL = 10

    # Environment settings
    MAX_EPISODE_STEPS = 500
    NORMALIZE_OBS = True

    @classmethod
    @staticmethod
    def to_dict() -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'LEARNING_RATE': TrainingConfig.LEARNING_RATE,
            'N_STEPS': TrainingConfig.N_STEPS,
            'BATCH_SIZE': TrainingConfig.BATCH_SIZE,
            'N_EPOCHS': TrainingConfig.N_EPOCHS,
            'GAMMA': TrainingConfig.GAMMA,
            'GAE_LAMBDA': TrainingConfig.GAE_LAMBDA,
            'CLIP_RANGE': TrainingConfig.CLIP_RANGE,
            'ENT_COEF': TrainingConfig.ENT_COEF,
            'VF_COEF': TrainingConfig.VF_COEF,
            'MAX_GRAD_NORM': TrainingConfig.MAX_GRAD_NORM,
            'TOTAL_TIMESTEPS': TrainingConfig.TOTAL_TIMESTEPS,
            'CHECKPOINT_FREQ': TrainingConfig.CHECKPOINT_FREQ,
            'LOG_INTERVAL': TrainingConfig.LOG_INTERVAL,
            'MAX_EPISODE_STEPS': TrainingConfig.MAX_EPISODE_STEPS,
            'NORMALIZE_OBS': TrainingConfig.NORMALIZE_OBS
        }


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_advanced():
    """
    Main training function with advanced features.
    """

    print("\n" + "=" * 80)
    print("ADVANCED PHYSICS-INFORMED RL TRAINING")
    print("=" * 80 + "\n")

    # Setup directories
    models_dir = os.path.join(project_root, "data", "models", "ppo_geant4_advanced")
    logs_dir = os.path.join(project_root, "data", "logs", "advanced")
    best_model_dir = os.path.join(models_dir, "best")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    print(f"[Setup] Models directory: {models_dir}")
    print(f"[Setup] Logs directory: {logs_dir}")
    print(f"[Setup] TensorBoard: tensorboard --logdir={logs_dir}")

    # Save training configuration
    config_path = os.path.join(models_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(TrainingConfig.to_dict(), f, indent=2)
    print(f"[Setup] Config saved: {config_path}\n")

    # ========================================================================
    # Create Environment
    # ========================================================================
    print("[Env] Creating training environment...")
    try:
        env = Geant4ParticleEnv(
            history_length=3,
            max_episode_steps=TrainingConfig.MAX_EPISODE_STEPS,
            normalize_observations=TrainingConfig.NORMALIZE_OBS,
            verbose=False
        )
        print("[Env] ✅ Environment created successfully\n")
    except Exception as e:
        print(f"[Env] ❌ Failed to create environment: {e}")
        return

    # ========================================================================
    # Create PPO Agent
    # ========================================================================
    print("[Agent] Initializing PPO agent...")

    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # Deeper network for complex physics
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=TrainingConfig.LEARNING_RATE,
        n_steps=TrainingConfig.N_STEPS,
        batch_size=TrainingConfig.BATCH_SIZE,
        n_epochs=TrainingConfig.N_EPOCHS,
        gamma=TrainingConfig.GAMMA,
        gae_lambda=TrainingConfig.GAE_LAMBDA,
        clip_range=TrainingConfig.CLIP_RANGE,
        ent_coef=TrainingConfig.ENT_COEF,
        vf_coef=TrainingConfig.VF_COEF,
        max_grad_norm=TrainingConfig.MAX_GRAD_NORM,
        verbose=1,
        tensorboard_log=logs_dir,
        policy_kwargs=policy_kwargs
    )

    print("[Agent] ✅ PPO agent initialized")
    print(f"[Agent] Network architecture: {policy_kwargs['net_arch']}")
    print(f"[Agent] Learning rate: {TrainingConfig.LEARNING_RATE}")
    print(f"[Agent] Total parameters: ~{sum(p.numel() for p in model.policy.parameters()):,}\n")

    # ========================================================================
    # Setup Callbacks
    # ========================================================================
    print("[Callbacks] Setting up training callbacks...")

    # Checkpoint callback - saves model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=TrainingConfig.CHECKPOINT_FREQ,
        save_path=models_dir,
        name_prefix="checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False
    )

    # Reward component logger
    reward_logger = RewardComponentLogger(
        log_freq=100,
        verbose=1
    )

    # Physics metrics logger
    physics_logger = PhysicsMetricsCallback(
        log_freq=1000,
        verbose=1
    )

    # Best model saver
    best_model_callback = BestModelCallback(
        save_path=best_model_dir,
        check_freq=10000,
        verbose=1
    )

    # Combine all callbacks
    callback = CallbackList([
        checkpoint_callback,
        reward_logger,
        physics_logger,
        best_model_callback
    ])

    print("[Callbacks] ✅ Callbacks configured\n")

    # ========================================================================
    # Training Loop
    # ========================================================================
    print("=" * 80)
    print(f"STARTING TRAINING - {TrainingConfig.TOTAL_TIMESTEPS:,} timesteps")
    print("=" * 80 + "\n")

    start_time = time.time()

    try:
        model.learn(
            total_timesteps=TrainingConfig.TOTAL_TIMESTEPS,
            callback=callback,
            log_interval=TrainingConfig.LOG_INTERVAL,
            progress_bar=True
        )

        training_time = time.time() - start_time

        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Training time: {training_time / 3600:.2f} hours ({training_time:.1f} seconds)")
        print(f"Steps per second: {TrainingConfig.TOTAL_TIMESTEPS / training_time:.1f}")

    except KeyboardInterrupt:
        print("\n\n[Training] Interrupted by user. Saving current model...")
        training_time = time.time() - start_time

    except Exception as e:
        print(f"\n\n[Training] ❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========================================================================
    # Save Final Model
    # ========================================================================
    final_path = os.path.join(models_dir, "final_model")
    model.save(final_path)
    print(f"\n[Save] Final model saved: {final_path}")

    # Save final metadata
    final_metadata = {
        'timesteps': TrainingConfig.TOTAL_TIMESTEPS,
        'training_time_seconds': training_time,
        'config': TrainingConfig.to_dict(),
        'timestamp': time.time()
    }

    with open(os.path.join(models_dir, 'final_model_info.json'), 'w') as f:
        json.dump(final_metadata, f, indent=2)

    # ========================================================================
    # Quick Evaluation
    # ========================================================================
    print("\n" + "=" * 80)
    print("QUICK EVALUATION")
    print("=" * 80 + "\n")

    print("[Eval] Running 10 test episodes...")

    episode_rewards = []
    episode_lengths = []

    for i in range(10):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done and episode_length < 200:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"  Episode {i + 1}: Reward={episode_reward:.2f}, Length={episode_length}")

    print(f"\n[Eval] Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"[Eval] Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")

    # ========================================================================
    # Cleanup
    # ========================================================================
    env.close()

    print("\n" + "=" * 80)
    print("✅ ALL DONE!")
    print("=" * 80)
    print(f"\nTo view training progress:")
    print(f"  tensorboard --logdir={logs_dir}")
    print(f"\nBest model saved at:")
    print(f"  {os.path.join(best_model_dir, 'best_model.zip')}")
    print("=" * 80 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    train_advanced()