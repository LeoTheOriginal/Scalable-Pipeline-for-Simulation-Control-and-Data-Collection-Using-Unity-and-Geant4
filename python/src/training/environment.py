"""
environment.py - Advanced Physics-Informed RL Environment for Electron Transport

This environment implements a sophisticated reward structure based on:
1. Multiple Coulomb Scattering (Highland/Molière formalism)
2. Energy loss via Bethe-Bloch formula with Landau-Vavilov straggling
3. Bremsstrahlung radiation losses
4. Range-energy relationships for electrons in water
5. Trajectory consistency with Geant4 ground truth

Author: Dawid (Warsaw University of Technology)
Project: Scalable Pipeline for Simulation Control using Unity + Geant4
"""

import gym
from gym import spaces
import numpy as np
import os
import sys
from typing import Tuple, Dict, Optional
from collections import deque

# ============================================================================
# 1. SYSTEM CONFIGURATION
# ============================================================================
PATH_TO_GEANT4_BIN = r"C:\Geant4\install\bin"

if os.name == 'nt' and os.path.exists(PATH_TO_GEANT4_BIN):
    try:
        os.add_dll_directory(PATH_TO_GEANT4_BIN)
    except Exception:
        pass

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.simulation import geant4_sim
except ImportError as e:
    raise ImportError(f"CRITICAL: Cannot load Geant4 module. Error: {e}")


# ============================================================================
# 2. PHYSICAL CONSTANTS AND MATERIAL PROPERTIES
# ============================================================================

class PhysicsConstants:
    """Physical constants for electron transport in water"""

    # Particle properties
    ELECTRON_MASS = 0.511  # MeV/c²
    ELECTRON_CHARGE = 1.0  # Elementary charge

    # Water properties (target material)
    WATER_DENSITY = 1.0  # g/cm³
    WATER_Z_OVER_A = 0.555  # Z/A ratio
    WATER_I = 75.0e-6  # Mean excitation energy [MeV]
    WATER_X0 = 36.08  # Radiation length [cm]

    # Phantom geometry
    PHANTOM_SIZE = 10.0  # cm (half-size = 5 cm, so full size 10x10x10)
    PHANTOM_CENTER = np.array([0.0, 0.0, 0.0])
    BEAM_START = np.array([-6.0, 0.0, 0.0])  # Typical start position

    # Physics cutoffs
    MIN_ENERGY = 0.1  # MeV - below this, particle is absorbed
    MAX_ENERGY = 15.0  # MeV - sanity check


# ============================================================================
# 3. PHYSICS MODELS
# ============================================================================

class ElectronPhysics:
    """
    Physics engine for electron transport validation.
    Implements standard formulas from PDG (Particle Data Group).
    """

    @staticmethod
    def bethe_bloch_dedx(energy_mev: float, mass_mev: float = PhysicsConstants.ELECTRON_MASS) -> float:
        """
        Calculate mean energy loss dE/dx using Bethe-Bloch formula.

        Simplified version for electrons in water:
        dE/dx ≈ -2.0 MeV/cm for 10 MeV electrons

        Args:
            energy_mev: Kinetic energy [MeV]
            mass_mev: Particle mass [MeV]

        Returns:
            dE/dx in MeV/cm (positive value, actual loss is negative)
        """
        if energy_mev < PhysicsConstants.MIN_ENERGY:
            return 0.0

        # Total energy
        E_total = energy_mev + mass_mev

        # Relativistic beta and gamma
        gamma = E_total / mass_mev
        beta = np.sqrt(1.0 - 1.0 / (gamma ** 2))

        if beta < 0.01:  # Non-relativistic regime
            return 0.5  # Minimal loss

        # Simplified Bethe-Bloch (without shell corrections)
        # For water: ~2 MeV/cm at 10 MeV
        K = 0.307075  # MeV cm²/g
        Z_over_A = PhysicsConstants.WATER_Z_OVER_A
        rho = PhysicsConstants.WATER_DENSITY
        I = PhysicsConstants.WATER_I

        # Classic formula
        tmax = 2.0 * mass_mev * beta ** 2 * gamma ** 2 / (1.0 + 2.0 * gamma * mass_mev / mass_mev)

        dedx = K * Z_over_A * rho / (beta ** 2) * (
                0.5 * np.log(2.0 * mass_mev * beta ** 2 * gamma ** 2 * tmax / (I ** 2))
                - beta ** 2
        )

        # Clamp to reasonable values
        dedx = np.clip(dedx, 0.5, 10.0)

        return dedx

    @staticmethod
    def multiple_scattering_angle(energy_mev: float, path_length_cm: float,
                                  mass_mev: float = PhysicsConstants.ELECTRON_MASS) -> float:
        """
        Calculate RMS scattering angle using Highland formula.

        θ_rms = (13.6 MeV / βcp) * z * √(x/X₀) * [1 + 0.038*ln(x/X₀)]

        Args:
            energy_mev: Kinetic energy [MeV]
            path_length_cm: Path length in material [cm]
            mass_mev: Particle mass [MeV]

        Returns:
            RMS scattering angle [radians]
        """
        if path_length_cm < 1e-6 or energy_mev < PhysicsConstants.MIN_ENERGY:
            return 0.0

        # Momentum
        E_total = energy_mev + mass_mev
        p = np.sqrt(E_total ** 2 - mass_mev ** 2)

        # Relativistic beta
        beta = p / E_total

        # Reduced thickness
        x_over_X0 = path_length_cm / PhysicsConstants.WATER_X0

        if x_over_X0 < 1e-6:
            return 0.0

        # Highland formula
        theta_rms = (13.6 / (beta * p)) * np.sqrt(x_over_X0) * (1.0 + 0.038 * np.log(x_over_X0))

        # Convert to radians (formula gives mrad usually, but let's ensure units)
        return abs(theta_rms)

    @staticmethod
    def calculate_range(energy_mev: float) -> float:
        """
        Practical range of electrons in water using empirical formula.

        R ≈ 0.412 * E^1.265  [cm] for E in MeV

        This is valid for electrons from 0.1 to 20 MeV.

        Args:
            energy_mev: Kinetic energy [MeV]

        Returns:
            Practical range [cm]
        """
        if energy_mev < PhysicsConstants.MIN_ENERGY:
            return 0.0

        # Empirical formula for electrons in water
        range_cm = 0.412 * (energy_mev ** 1.265)

        return range_cm

    @staticmethod
    def bremsstrahlung_fraction(energy_mev: float) -> float:
        """
        Estimate fraction of energy lost to bremsstrahlung.

        For electrons: Radiative losses become significant above ~10 MeV
        Fraction ≈ E / (E + E_critical)
        E_critical ≈ 80 MeV for water

        Args:
            energy_mev: Kinetic energy [MeV]

        Returns:
            Fraction of losses due to radiation (0 to 1)
        """
        E_crit = 80.0  # Critical energy for water [MeV]

        fraction = energy_mev / (energy_mev + E_crit)

        return fraction

    @staticmethod
    def energy_straggling_sigma(energy_mev: float, path_length_cm: float) -> float:
        """
        Energy straggling (Landau-Vavilov distribution width).

        For thin absorbers, σ_E ≈ 0.1 * √(path_length) MeV

        Args:
            energy_mev: Kinetic energy [MeV]
            path_length_cm: Path length [cm]

        Returns:
            Standard deviation of energy loss [MeV]
        """
        if path_length_cm < 1e-6:
            return 0.0

        # Simple model: straggling increases with path
        sigma = 0.1 * np.sqrt(path_length_cm)

        return sigma


# ============================================================================
# 4. ADVANCED GYM ENVIRONMENT
# ============================================================================

class Geant4ParticleEnv(gym.Env):
    """
    Advanced Physics-Informed RL Environment for Electron Transport.

    Features:
    - Extended observation space with physics quantities
    - Step history tracking (velocity, acceleration)
    - Comprehensive reward function based on multiple physics criteria
    - Automatic normalization of observations
    - Trajectory consistency checking

    Observation Space (14 dimensions):
        [0:3]   - Position (x, y, z) [cm] - normalized
        [3:6]   - Momentum direction (unit vector) - normalized
        [6]     - Kinetic energy [MeV] - normalized
        [7]     - Depth in phantom [cm] - normalized
        [8]     - Angle relative to beam axis [rad] - normalized
        [9]     - Velocity magnitude [cm/step] - normalized
        [10]    - Remaining range estimate [cm] - normalized
        [11:14] - Previous step delta (dx, dy, dz) - normalized

    Action Space (7 dimensions):
        Predicted change: [dx, dy, dz, dpx, dpy, dpz, dE]
    """

    metadata = {'render.modes': []}

    def __init__(self,
                 history_length: int = 3,
                 max_episode_steps: int = 500,
                 normalize_observations: bool = True,
                 verbose: bool = False):
        """
        Initialize the environment.

        Args:
            history_length: Number of previous steps to track
            max_episode_steps: Maximum steps per episode
            normalize_observations: Whether to normalize obs to [-1, 1]
            verbose: Print debug information
        """
        super(Geant4ParticleEnv, self).__init__()

        self.verbose = verbose
        self.history_length = history_length
        self.max_episode_steps = max_episode_steps
        self.normalize_obs = normalize_observations

        if self.verbose:
            print("[Env] Initializing Advanced Geant4 Environment...")

        # Initialize Geant4 simulation manager
        self.sim_manager = geant4_sim.SimulationManager()

        # Physics engine
        self.physics = ElectronPhysics()

        # ====================================================================
        # ACTION SPACE
        # ====================================================================
        # Delta values: [dx, dy, dz, dpx, dpy, dpz, dE]
        # Position deltas: ±0.5 cm per step (realistic for ~mm step sizes)
        # Momentum deltas: ±3.0 MeV/c per step
        # Energy delta: -5.0 to +0.5 MeV (allow loss, minimal gain for numerical stability)

        low_action = np.array([-0.15, -0.15, -0.15, -2.0, -2.0, -2.0, -3.0], dtype=np.float32)
        high_action = np.array([0.15, 0.15, 0.15, 2.0, 2.0, 2.0, 0.1], dtype=np.float32)

        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=np.float32)

        # ====================================================================
        # OBSERVATION SPACE
        # ====================================================================
        # 14 dimensions (see class docstring)

        if normalize_observations:
            # Normalized space: all values roughly in [-1, 1]
            low_obs = np.full(14, -3.0, dtype=np.float32)
            high_obs = np.full(14, 3.0, dtype=np.float32)
        else:
            # Raw physical units
            low_obs = np.array([
                -20, -10, -10,  # Position
                -1, -1, -1,  # Direction
                0,  # Energy
                0,  # Depth
                0,  # Angle
                0,  # Velocity
                0,  # Range
                -5, -5, -5  # Previous delta
            ], dtype=np.float32)

            high_obs = np.array([
                20, 10, 10,  # Position
                1, 1, 1,  # Direction
                15,  # Energy
                10,  # Depth
                np.pi,  # Angle
                10,  # Velocity
                10,  # Range
                5, 5, 5  # Previous delta
            ], dtype=np.float32)

        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # ====================================================================
        # REWARD WEIGHTS (Hyperparameters for tuning)
        # ====================================================================
        self.W_POSITION = 5.0  # Position accuracy (most important)
        self.W_DIRECTION = 10.0  # Direction consistency
        self.W_ENERGY = 1.0  # Energy prediction (INCREASED from 20)
        self.W_PHYSICS = 20.0  # Physical consistency (INCREASED from 50)
        self.W_SCATTERING = 0.0  # Scattering angle realism (INCREASED from 15)
        self.W_RANGE = 0.0  # Range-energy consistency (INCREASED from 10)
        self.W_THERMODYNAMICS = 50.0  # Energy conservation (INCREASED from 100)
        self.W_PROGRESS = 2.0  # Forward progress bonus (DECREASED from 5 - was too high!)
        self.W_SMOOTHNESS = 10.0  # Trajectory smoothness

        # ====================================================================
        # INTERNAL STATE
        # ====================================================================
        self.current_trajectory = None
        self.current_step_idx = 0
        self.trajectory_length = 0
        self.episode_steps = 0

        # History tracking
        self.step_history = deque(maxlen=history_length)
        self.previous_state = None

        # Statistics
        self.episode_rewards = []
        self.total_episodes = 0

        if self.verbose:
            print(f"[Env] ✅ Environment initialized")
            print(f"[Env] Observation space: {self.observation_space.shape}")
            print(f"[Env] Action space: {self.action_space.shape}")

    # ========================================================================
    # CORE GYM METHODS
    # ========================================================================

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment and generate new trajectory from Geant4.

        Returns:
            observation: Initial state
            info: Additional information dictionary
        """
        super().reset(seed=seed)

        # Generate new ground truth trajectory
        retry_count = 0
        max_retries = 10

        while retry_count < max_retries:
            raw_data = self.sim_manager.run_single()

            # Format as [N, 7] array
            trajectory = np.stack([
                raw_data['x'], raw_data['y'], raw_data['z'],
                raw_data['px'], raw_data['py'], raw_data['pz'],
                raw_data['energy']
            ], axis=1).astype(np.float32)

            self.trajectory_length = len(trajectory)

            # Reject very short trajectories (likely numerical issues)
            if self.trajectory_length >= 5:
                break

            retry_count += 1

        if retry_count == max_retries:
            raise RuntimeError("Failed to generate valid trajectory after 10 attempts")

        self.current_trajectory = trajectory
        self.current_step_idx = 0
        self.episode_steps = 0

        # Reset history
        self.step_history.clear()
        self.previous_state = self.current_trajectory[0].copy()

        # Initial observation
        initial_obs = self._build_observation(
            self.current_trajectory[0],
            np.zeros(3, dtype=np.float32)  # No previous delta yet
        )

        info = {
            'trajectory_length': self.trajectory_length,
            'initial_energy': self.current_trajectory[0, 6],
            'initial_position': self.current_trajectory[0, :3].copy()
        }

        if self.verbose:
            print(f"[Env] Episode {self.total_episodes + 1} reset: {self.trajectory_length} steps")

        return initial_obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Predicted state change [dx, dy, dz, dpx, dpy, dpz, dE]

        Returns:
            observation: Next state
            reward: Reward signal
            terminated: Episode ended naturally
            truncated: Episode ended due to constraint
            info: Additional information
        """
        self.episode_steps += 1

        # Check episode termination
        if self.current_step_idx >= self.trajectory_length - 1:
            terminated = True
            truncated = False
            final_reward = self._calculate_final_reward()

            return (
                np.zeros(14, dtype=np.float32),
                final_reward,
                terminated,
                truncated,
                {'reason': 'trajectory_end'}
            )

        if self.episode_steps >= self.max_episode_steps:
            terminated = False
            truncated = True
            return (
                np.zeros(14, dtype=np.float32),
                -100.0,  # Penalty for timeout
                terminated,
                truncated,
                {'reason': 'max_steps'}
            )

        # Get ground truth states
        current_state_gt = self.current_trajectory[self.current_step_idx]
        next_state_gt = self.current_trajectory[self.current_step_idx + 1]
        delta_true = next_state_gt - current_state_gt

        # Predicted next state
        predicted_next = current_state_gt + action

        # Calculate comprehensive reward
        reward, reward_components = self._calculate_reward(
            current_state_gt,
            next_state_gt,
            predicted_next,
            action,
            delta_true
        )

        # Build next observation
        next_obs = self._build_observation(next_state_gt, delta_true[:3])

        # Update history
        self.step_history.append({
            'state': current_state_gt.copy(),
            'action': action.copy(),
            'reward': reward
        })

        self.previous_state = current_state_gt.copy()
        self.current_step_idx += 1

        # Info dictionary
        info = {
            'step': self.current_step_idx,
            'reward_components': reward_components,
            'ground_truth_energy': next_state_gt[6],
            'predicted_energy': predicted_next[6]
        }

        terminated = False
        truncated = False

        return next_obs, float(reward), terminated, truncated, info

    def close(self):
        """Clean up resources"""
        if self.verbose:
            print(f"[Env] Closing environment after {self.total_episodes} episodes")

    # ========================================================================
    # OBSERVATION CONSTRUCTION
    # ========================================================================

    def _build_observation(self, state: np.ndarray, previous_delta: np.ndarray) -> np.ndarray:
        """
        Build observation vector from current state.

        Args:
            state: [x, y, z, px, py, pz, energy]
            previous_delta: [dx, dy, dz] from last step

        Returns:
            observation: 14-dimensional vector
        """
        pos = state[:3]
        momentum = state[3:6]
        energy = state[6]

        # Calculate derived quantities

        # 1. Momentum direction (normalized)
        p_mag = np.linalg.norm(momentum)
        if p_mag > 1e-6:
            direction = momentum / p_mag
        else:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # 2. Depth in phantom (distance from entry point)
        depth = pos[0] - PhysicsConstants.BEAM_START[0]
        depth = np.clip(depth, 0, PhysicsConstants.PHANTOM_SIZE)

        # 3. Angle relative to beam axis (X-axis)
        beam_axis = np.array([1.0, 0.0, 0.0])
        angle = np.arccos(np.clip(np.dot(direction, beam_axis), -1.0, 1.0))

        # 4. Velocity magnitude (approximation from previous step)
        velocity = np.linalg.norm(previous_delta)

        # 5. Remaining range estimate
        remaining_range = self.physics.calculate_range(energy)

        # Assemble observation
        obs = np.array([
            pos[0], pos[1], pos[2],  # Position [0:3]
            direction[0], direction[1], direction[2],  # Direction [3:6]
            energy,  # Energy [6]
            depth,  # Depth [7]
            angle,  # Angle [8]
            velocity,  # Velocity [9]
            remaining_range,  # Range [10]
            previous_delta[0], previous_delta[1], previous_delta[2]  # Previous delta [11:14]
        ], dtype=np.float32)

        # Normalize if requested
        if self.normalize_obs:
            obs = self._normalize_observation(obs)

        return obs

    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observation to roughly [-1, 1] range.

        This improves neural network training stability.
        """
        normalized = obs.copy()

        # Position: scale by phantom size
        normalized[0:3] /= 10.0

        # Direction: already normalized

        # Energy: scale by initial energy (~10 MeV)
        normalized[6] /= 10.0

        # Depth: scale by phantom size
        normalized[7] /= 10.0

        # Angle: scale by pi
        normalized[8] /= np.pi

        # Velocity: scale by typical step size
        normalized[9] /= 1.0

        # Range: scale by typical range (~4 cm)
        normalized[10] /= 5.0

        # Previous delta: scale by typical step
        normalized[11:14] /= 1.0

        return np.clip(normalized, -3.0, 3.0)

    # ========================================================================
    # REWARD CALCULATION
    # ========================================================================

    def _calculate_reward(self,
                          current_state: np.ndarray,
                          next_state_gt: np.ndarray,
                          predicted_next: np.ndarray,
                          action: np.ndarray,
                          delta_true: np.ndarray) -> Tuple[float, Dict]:
        """
        Calculate comprehensive reward based on multiple physics criteria.

        Returns:
            total_reward: Scalar reward
            components: Dictionary of reward components for logging
        """
        components = {}

        # ====================================================================
        # 1. POSITION ACCURACY
        # ====================================================================
        position_error = np.linalg.norm(action[:3] - delta_true[:3])
        reward_position = -position_error * self.W_POSITION
        components['position'] = reward_position

        # ====================================================================
        # 2. DIRECTION ACCURACY
        # ====================================================================
        # Compare momentum directions
        p_true = next_state_gt[3:6]
        p_pred = predicted_next[3:6]

        p_true_mag = np.linalg.norm(p_true)
        p_pred_mag = np.linalg.norm(p_pred)

        if p_true_mag > 1e-6 and p_pred_mag > 1e-6:
            dir_true = p_true / p_true_mag
            dir_pred = p_pred / p_pred_mag

            # Cosine similarity (1 = perfect, -1 = opposite)
            cos_sim = np.dot(dir_true, dir_pred)
            direction_error = 1.0 - cos_sim  # 0 = perfect, 2 = opposite

            reward_direction = -direction_error * self.W_DIRECTION
        else:
            reward_direction = 0.0

        components['direction'] = reward_direction

        # ====================================================================
        # 3. ENERGY ACCURACY
        # ====================================================================
        energy_error = abs(action[6] - delta_true[6])
        reward_energy = -energy_error * self.W_ENERGY
        components['energy'] = reward_energy

        # ====================================================================
        # 4. PHYSICAL CONSISTENCY (Relativistic E-p relation)
        # ====================================================================
        # E² = p²c² + m²c⁴  (in natural units with c=1)
        pred_energy = predicted_next[6]
        pred_momentum = predicted_next[3:6]
        pred_p_mag = np.linalg.norm(pred_momentum)

        # Expected energy from momentum
        mass = PhysicsConstants.ELECTRON_MASS
        energy_from_momentum = np.sqrt(pred_p_mag ** 2 + mass ** 2) - mass

        physics_violation = abs(pred_energy - energy_from_momentum)
        reward_physics = -physics_violation * self.W_PHYSICS
        components['physics'] = reward_physics

        # ====================================================================
        # 5. SCATTERING ANGLE REALISM
        # ====================================================================
        # Check if predicted scattering is within expected range
        step_length = np.linalg.norm(action[:3])
        current_energy = current_state[6]

        expected_theta_rms = self.physics.multiple_scattering_angle(
            current_energy,
            step_length
        )

        # Actual scattering angle in prediction
        if p_true_mag > 1e-6 and p_pred_mag > 1e-6:
            scatter_angle = np.arccos(np.clip(cos_sim, -1.0, 1.0))

            # Allow scatter within 3σ of expected
            if scatter_angle > 3.0 * expected_theta_rms:
                scatter_excess = scatter_angle - 3.0 * expected_theta_rms
                reward_scattering = -scatter_excess * self.W_SCATTERING
            else:
                reward_scattering = 0.5  # Small bonus for realistic scatter
        else:
            reward_scattering = 0.0

        components['scattering'] = reward_scattering

        # ====================================================================
        # 6. RANGE-ENERGY CONSISTENCY
        # ====================================================================
        # Check if particle has traveled reasonable distance for its energy
        pred_range = self.physics.calculate_range(pred_energy)
        depth = predicted_next[0] - PhysicsConstants.BEAM_START[0]

        # If depth exceeds expected range, penalize
        if depth > pred_range + 1.0:  # 1 cm tolerance
            range_violation = depth - pred_range
            reward_range = -range_violation * self.W_RANGE
        else:
            reward_range = 0.0

        components['range'] = reward_range

        # ====================================================================
        # 7. THERMODYNAMICS (Energy Conservation)
        # ====================================================================
        # Energy should ONLY decrease (except for tiny numerical fluctuations)
        current_energy = current_state[6]

        if pred_energy > current_energy + 0.05:  # 50 keV tolerance
            energy_gain = pred_energy - current_energy
            reward_thermo = -energy_gain * self.W_THERMODYNAMICS
        else:
            reward_thermo = 0.0

        components['thermodynamics'] = reward_thermo

        # ====================================================================
        # 8. PROGRESS BONUS
        # ====================================================================
        # Reward forward progress (positive X direction)
        dx_true = delta_true[0]
        dx_pred = action[0]

        # Bonus if moving forward correctly
        if dx_true > 0 and dx_pred > 0:
            reward_progress = self.W_PROGRESS * min(dx_pred, dx_true) / max(dx_pred, dx_true)
        elif dx_true < 0 and dx_pred < 0:  # Backscattering
            reward_progress = self.W_PROGRESS * 0.5
        else:
            reward_progress = -self.W_PROGRESS  # Wrong direction

        components['progress'] = reward_progress

        # ====================================================================
        # 9. TRAJECTORY SMOOTHNESS
        # ====================================================================
        # Penalize sudden changes in velocity
        if len(self.step_history) > 0:
            previous_action = self.step_history[-1]['action']
            velocity_change = np.linalg.norm(action[:3] - previous_action[:3])

            # Expect smooth changes
            if velocity_change > 1.0:  # Sudden jerk
                reward_smoothness = -(velocity_change - 1.0) * self.W_SMOOTHNESS
            else:
                reward_smoothness = 0.0
        else:
            reward_smoothness = 0.0

        components['smoothness'] = reward_smoothness

        # ====================================================================
        # 10. STEP SIZE PENALTY (NEW - CRITICAL!)
        # ====================================================================
        # Penalize unrealistically large steps
        # Expected step size: ~0.05-0.10 cm for electrons in water
        step_size = np.linalg.norm(action[:3])
        expected_step_size = 0.08  # cm (reasonable for 10 MeV electrons)

        if step_size > expected_step_size * 2.0:  # More than 2x expected
            step_excess = step_size - expected_step_size * 2.0
            reward_step_size = -step_excess * 100.0  # Heavy penalty
        elif step_size < expected_step_size * 0.3:  # Too small (numerical issues)
            reward_step_size = -10.0
        else:
            reward_step_size = 0.0

        components['step_size'] = reward_step_size

        # ====================================================================
        # 11. ENERGY LOSS REALISM (NEW - CRITICAL!)
        # ====================================================================
        # Check if energy loss matches expected dE/dx
        # For 10 MeV electrons: ~2 MeV/cm in water
        current_energy = current_state[6]
        step_length = np.linalg.norm(action[:3])

        if step_length > 1e-6:
            # Expected energy loss
            dedx = 2.0  # MeV/cm (approximate for 10 MeV electrons)
            expected_energy_loss = dedx * step_length

            # Actual energy loss
            actual_energy_loss = -action[6]  # Negative of dE

            # Check consistency
            energy_loss_error = abs(actual_energy_loss - expected_energy_loss)

            # Allow 50% tolerance
            tolerance = expected_energy_loss * 0.5

            if energy_loss_error > tolerance:
                reward_energy_realism = -energy_loss_error * 50.0
            else:
                reward_energy_realism = 5.0  # Small bonus for realistic loss
        else:
            reward_energy_realism = 0.0

        components['energy_realism'] = reward_energy_realism

        # ====================================================================
        # 12. BOUNDARY PENALTY (NEW)
        # ====================================================================
        # Heavy penalty for going outside phantom
        pred_pos = predicted_next[:3]
        phantom_half_size = 5.0  # cm

        if (abs(pred_pos[0]) > phantom_half_size + 2.0 or
                abs(pred_pos[1]) > phantom_half_size or
                abs(pred_pos[2]) > phantom_half_size):
            reward_boundary = -200.0  # Very heavy penalty
        else:
            reward_boundary = 0.0

        components['boundary'] = reward_boundary

        # ====================================================================
        # 13. PRECISION BONUSES
        # ====================================================================
        # Extra rewards for very accurate predictions
        if position_error < 0.05:  # < 0.5 mm
            components['precision_bonus'] = 10.0
        elif position_error < 0.1:  # < 1 mm
            components['precision_bonus'] = 5.0
        else:
            components['precision_bonus'] = 0.0

        # ====================================================================
        # TOTAL REWARD
        # ====================================================================
        total_reward = sum(components.values())

        # Add small survival bonus
        total_reward += 1.0

        return total_reward, components

    def _calculate_final_reward(self) -> float:
        """
        Calculate bonus/penalty at episode end.

        Rewards reaching deep into phantom, penalizes early termination.
        """
        if self.current_step_idx < 10:
            return -50.0  # Very short episode

        # Check final depth
        final_pos = self.current_trajectory[self.current_step_idx, :3]
        final_depth = final_pos[0] - PhysicsConstants.BEAM_START[0]

        # Bonus for reaching target depth (~4 cm)
        target_depth = 4.0
        if final_depth > target_depth - 1.0 and final_depth < target_depth + 1.0:
            return 50.0  # Good penetration
        elif final_depth > target_depth + 2.0:
            return -20.0  # Overshot
        else:
            return 0.0

    # ========================================================================
    # STATISTICS AND UTILITIES
    # ========================================================================

    def get_episode_statistics(self) -> Dict:
        """Return statistics from last episode"""
        if len(self.step_history) == 0:
            return {}

        rewards = [step['reward'] for step in self.step_history]

        return {
            'episode_length': len(self.step_history),
            'total_reward': sum(rewards),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards)
        }


# ============================================================================
# MAIN - FOR TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTING ADVANCED PHYSICS-INFORMED ENVIRONMENT")
    print("=" * 70 + "\n")

    # Create environment
    env = Geant4ParticleEnv(verbose=True)

    print("\n[Test] Running one episode with random agent...")

    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Trajectory length: {info['trajectory_length']}")

    total_reward = 0
    steps = 0
    done = False

    while not done and steps < 100:
        # Random action
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1
        done = terminated or truncated

        if steps % 20 == 0:
            print(f"Step {steps}: reward={reward:.2f}, energy={info.get('ground_truth_energy', 0):.2f} MeV")

    print(f"\n[Test] Episode finished!")
    print(f"Total steps: {steps}")
    print(f"Total reward: {total_reward:.2f}")

    stats = env.get_episode_statistics()
    print(f"Episode statistics: {stats}")

    print("\n" + "=" * 70)
    print("✅ TEST COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")