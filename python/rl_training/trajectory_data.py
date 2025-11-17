"""
Trajectory Data Structures
Data classes for particle trajectories
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np
from datetime import datetime


@dataclass
class ParticleStep:
    """Single step in particle trajectory"""

    step_number: int
    position: np.ndarray  # [x, y, z] in cm
    direction: np.ndarray  # [dx, dy, dz] unit vector
    energy: float  # MeV
    energy_deposit: float  # MeV deposited in this step
    step_length: float  # cm
    process: str = ""  # Physics process name

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'step_number': self.step_number,
            'position': self.position.tolist() if isinstance(self.position, np.ndarray) else self.position,
            'direction': self.direction.tolist() if isinstance(self.direction, np.ndarray) else self.direction,
            'energy': float(self.energy),
            'energy_deposit': float(self.energy_deposit),
            'step_length': float(self.step_length),
            'process': self.process
        }


@dataclass
class ParticleTrajectory:
    """Complete trajectory of a single particle"""

    # Identification
    trajectory_id: int
    agent_id: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Initial conditions
    initial_energy: float = 0.0  # MeV
    initial_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    initial_direction: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0]))
    particle_type: str = "e-"

    # Trajectory steps
    steps: List[ParticleStep] = field(default_factory=list)

    # Results
    total_energy_deposited: float = 0.0
    final_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    final_energy: float = 0.0
    num_steps: int = 0

    # Status
    completed: bool = False
    exit_reason: str = ""  # "stopped", "exited", "energy_depleted"

    # Metadata
    source: str = "unity"  # "unity" or "geant4"

    def add_step(self, step: ParticleStep):
        """Add a step to trajectory"""
        self.steps.append(step)
        self.num_steps = len(self.steps)

        # Update totals
        self.total_energy_deposited += step.energy_deposit
        self.final_position = step.position.copy()
        self.final_energy = step.energy

    def get_positions_array(self) -> np.ndarray:
        """Get all positions as numpy array (N, 3)"""
        if not self.steps:
            return np.array([]).reshape(0, 3)
        return np.array([step.position for step in self.steps])

    def get_energies_array(self) -> np.ndarray:
        """Get all energies as numpy array (N,)"""
        if not self.steps:
            return np.array([])
        return np.array([step.energy for step in self.steps])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'trajectory_id': self.trajectory_id,
            'agent_id': self.agent_id,
            'timestamp': self.timestamp,
            'initial_energy': float(self.initial_energy),
            'initial_position': self.initial_position.tolist(),
            'initial_direction': self.initial_direction.tolist(),
            'particle_type': self.particle_type,
            'steps': [step.to_dict() for step in self.steps],
            'total_energy_deposited': float(self.total_energy_deposited),
            'final_position': self.final_position.tolist(),
            'final_energy': float(self.final_energy),
            'num_steps': self.num_steps,
            'completed': self.completed,
            'exit_reason': self.exit_reason,
            'source': self.source
        }

    @classmethod
    def from_geant4_result(cls,
                           trajectory_id: int,
                           geant4_result: Dict[str, Any]) -> 'ParticleTrajectory':
        """
        Create trajectory from Geant4 simulation result

        Args:
            trajectory_id: Unique trajectory ID
            geant4_result: Result from Geant4 runner

        Returns:
            ParticleTrajectory object
        """
        params = geant4_result.get('parameters', {})

        trajectory = cls(
            trajectory_id=trajectory_id,
            initial_energy=params.get('particle_energy', 0.0),
            initial_position=np.array(params.get('particle_position', [0, 0, 0])),
            initial_direction=np.array(params.get('particle_direction', [1, 0, 0])),
            particle_type=params.get('particle_type', 'e-'),
            source='geant4'
        )

        # Parse events into steps
        if 'events' in geant4_result:
            for event in geant4_result['events']:
                if 'steps' in event:
                    for step_data in event['steps']:
                        step = ParticleStep(
                            step_number=step_data.get('step_number', 0),
                            position=np.array(step_data.get('position', [0, 0, 0])),
                            direction=np.array(step_data.get('direction', [1, 0, 0])),
                            energy=step_data.get('energy', 0.0),
                            energy_deposit=step_data.get('energy_deposit', 0.0),
                            step_length=step_data.get('step_length', 0.0),
                            process=step_data.get('process', '')
                        )
                        trajectory.add_step(step)

        # Set final values
        trajectory.total_energy_deposited = geant4_result.get('total_energy_deposit', 0.0)
        trajectory.completed = geant4_result.get('success', False)

        return trajectory

    @classmethod
    def from_unity_observation(cls,
                               trajectory_id: int,
                               agent_id: int,
                               observations: List[np.ndarray],
                               start_position: np.ndarray = np.array([-6.0, 0.0, 0.0]),
                               initial_energy_mev: float = 10.0) -> 'ParticleTrajectory':
        """
        Create trajectory from Unity agent observations

        Args:
            trajectory_id: Unique trajectory ID
            agent_id: Unity agent ID
            observations: List of observations [step0, step1, ...]
                         Each observation: [rel_pos_x, rel_pos_y, rel_pos_z,
                                           vel_x, vel_y, vel_z,
                                           energy_normalized, dir_x, dir_y, dir_z]
            start_position: Agent start position in world coordinates (default: [-6, 0, 0])
            initial_energy_mev: Initial energy in MeV (default: 10.0)

        Returns:
            ParticleTrajectory object
        """
        if not observations:
            raise ValueError("Empty observations list")

        first_obs = observations[0]

        trajectory = cls(
            trajectory_id=trajectory_id,
            agent_id=agent_id,
            initial_energy=initial_energy_mev,  # Real energy in MeV
            initial_position=start_position.copy(),  # Absolute position
            initial_direction=first_obs[7:10].copy(),
            source='unity'
        )

        # Convert observations to steps
        for i, obs in enumerate(observations):
            # Relative position from observation
            rel_position = obs[0:3]

            # Convert to absolute position
            abs_position = start_position + rel_position

            velocity = obs[3:6]
            energy_normalized = obs[6]
            direction = obs[7:10]

            # De-normalize energy
            energy = energy_normalized * initial_energy_mev

            # Estimate energy deposit (difference from previous step)
            if i > 0:
                prev_energy_normalized = observations[i - 1][6]
                prev_energy = prev_energy_normalized * initial_energy_mev
                energy_deposit = prev_energy - energy
            else:
                energy_deposit = 0.0

            # Estimate step length
            if i > 0:
                prev_rel_pos = observations[i - 1][0:3]
                prev_abs_pos = start_position + prev_rel_pos
                step_length = np.linalg.norm(abs_position - prev_abs_pos)
            else:
                step_length = 0.0

            step = ParticleStep(
                step_number=i,
                position=abs_position,  # Absolute position!
                direction=direction,
                energy=energy,  # Real energy in MeV!
                energy_deposit=energy_deposit,
                step_length=step_length,
                process="unity_step"
            )

            trajectory.add_step(step)

        return trajectory


@dataclass
class TrajectoryPair:
    """Pair of Unity and Geant4 trajectories for comparison"""

    pair_id: int
    unity_trajectory: ParticleTrajectory
    geant4_trajectory: ParticleTrajectory = None

    # Comparison metrics
    position_distance: float = 0.0  # Average distance between trajectories
    energy_difference: float = 0.0  # Difference in total energy deposit
    step_count_ratio: float = 0.0  # Unity steps / Geant4 steps

    # Reward
    reward: float = 0.0

    def calculate_metrics(self):
        """Calculate comparison metrics"""
        if self.geant4_trajectory is None:
            return

        unity_pos = self.unity_trajectory.get_positions_array()
        geant4_pos = self.geant4_trajectory.get_positions_array()

        # Position distance (using DTW or simple point-to-point)
        min_len = min(len(unity_pos), len(geant4_pos))
        if min_len > 0:
            distances = np.linalg.norm(unity_pos[:min_len] - geant4_pos[:min_len], axis=1)
            self.position_distance = float(np.mean(distances))

        # Energy difference
        self.energy_difference = abs(
            self.unity_trajectory.total_energy_deposited -
            self.geant4_trajectory.total_energy_deposited
        )

        # Step count ratio
        if self.geant4_trajectory.num_steps > 0:
            self.step_count_ratio = (
                    self.unity_trajectory.num_steps / self.geant4_trajectory.num_steps
            )