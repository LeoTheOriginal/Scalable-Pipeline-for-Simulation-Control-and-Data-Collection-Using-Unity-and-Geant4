"""
Real-time Geant4 Step-by-Step Simulator
Executes Geant4 one step at a time for immediate comparison with Unity
"""

import subprocess
import numpy as np
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import os

logger = logging.getLogger(__name__)


class RealtimeGeant4Simulator:
    """
    Manages Geant4 simulation with step-by-step control
    Each instance handles one particle trajectory
    """

    def __init__(self, geant4_executable: str):
        """
        Initialize simulator

        Args:
            geant4_executable: Path to Geant4 executable
        """
        self.geant4_executable = geant4_executable

        # Current particle state
        self.particle_type: Optional[str] = None
        self.current_position: Optional[np.ndarray] = None
        self.current_energy: Optional[float] = None
        self.current_direction: Optional[np.ndarray] = None

        # Particle history
        self.step_history = []
        self.step_count = 0

        # Geant4 process (if using persistent process)
        self.process: Optional[subprocess.Popen] = None

        # Temporary directory for this simulator
        self.temp_dir = tempfile.mkdtemp(prefix="geant4_realtime_")

        logger.info(f"RealtimeGeant4Simulator initialized: {self.temp_dir}")

    def initialize_particle(self,
                           particle_type: str,
                           energy: float,
                           position: np.ndarray,
                           direction: np.ndarray):
        """
        Initialize particle with given conditions

        Args:
            particle_type: "e-", "gamma", "proton", etc.
            energy: Initial energy in MeV
            position: Initial position [x, y, z] in cm
            direction: Initial direction [dx, dy, dz] (will be normalized)
        """
        self.particle_type = particle_type
        self.current_position = np.array(position, dtype=float)
        self.current_energy = float(energy)
        self.current_direction = np.array(direction, dtype=float)
        self.current_direction /= np.linalg.norm(self.current_direction)  # Normalize

        # Reset history
        self.step_history = []
        self.step_count = 0

        # Record initial state
        self.step_history.append({
            'step': 0,
            'position': self.current_position.copy(),
            'energy': self.current_energy,
            'direction': self.current_direction.copy(),
            'energy_deposited': 0.0,
            'step_length': 0.0,
            'process': 'initialization'
        })

        logger.info(f"Particle initialized: {particle_type}, {energy:.2f} MeV, "
                   f"pos={position}, dir={direction}")

    def execute_step(self,
                    unity_position: np.ndarray,
                    unity_energy: float,
                    unity_direction: np.ndarray) -> Dict[str, Any]:
        """
        Execute single Geant4 step and return results

        This is the KEY method for per-step RL!

        Args:
            unity_position: Current Unity position [x, y, z] cm
            unity_energy: Current Unity energy MeV
            unity_direction: Current Unity direction [dx, dy, dz]

        Returns:
            {
                'position': np.ndarray [x, y, z],
                'energy': float,
                'direction': np.ndarray [dx, dy, dz],
                'energy_deposited': float,
                'step_length': float,
                'process_name': str,
                'particle_stopped': bool
            }
        """
        if self.particle_type is None:
            raise RuntimeError("Particle not initialized! Call initialize_particle() first")

        # For now, we use a SIMPLIFIED physics model
        # In production, this would communicate with actual Geant4 via pipes/sockets

        # OPTION 1: Simplified physics (for rapid prototyping)
        result = self._execute_simplified_step()

        # OPTION 2: Full Geant4 (uncomment when C++ step controller is ready)
        # result = self._execute_geant4_step()

        self.step_count += 1
        self.step_history.append(result)

        # Update current state
        self.current_position = result['position']
        self.current_energy = result['energy']
        self.current_direction = result['direction']

        return result

    def _execute_simplified_step(self) -> Dict[str, Any]:
        """
        Simplified physics model for rapid development

        This approximates Geant4 behavior without running full simulation
        Used for testing and development

        Physics model:
        - Continuous energy loss (Bethe-Bloch approximation)
        - Small angle scattering (Gaussian)
        - Step length based on energy
        """
        # Step length depends on energy (simple model)
        # Higher energy = longer steps
        step_length = min(0.1 + self.current_energy * 0.01, 0.5)  # cm

        # Energy loss per step (simplified Bethe-Bloch)
        # dE/dx ≈ constant for water
        energy_loss_per_cm = 2.0  # MeV/cm (rough approximation for electrons in water)
        energy_deposited = energy_loss_per_cm * step_length

        # Don't lose more energy than we have
        energy_deposited = min(energy_deposited, self.current_energy * 0.9)
        new_energy = self.current_energy - energy_deposited

        # Small angle scattering (Gaussian)
        scatter_angle = np.random.normal(0, 0.05)  # radians

        # Rotate direction slightly
        # Simple rotation around random axis
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)

        # Rodrigues' rotation formula
        new_direction = self._rotate_vector(
            self.current_direction,
            axis,
            scatter_angle
        )

        # Move along direction
        new_position = self.current_position + self.current_direction * step_length

        # Check if particle stopped
        particle_stopped = new_energy < 0.01  # MeV threshold

        return {
            'position': new_position,
            'energy': new_energy,
            'direction': new_direction,
            'energy_deposited': energy_deposited,
            'step_length': step_length,
            'process_name': 'eIoni' if not particle_stopped else 'Stopped',
            'particle_stopped': particle_stopped
        }

    def _execute_geant4_step(self) -> Dict[str, Any]:
        """
        Execute actual Geant4 step (requires StepController C++ implementation)

        This will communicate with Geant4 via JSON/pipes when ready
        """
        # Create input JSON for Geant4
        input_data = {
            'command': 'step',
            'particle_type': self.particle_type,
            'position': self.current_position.tolist(),
            'energy': float(self.current_energy),
            'direction': self.current_direction.tolist()
        }

        input_file = Path(self.temp_dir) / f"step_{self.step_count}_input.json"
        output_file = Path(self.temp_dir) / f"step_{self.step_count}_output.json"

        # Write input
        with open(input_file, 'w') as f:
            json.dump(input_data, f)

        # Run Geant4 in step mode
        try:
            result = subprocess.run(
                [
                    self.geant4_executable,
                    '--step-mode',
                    '--input', str(input_file),
                    '--output', str(output_file)
                ],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                logger.error(f"Geant4 step failed: {result.stderr}")
                raise RuntimeError(f"Geant4 execution failed: {result.stderr}")

            # Read output
            with open(output_file, 'r') as f:
                output_data = json.load(f)

            return {
                'position': np.array(output_data['position']),
                'energy': output_data['energy'],
                'direction': np.array(output_data['direction']),
                'energy_deposited': output_data['energy_deposited'],
                'step_length': output_data['step_length'],
                'process_name': output_data['process_name'],
                'particle_stopped': output_data['particle_stopped']
            }

        except subprocess.TimeoutExpired:
            logger.error("Geant4 step timeout")
            raise RuntimeError("Geant4 step timeout")
        except Exception as e:
            logger.error(f"Geant4 step error: {e}")
            raise

    def _rotate_vector(self, v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate vector v around axis by angle using Rodrigues' formula

        Args:
            v: Vector to rotate
            axis: Rotation axis (normalized)
            angle: Rotation angle in radians

        Returns:
            Rotated vector
        """
        axis = axis / np.linalg.norm(axis)

        v_rot = (v * np.cos(angle) +
                np.cross(axis, v) * np.sin(angle) +
                axis * np.dot(axis, v) * (1 - np.cos(angle)))

        return v_rot / np.linalg.norm(v_rot)  # Normalize

    def get_trajectory(self) -> list:
        """Get full particle trajectory"""
        return self.step_history

    def reset(self):
        """Reset simulator state"""
        self.particle_type = None
        self.current_position = None
        self.current_energy = None
        self.current_direction = None
        self.step_history = []
        self.step_count = 0

        logger.info("Simulator reset")

    def __del__(self):
        """Cleanup"""
        # Remove temporary directory
        import shutil
        try:
            if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
                shutil.rmtree(self.temp_dir)
        except:
            pass