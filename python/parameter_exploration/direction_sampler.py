"""
4π Uniform Direction Sampling
Generates uniformly distributed particle directions on unit sphere
"""

import numpy as np
from typing import Optional


class DirectionSampler:
    """
    Sample particle directions uniformly over 4π steradians

    Meeting requirement: "rozkład jednorodny losujemy 4pi"
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize direction sampler

        Args:
            random_seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(random_seed)

    def sample_4pi_uniform(self, n_samples: int = 1) -> np.ndarray:
        """
        Sample n directions uniformly on unit sphere (4π steradians)

        Uses spherical coordinates:
        - θ ∈ [0, π]   (polar angle from z-axis)
        - φ ∈ [0, 2π]  (azimuthal angle in x-y plane)

        For uniform distribution on sphere:
        - cos(θ) must be uniform in [-1, 1]
        - φ must be uniform in [0, 2π]

        Args:
            n_samples: Number of directions to sample

        Returns:
            Array of shape (n_samples, 3) with unit direction vectors

        Example:
            >>> sampler = DirectionSampler(seed=42)
            >>> directions = sampler.sample_4pi_uniform(5)
            >>> print(directions.shape)
            (5, 3)
            >>> print(np.linalg.norm(directions, axis=1))  # Should be all 1.0
            [1. 1. 1. 1. 1.]
        """
        # Sample cos(θ) uniformly in [-1, 1] for uniform solid angle
        cos_theta = self.rng.uniform(-1.0, 1.0, n_samples)
        theta = np.arccos(cos_theta)

        # Sample φ uniformly in [0, 2π]
        phi = self.rng.uniform(0.0, 2.0 * np.pi, n_samples)

        # Convert spherical to Cartesian coordinates
        sin_theta = np.sin(theta)

        x = sin_theta * np.cos(phi)
        y = sin_theta * np.sin(phi)
        z = cos_theta

        # Stack into (n_samples, 3) array
        directions = np.column_stack([x, y, z])

        return directions

    def sample_hemisphere(self,
                          normal: np.ndarray,
                          n_samples: int = 1) -> np.ndarray:
        """
        Sample directions in hemisphere defined by normal vector

        Args:
            normal: Normal vector defining hemisphere (will be normalized)
            n_samples: Number of directions to sample

        Returns:
            Array of shape (n_samples, 3) with directions in hemisphere
        """
        # Normalize normal vector
        normal = normal / np.linalg.norm(normal)

        # Sample full sphere
        directions = self.sample_4pi_uniform(n_samples * 2)  # Oversample

        # Keep only directions in hemisphere (dot product > 0)
        dots = directions @ normal
        hemisphere_mask = dots > 0
        hemisphere_directions = directions[hemisphere_mask]

        # Resample if not enough
        while len(hemisphere_directions) < n_samples:
            extra = self.sample_4pi_uniform(n_samples)
            extra_dots = extra @ normal
            hemisphere_directions = np.vstack([
                hemisphere_directions,
                extra[extra_dots > 0]
            ])

        return hemisphere_directions[:n_samples]

    def sample_forward_cone(self,
                            forward_direction: np.ndarray,
                            cone_angle_deg: float,
                            n_samples: int = 1) -> np.ndarray:
        """
        Sample directions within cone around forward direction

        Useful for beam-like initial conditions

        Args:
            forward_direction: Central direction of cone
            cone_angle_deg: Half-angle of cone in degrees
            n_samples: Number of directions to sample

        Returns:
            Array of directions within cone
        """
        # Normalize forward direction
        forward = forward_direction / np.linalg.norm(forward_direction)

        # Convert cone angle to radians
        cone_angle_rad = np.deg2rad(cone_angle_deg)
        cos_cone_angle = np.cos(cone_angle_rad)

        # Sample full sphere
        directions = self.sample_4pi_uniform(n_samples * 3)  # Oversample

        # Keep only directions within cone
        dots = directions @ forward
        cone_mask = dots >= cos_cone_angle
        cone_directions = directions[cone_mask]

        # Resample if needed
        while len(cone_directions) < n_samples:
            extra = self.sample_4pi_uniform(n_samples)
            extra_dots = extra @ forward
            cone_directions = np.vstack([
                cone_directions,
                extra[extra_dots >= cos_cone_angle]
            ])

        return cone_directions[:n_samples]

    def visualize_samples(self, n_samples: int = 1000):
        """
        Visualize sampled directions (for debugging)

        Requires matplotlib
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("matplotlib required for visualization")
            return

        directions = self.sample_4pi_uniform(n_samples)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(directions[:, 0],
                   directions[:, 1],
                   directions[:, 2],
                   c='blue', alpha=0.1, s=1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'4π Uniform Sampling ({n_samples} samples)')

        # Set equal aspect ratio
        max_range = 1.0
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])

        plt.show()


# === Unity C# VERSION (for reference) ===
"""
using UnityEngine;

public class DirectionSampler : MonoBehaviour
{
    /// <summary>
    /// Sample uniform direction on 4π sphere
    /// </summary>
    public static Vector3 Sample4PiUniform()
    {
        // Sample cos(θ) uniformly in [-1, 1]
        float cosTheta = Random.Range(-1f, 1f);
        float theta = Mathf.Acos(cosTheta);

        // Sample φ uniformly in [0, 2π]
        float phi = Random.Range(0f, 2f * Mathf.PI);

        // Convert to Cartesian
        float sinTheta = Mathf.Sin(theta);

        float x = sinTheta * Mathf.Cos(phi);
        float y = sinTheta * Mathf.Sin(phi);
        float z = cosTheta;

        return new Vector3(x, y, z);
    }
}
"""