"""
Test 4π Uniform Direction Sampling
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from parameter_exploration.direction_sampler import DirectionSampler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def test_unit_length():
    """Test that all sampled directions are unit vectors"""

    print("\n" + "=" * 60)
    print("TEST 1: Unit Length")
    print("=" * 60)

    sampler = DirectionSampler(random_seed=42)
    directions = sampler.sample_4pi_uniform(1000)

    # Check all directions are unit length
    lengths = np.linalg.norm(directions, axis=1)

    print(f"\nSampled {len(directions)} directions")
    print(f"Mean length: {np.mean(lengths):.6f}")
    print(f"Std length: {np.std(lengths):.6f}")
    print(f"Min length: {np.min(lengths):.6f}")
    print(f"Max length: {np.max(lengths):.6f}")

    # All should be very close to 1.0
    assert np.allclose(lengths, 1.0, atol=1e-10), "Not all directions are unit vectors!"

    print("✅ PASSED")


def test_mean_near_zero():
    """Test that mean of many samples is near origin (uniformity)"""

    print("\n" + "=" * 60)
    print("TEST 2: Mean Near Zero (Uniformity)")
    print("=" * 60)

    sampler = DirectionSampler(random_seed=42)
    directions = sampler.sample_4pi_uniform(10000)

    # Mean should be near zero for uniform distribution
    mean = np.mean(directions, axis=0)
    mean_magnitude = np.linalg.norm(mean)

    print(f"\nSampled {len(directions)} directions")
    print(f"Mean vector: {mean}")
    print(f"Mean magnitude: {mean_magnitude:.6f}")

    # With 10k samples, mean should be very close to zero
    assert mean_magnitude < 0.05, f"Mean not near zero: {mean_magnitude}"

    print("✅ PASSED")


def test_hemisphere():
    """Test hemisphere sampling"""

    print("\n" + "=" * 60)
    print("TEST 3: Hemisphere Sampling")
    print("=" * 60)

    sampler = DirectionSampler(random_seed=42)

    # Sample hemisphere with normal pointing up (+Z)
    normal = np.array([0, 0, 1])
    directions = sampler.sample_hemisphere(normal, n_samples=1000)

    # All directions should have positive Z component
    z_components = directions[:, 2]

    print(f"\nSampled {len(directions)} directions in hemisphere")
    print(f"Min Z: {np.min(z_components):.6f}")
    print(f"Max Z: {np.max(z_components):.6f}")
    print(f"All positive Z: {np.all(z_components > 0)}")

    assert np.all(z_components > 0), "Some directions not in hemisphere!"

    print("✅ PASSED")


def test_cone():
    """Test cone sampling"""

    print("\n" + "=" * 60)
    print("TEST 4: Cone Sampling (30 degrees)")
    print("=" * 60)

    sampler = DirectionSampler(random_seed=42)

    # Sample within 30° cone around +X axis
    forward = np.array([1, 0, 0])
    cone_angle = 30.0  # degrees
    directions = sampler.sample_forward_cone(forward, cone_angle, n_samples=1000)

    # All directions should be within cone
    dots = directions @ forward
    angles = np.arccos(np.clip(dots, -1, 1))
    angles_deg = np.rad2deg(angles)

    print(f"\nSampled {len(directions)} directions in 30° cone")
    print(f"Max angle: {np.max(angles_deg):.2f}°")
    print(f"All within cone: {np.all(angles_deg <= cone_angle)}")

    assert np.all(angles_deg <= cone_angle + 0.1), "Some directions outside cone!"

    print("✅ PASSED")


def test_coordinate_distribution():
    """Test that each coordinate is uniformly distributed"""

    print("\n" + "=" * 60)
    print("TEST 5: Coordinate Distribution")
    print("=" * 60)

    sampler = DirectionSampler(random_seed=42)
    directions = sampler.sample_4pi_uniform(10000)

    # Each coordinate should have mean ~0 and std ~1/sqrt(3) ≈ 0.577
    x_mean = np.mean(directions[:, 0])
    y_mean = np.mean(directions[:, 1])
    z_mean = np.mean(directions[:, 2])

    x_std = np.std(directions[:, 0])
    y_std = np.std(directions[:, 1])
    z_std = np.std(directions[:, 2])

    expected_std = 1.0 / np.sqrt(3)

    print(f"\nCoordinate statistics:")
    print(f"X: mean={x_mean:.4f}, std={x_std:.4f}")
    print(f"Y: mean={y_mean:.4f}, std={y_std:.4f}")
    print(f"Z: mean={z_mean:.4f}, std={z_std:.4f}")
    print(f"Expected std: {expected_std:.4f}")

    # Check means are near zero
    assert abs(x_mean) < 0.05
    assert abs(y_mean) < 0.05
    assert abs(z_mean) < 0.05

    # Check std is close to 1/sqrt(3)
    assert abs(x_std - expected_std) < 0.05
    assert abs(y_std - expected_std) < 0.05
    assert abs(z_std - expected_std) < 0.05

    print("✅ PASSED")


def visualize_sampling(save_plot=True):
    """Visualize sampled directions"""

    print("\n" + "=" * 60)
    print("VISUALIZATION: 4π Sampling")
    print("=" * 60)

    sampler = DirectionSampler(random_seed=42)
    directions = sampler.sample_4pi_uniform(2000)

    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(directions[:, 0],
               directions[:, 1],
               directions[:, 2],
               c='blue', alpha=0.3, s=10, edgecolors='none')

    # Draw sphere wireframe
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.1, linewidth=0.5)

    # Axis lines
    axis_length = 1.2
    ax.plot([-axis_length, axis_length], [0, 0], [0, 0], 'r-', linewidth=2, label='X')
    ax.plot([0, 0], [-axis_length, axis_length], [0, 0], 'g-', linewidth=2, label='Y')
    ax.plot([0, 0], [0, 0], [-axis_length, axis_length], 'b-', linewidth=2, label='Z')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('4π Uniform Direction Sampling (2000 samples)', fontsize=14, fontweight='bold')
    ax.legend()

    # Set equal aspect ratio
    max_range = 1.2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    ax.set_box_aspect([1, 1, 1])

    if save_plot:
        output_path = Path(__file__).parent.parent / "analysis_results" / "direction_sampling_4pi.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Plot saved to: {output_path}")

    plt.show()

    print("✅ Visualization complete")


def main():
    """Run all tests"""

    print("\n" + "🧪" * 30)
    print("DIRECTION SAMPLING TESTS")
    print("🧪" * 30)

    try:
        test_unit_length()
        test_mean_near_zero()
        test_hemisphere()
        test_cone()
        test_coordinate_distribution()

        # Optional: visualize (comment out if you don't want plots)
        visualize_sampling(save_plot=True)

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())