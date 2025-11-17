"""
Simple test script to verify the ML-Agents environment setup
"""
import sys
import numpy as np


def test_python_version():
    """Test Python version"""
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")


def test_numpy():
    """Test NumPy installation"""
    print(f"\nNumPy version: {np.__version__}")
    arr = np.array([1, 2, 3, 4, 5])
    print(f"NumPy test array: {arr}")
    print(f"Array mean: {arr.mean()}")


def test_mlagents():
    """Test ML-Agents installation"""
    try:
        import mlagents
        print(f"\nML-Agents installed successfully!")
        print(f"ML-Agents package location: {mlagents.__file__}")
    except ImportError as e:
        print(f"\nML-Agents not found: {e}")


def main():
    print("=" * 50)
    print("ML-Agents Environment Test")
    print("=" * 50)

    test_python_version()
    test_numpy()
    test_mlagents()

    print("\n" + "=" * 50)
    print("Test completed successfully! ✓")
    print("=" * 50)


if __name__ == "__main__":
    main()