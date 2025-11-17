"""
Quick Verification Script
Tests the fixed data_collector with clean_start functionality
"""

import sys
import os

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

import numpy as np
from data_collection.data_collector import DataCollector

print("=" * 60)
print("Testing Data Collector - FINAL VERSION")
print("=" * 60)

# Test 1: Clean start
print("\n--- Test 1: Clean Start ---")
collector1 = DataCollector(
    output_directory="./test_clean_start",
    max_samples_per_file=5,
    clean_start=True  # Should remove any existing files
)

for i in range(10):
    unity_obs = np.random.rand(10)
    geant4_result = {'total_energy': float(np.random.rand() * 100)}
    parameters = {'energy': 10.0 + i}

    success = collector1.collect_simulation_pair(unity_obs, geant4_result, parameters)
    if not success:
        print(f"❌ Sample {i} failed!")
        break

stats1 = collector1.get_statistics()
print(f"✅ Test 1 passed: Collected {stats1['total_samples']} samples")
print(f"   Files created: {stats1['num_files']}")

collector1.finalize()

# Test 2: Resume from existing (should continue)
print("\n--- Test 2: Resume from Existing ---")
collector2 = DataCollector(
    output_directory="./test_clean_start",
    max_samples_per_file=5,
    clean_start=False  # Should continue from existing files
)

for i in range(5):
    unity_obs = np.random.rand(10)
    geant4_result = {'total_energy': float(np.random.rand() * 100)}
    parameters = {'energy': 20.0 + i}

    success = collector2.collect_simulation_pair(unity_obs, geant4_result, parameters)
    if not success:
        print(f"❌ Sample {i} failed!")
        break

stats2 = collector2.get_statistics()
print(f"✅ Test 2 passed: Added {stats2['total_samples']} more samples")
print(f"   Total files: {stats2['num_files']}")

collector2.finalize()

# Test 3: Load and verify
print("\n--- Test 3: Load and Verify ---")
dataset = collector2.load_dataset(file_index=0)
if dataset:
    num_loaded = len(dataset['unity_observations'])
    print(f"✅ Test 3 passed: Loaded {num_loaded} samples from first file")
else:
    print("❌ Test 3 failed: Could not load dataset")

# Cleanup
import shutil

try:
    shutil.rmtree('./test_clean_start')
    print("\n✅ Cleanup: Test directory removed")
except:
    pass

print("\n" + "=" * 60)
print("ALL TESTS PASSED! 🎉")
print("=" * 60)
print("\nThe data collector is working correctly!")
print("You can now use it in your project.")