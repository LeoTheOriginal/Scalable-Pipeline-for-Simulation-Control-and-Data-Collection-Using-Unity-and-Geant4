"""
Inspect collected training data
"""

import h5py
import json
import numpy as np
from pathlib import Path


def inspect_hdf5_file(filepath: str):
    """Inspect HDF5 file structure and content"""

    print("\n" + "=" * 70)
    print(f"INSPECTING: {filepath}")
    print("=" * 70)

    with h5py.File(filepath, 'r') as f:
        # Count samples
        samples = [key for key in f.keys() if key.startswith('sample_')]
        print(f"\n📊 Total samples: {len(samples)}")

        if not samples:
            print("❌ No samples found!")
            return

        # Show first sample structure
        print(f"\n🔍 First sample structure: {samples[0]}")
        first_sample = f[samples[0]]

        def print_structure(group, indent=0):
            """Recursively print HDF5 structure"""
            for key in group.keys():
                item = group[key]
                prefix = "  " * indent

                if isinstance(item, h5py.Group):
                    print(f"{prefix}📁 {key}/")
                    print_structure(item, indent + 1)
                elif isinstance(item, h5py.Dataset):
                    shape = item.shape
                    dtype = item.dtype
                    print(f"{prefix}📄 {key}: shape={shape}, dtype={dtype}")

        print_structure(first_sample, indent=1)

        # Show sample data
        print(f"\n📈 Sample data from {samples[0]}:")

        # Unity observation
        if 'unity_observation' in first_sample:
            unity_obs = first_sample['unity_observation'][:]
            print(f"\n  Unity Observation:")
            print(f"    Shape: {unity_obs.shape}")
            print(f"    Values: {unity_obs.flatten()[:10]}...")  # First 10

        # Geant4 results
        if 'geant4_result' in first_sample:
            g4_group = first_sample['geant4_result']
            print(f"\n  Geant4 Results:")

            if 'total_energy_deposit' in g4_group:
                energy = g4_group['total_energy_deposit'][()]
                print(f"    Energy deposit: {energy:.3f} MeV")

            if 'num_events' in g4_group:
                num_events = g4_group['num_events'][()]
                print(f"    Num events: {num_events}")

            if 'result_json' in g4_group:
                json_data = g4_group['result_json'][()]
                if isinstance(json_data, bytes):
                    json_data = json_data.decode('utf-8')
                result = json.loads(json_data)
                print(f"    Full result available: {list(result.keys())}")

        # Parameters
        if 'parameters' in first_sample:
            params_group = first_sample['parameters']
            print(f"\n  Parameters:")

            # Attributes
            for key, value in params_group.attrs.items():
                print(f"    {key}: {value}")

            # Datasets
            for key in params_group.keys():
                value = params_group[key][()]
                if isinstance(value, np.ndarray):
                    print(f"    {key}: {value}")
                else:
                    print(f"    {key}: {value}")

        # Statistics across all samples
        print(f"\n📊 Statistics across all samples:")

        energies = []
        for sample_key in samples:
            sample = f[sample_key]
            if 'geant4_result' in sample and 'total_energy_deposit' in sample['geant4_result']:
                energy = sample['geant4_result']['total_energy_deposit'][()]
                energies.append(energy)

        if energies:
            print(f"    Energy deposits:")
            print(f"      Count: {len(energies)}")
            print(f"      Mean: {np.mean(energies):.3f} MeV")
            print(f"      Std: {np.std(energies):.3f} MeV")
            print(f"      Min: {np.min(energies):.3f} MeV")
            print(f"      Max: {np.max(energies):.3f} MeV")


def load_and_test_dataset():
    """Load dataset using DataCollector"""
    print("\n" + "=" * 70)
    print("LOADING DATASET WITH DATA_COLLECTOR")
    print("=" * 70)

    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from data_collection.data_collector import DataCollector

    collector = DataCollector(output_directory=r"C:\Thesis\python\training_data")

    # Load dataset
    dataset = collector.load_dataset(file_index=0)

    if dataset:
        print(f"\n✅ Successfully loaded dataset!")
        print(f"   Unity observations: {len(dataset['unity_observations'])} samples")
        print(f"   Geant4 results: {len(dataset['geant4_results'])} samples")
        print(f"   Parameters: {len(dataset['parameters'])} samples")

        if dataset['unity_observations']:
            print(f"\n📊 First Unity observation:")
            print(f"   Shape: {dataset['unity_observations'][0].shape}")
            print(f"   Values: {dataset['unity_observations'][0].flatten()}")

        if dataset['geant4_results']:
            print(f"\n📊 First Geant4 result:")
            for key, value in dataset['geant4_results'][0].items():
                if key != 'full_result':
                    print(f"   {key}: {value}")

        if dataset['parameters']:
            print(f"\n📊 First parameters:")
            for key, value in dataset['parameters'][0].items():
                print(f"   {key}: {value}")
    else:
        print("❌ Failed to load dataset")


def main():
    """Main inspection routine"""

    print("\n" + "🔬" * 35)
    print("TRAINING DATA INSPECTION")
    print("🔬" * 35)

    data_dir = Path(r"C:\Thesis\python\training_data")

    # Check directory exists
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return

    # List files
    print(f"\n📁 Contents of {data_dir}:")
    for item in sorted(data_dir.iterdir()):
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"   📄 {item.name:30s} ({size_mb:.2f} MB)")
        elif item.is_dir():
            num_items = len(list(item.iterdir()))
            print(f"   📁 {item.name:30s} ({num_items} items)")

    # Check metadata
    metadata_file = data_dir / "metadata.json"
    if metadata_file.exists():
        print(f"\n📋 Metadata:")
        with open(metadata_file) as f:
            metadata = json.load(f)
            for key, value in metadata.items():
                print(f"   {key}: {value}")

    # Inspect HDF5 file
    hdf5_files = list(data_dir.glob("data_*.hdf5"))
    if hdf5_files:
        for hdf5_file in hdf5_files:
            inspect_hdf5_file(str(hdf5_file))
    else:
        print("\n❌ No HDF5 files found!")

    # Test loading with DataCollector
    if hdf5_files:
        load_and_test_dataset()

    print("\n" + "=" * 70)
    print("✅ INSPECTION COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()