"""
Geant4 Data Parser
Parses and processes Geant4 simulation output files
"""

import numpy as np
import h5py
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class Geant4DataParser:
    """
    Parses Geant4 simulation output and extracts relevant data
    """

    def __init__(self, output_format: str = 'root'):
        """
        Initialize data parser

        Args:
            output_format: Format of Geant4 output ('root', 'hdf5', 'csv')
        """
        self.output_format = output_format.lower()

        # Check if required libraries are available
        if self.output_format == 'root':
            try:
                import uproot
                self.uproot = uproot
            except ImportError:
                logger.warning("uproot not installed. ROOT file parsing will not work.")
                logger.warning("Install with: pip install uproot")

    def parse_root_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse ROOT file from Geant4 simulation

        Args:
            file_path: Path to ROOT file

        Returns:
            dict: Parsed simulation data
        """
        if not hasattr(self, 'uproot'):
            raise ImportError("uproot is required for ROOT file parsing")

        try:
            # Open ROOT file
            root_file = self.uproot.open(file_path)

            # Extract data from trees
            # Note: Tree names and structure depend on your Geant4 setup
            data = {}

            # Example: Extract energy deposition data
            if 'Hits' in root_file:
                hits_tree = root_file['Hits']
                data['energy_deposition'] = hits_tree['edep'].array(library='np')
                data['x_position'] = hits_tree['x'].array(library='np')
                data['y_position'] = hits_tree['y'].array(library='np')
                data['z_position'] = hits_tree['z'].array(library='np')

            # Example: Extract particle information
            if 'Particles' in root_file:
                particle_tree = root_file['Particles']
                data['particle_energy'] = particle_tree['energy'].array(library='np')
                data['particle_type'] = particle_tree['pdg'].array(library='np')

            logger.info(f"Successfully parsed ROOT file: {file_path}")
            return data

        except Exception as e:
            logger.error(f"Error parsing ROOT file {file_path}: {e}")
            return {}

    def parse_hdf5_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse HDF5 file from Geant4 simulation

        Args:
            file_path: Path to HDF5 file

        Returns:
            dict: Parsed simulation data
        """
        try:
            data = {}

            with h5py.File(file_path, 'r') as f:
                # Extract datasets
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        data[key] = f[key][:]
                    elif isinstance(f[key], h5py.Group):
                        # Handle groups
                        group_data = {}
                        for subkey in f[key].keys():
                            if isinstance(f[key][subkey], h5py.Dataset):
                                group_data[subkey] = f[key][subkey][:]
                        data[key] = group_data

            logger.info(f"Successfully parsed HDF5 file: {file_path}")
            return data

        except Exception as e:
            logger.error(f"Error parsing HDF5 file {file_path}: {e}")
            return {}

    def parse_csv_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse CSV file from Geant4 simulation

        Args:
            file_path: Path to CSV file

        Returns:
            dict: Parsed simulation data
        """
        try:
            data = np.genfromtxt(file_path, delimiter=',', names=True)

            # Convert to dictionary of arrays
            parsed_data = {name: data[name] for name in data.dtype.names}

            logger.info(f"Successfully parsed CSV file: {file_path}")
            return parsed_data

        except Exception as e:
            logger.error(f"Error parsing CSV file {file_path}: {e}")
            return {}

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse simulation output file (auto-detect format)

        Args:
            file_path: Path to output file

        Returns:
            dict: Parsed simulation data
        """
        path = Path(file_path)

        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return {}

        # Detect format from extension
        extension = path.suffix.lower()

        if extension == '.root':
            return self.parse_root_file(file_path)
        elif extension in ['.hdf5', '.h5']:
            return self.parse_hdf5_file(file_path)
        elif extension == '.csv':
            return self.parse_csv_file(file_path)
        else:
            logger.error(f"Unsupported file format: {extension}")
            return {}

    def extract_dose_distribution(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Extract 3D dose distribution from parsed data

        Args:
            data: Parsed simulation data

        Returns:
            np.ndarray: 3D dose distribution array
        """
        if 'energy_deposition' not in data:
            logger.error("Energy deposition data not found")
            return np.array([])

        try:
            # This is a simplified example - actual implementation depends on data structure
            edep = data['energy_deposition']

            if 'x_position' in data and 'y_position' in data and 'z_position' in data:
                # Create 3D histogram
                x = data['x_position']
                y = data['y_position']
                z = data['z_position']

                # Define binning (adjust as needed)
                bins = [50, 50, 50]

                dose_dist, edges = np.histogramdd(
                    np.column_stack([x, y, z]),
                    bins=bins,
                    weights=edep
                )

                return dose_dist
            else:
                return edep

        except Exception as e:
            logger.error(f"Error extracting dose distribution: {e}")
            return np.array([])

    def compute_statistics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute statistical summary of simulation data

        Args:
            data: Parsed simulation data

        Returns:
            dict: Statistical measures
        """
        stats = {}

        try:
            if 'energy_deposition' in data:
                edep = data['energy_deposition']
                stats['total_energy'] = np.sum(edep)
                stats['mean_energy'] = np.mean(edep)
                stats['std_energy'] = np.std(edep)
                stats['max_energy'] = np.max(edep)
                stats['min_energy'] = np.min(edep)

            if 'particle_energy' in data:
                particle_e = data['particle_energy']
                stats['mean_particle_energy'] = np.mean(particle_e)
                stats['num_particles'] = len(particle_e)

            logger.info("Computed simulation statistics")
            return stats

        except Exception as e:
            logger.error(f"Error computing statistics: {e}")
            return {}

    def save_processed_data(self,
                            data: Dict[str, Any],
                            output_path: str,
                            format: str = 'hdf5'):
        """
        Save processed data to file

        Args:
            data: Processed data dictionary
            output_path: Output file path
            format: Output format ('hdf5', 'npz')
        """
        try:
            if format == 'hdf5':
                with h5py.File(output_path, 'w') as f:
                    for key, value in data.items():
                        if isinstance(value, (np.ndarray, list, float, int)):
                            f.create_dataset(key, data=value)
                        elif isinstance(value, dict):
                            group = f.create_group(key)
                            for subkey, subvalue in value.items():
                                if isinstance(subvalue, (np.ndarray, list, float, int)):
                                    group.create_dataset(subkey, data=subvalue)

            elif format == 'npz':
                np.savez(output_path, **data)

            logger.info(f"Saved processed data to: {output_path}")

        except Exception as e:
            logger.error(f"Error saving processed data: {e}")