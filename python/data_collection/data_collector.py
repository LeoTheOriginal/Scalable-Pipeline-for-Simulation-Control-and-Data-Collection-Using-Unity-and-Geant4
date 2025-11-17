"""
Data Collector
Orchestrates data collection from Unity and Geant4 simulations
"""

import numpy as np
import h5py
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Collects and organizes data from simulations for training
    """

    def __init__(self,
                 output_directory: str = "./collected_data",
                 max_samples_per_file: int = 1000,
                 clean_start: bool = False):
        """
        Initialize data collector

        Args:
            output_directory: Directory to store collected data
            max_samples_per_file: Maximum samples per HDF5 file
            clean_start: If True, remove existing data files and start fresh
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        # Clean existing data if requested
        if clean_start:
            self._clean_existing_data()

        self.max_samples_per_file = max_samples_per_file
        self.current_file_index = 0
        self.current_sample_count = 0
        self.samples_in_current_file = 0  # Track samples in current file

        self.metadata = {
            'collection_start': datetime.now().isoformat(),
            'total_samples': 0,
            'files': []
        }

    def _clean_existing_data(self):
        """
        Remove all existing HDF5 files and metadata in output directory
        """
        logger.info(f"Cleaning existing data in {self.output_directory}")

        # Remove HDF5 files
        for hdf5_file in self.output_directory.glob("*.hdf5"):
            hdf5_file.unlink()
            logger.debug(f"Removed {hdf5_file}")

        # Remove metadata file
        metadata_file = self.output_directory / "metadata.json"
        if metadata_file.exists():
            metadata_file.unlink()
            logger.debug(f"Removed {metadata_file}")

    def collect_simulation_pair(self,
                               unity_observation: np.ndarray,
                               geant4_result: Dict[str, Any],
                               parameters: Dict[str, Any]) -> bool:
        """
        Collect paired data from Unity and Geant4

        Args:
            unity_observation: Observation from Unity environment
            geant4_result: Results from Geant4 simulation
            parameters: Simulation parameters used

        Returns:
            bool: True if successfully collected
        """
        try:
            # Prepare data sample
            sample = {
                'unity_observation': unity_observation,
                'geant4_output': geant4_result,
                'parameters': parameters,
                'timestamp': datetime.now().isoformat()
            }

            # Save to current file
            self._save_sample(sample)

            self.current_sample_count += 1
            self.samples_in_current_file += 1
            self.metadata['total_samples'] += 1

            # Check if we need a new file
            if self.samples_in_current_file >= self.max_samples_per_file:
                self._create_new_file()

            return True

        except Exception as e:
            logger.error(f"Error collecting simulation pair: {e}")
            return False

    def collect_batch(self,
                     unity_observations: List[np.ndarray],
                     geant4_results: List[Dict[str, Any]],
                     parameters_list: List[Dict[str, Any]]) -> int:
        """
        Collect batch of simulation pairs

        Args:
            unity_observations: List of Unity observations
            geant4_results: List of Geant4 results
            parameters_list: List of parameter sets

        Returns:
            int: Number of successfully collected samples
        """
        if not (len(unity_observations) == len(geant4_results) == len(parameters_list)):
            logger.error("Input lists must have the same length")
            return 0

        success_count = 0

        for unity_obs, geant4_res, params in zip(
            unity_observations, geant4_results, parameters_list
        ):
            if self.collect_simulation_pair(unity_obs, geant4_res, params):
                success_count += 1

        logger.info(f"Collected {success_count}/{len(unity_observations)} samples")
        return success_count

    def _save_sample(self, sample: Dict[str, Any]):
        """
        Save sample to current HDF5 file

        Args:
            sample: Data sample to save
        """
        file_path = self._get_current_file_path()

        # Create or open HDF5 file
        with h5py.File(file_path, 'a') as f:
            # If file already exists, count existing samples and continue from there
            if self.samples_in_current_file == 0 and len(f.keys()) > 0:
                # File exists with samples, find the highest sample number
                existing_samples = [key for key in f.keys() if key.startswith('sample_')]
                if existing_samples:
                    # Extract numbers and find max
                    sample_numbers = [int(s.split('_')[1]) for s in existing_samples]
                    self.samples_in_current_file = max(sample_numbers) + 1
                    logger.info(f"Resuming file {file_path} from sample {self.samples_in_current_file}")

            # Create group for this sample
            sample_id = f"sample_{self.samples_in_current_file:06d}"

            # Skip if group already exists (safety check)
            if sample_id in f:
                logger.warning(f"Sample {sample_id} already exists in file, skipping")
                return

            group = f.create_group(sample_id)

            # Save Unity observation
            if isinstance(sample['unity_observation'], np.ndarray):
                group.create_dataset('unity_observation',
                                   data=sample['unity_observation'])

            # ===== MODIFIED: Save Geant4 results (only numeric data) =====
            geant4_output = sample['geant4_output']
            geant4_group = group.create_group('geant4_result')

            # Save scalar values that are HDF5-compatible
            if 'total_energy_deposit' in geant4_output:
                geant4_group.create_dataset('total_energy_deposit',
                                          data=geant4_output['total_energy_deposit'])

            if 'energy_deposit_std' in geant4_output:
                geant4_group.create_dataset('energy_deposit_std',
                                          data=geant4_output['energy_deposit_std'])

            if 'num_events' in geant4_output:
                geant4_group.create_dataset('num_events',
                                          data=geant4_output['num_events'])

            # Save success flag
            if 'success' in geant4_output:
                geant4_group.attrs['success'] = str(geant4_output['success'])

            # Save output directory path
            if 'output_directory' in geant4_output:
                geant4_group.attrs['output_directory'] = str(geant4_output['output_directory'])

            # Save full result as JSON string for detailed analysis later
            try:
                # Create simplified version without large arrays
                simplified_result = {
                    'total_energy_deposit': float(geant4_output.get('total_energy_deposit', 0)),
                    'num_events': int(geant4_output.get('num_events', 0)),
                    'success': bool(geant4_output.get('success', False)),
                    'parameters': geant4_output.get('parameters', {}),
                }

                # Add event summary without full step data
                if 'events' in geant4_output and geant4_output['events']:
                    simplified_result['event_summaries'] = [
                        {
                            'event_id': e.get('event_id', 0),
                            'total_energy_deposit': e.get('total_energy_deposit', 0),
                            'num_steps': e.get('num_steps', 0),
                            'success': e.get('success', True)
                        }
                        for e in geant4_output['events']
                    ]

                result_json = json.dumps(simplified_result)
                geant4_group.create_dataset('result_json',
                                          data=result_json,
                                          dtype=h5py.special_dtype(vlen=str))
            except Exception as e:
                logger.warning(f"Could not save JSON result: {e}")
            # ===== END MODIFICATION =====

            # Save parameters as attributes
            params_group = group.create_group('parameters')
            for key, value in sample['parameters'].items():
                if isinstance(value, str):
                    params_group.attrs[key] = value
                elif isinstance(value, (int, float, bool)):
                    params_group.attrs[key] = value
                elif isinstance(value, (list, tuple, np.ndarray)):
                    # Convert to numpy array for HDF5 compatibility
                    try:
                        params_group.create_dataset(key, data=np.array(value))
                    except Exception as e:
                        logger.warning(f"Could not save parameter {key}: {e}")
                        # Try saving as JSON string
                        try:
                            params_group.attrs[key + '_json'] = json.dumps(value)
                        except:
                            pass

            # Save timestamp
            group.attrs['timestamp'] = sample['timestamp']

    def _get_current_file_path(self) -> Path:
        """Get path to current data file"""
        return self.output_directory / f"data_{self.current_file_index:04d}.hdf5"

    def _create_new_file(self):
        """Create new data file for next batch of samples"""
        self.current_file_index += 1
        self.samples_in_current_file = 0  # Reset sample counter for new file

        file_path = self._get_current_file_path()
        self.metadata['files'].append(str(file_path))

        logger.info(f"Creating new data file: {file_path}")

    def save_metadata(self):
        """Save collection metadata to JSON file"""
        metadata_path = self.output_directory / "metadata.json"

        self.metadata['collection_end'] = datetime.now().isoformat()

        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"Saved metadata to: {metadata_path}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get collection statistics

        Returns:
            dict: Statistics about collected data
        """
        return {
            'total_samples': self.metadata['total_samples'],
            'num_files': self.current_file_index + 1,
            'current_file_samples': self.current_sample_count,
            'output_directory': str(self.output_directory)
        }

    def load_dataset(self, file_index: int = 0) -> Optional[Dict[str, List]]:
        """
        Load dataset from specified file

        Args:
            file_index: Index of file to load

        Returns:
            dict: Loaded dataset with lists of samples
        """
        file_path = self.output_directory / f"data_{file_index:04d}.hdf5"

        if not file_path.exists():
            logger.error(f"Data file not found: {file_path}")
            return None

        try:
            dataset = {
                'unity_observations': [],
                'geant4_results': [],
                'parameters': []
            }

            with h5py.File(file_path, 'r') as f:
                for sample_key in sorted(f.keys()):
                    sample_group = f[sample_key]

                    # Load Unity observation
                    if 'unity_observation' in sample_group:
                        dataset['unity_observations'].append(
                            sample_group['unity_observation'][:]
                        )

                    # Load Geant4 results
                    if 'geant4_result' in sample_group:
                        geant4_data = {}
                        geant4_group = sample_group['geant4_result']

                        # Load numeric datasets
                        for key in geant4_group.keys():
                            dataset_obj = geant4_group[key]

                            if key == 'result_json':
                                # Parse JSON string
                                try:
                                    json_str = dataset_obj[()]
                                    if isinstance(json_str, bytes):
                                        json_str = json_str.decode('utf-8')
                                    geant4_data['full_result'] = json.loads(json_str)
                                except Exception as e:
                                    logger.warning(f"Could not parse JSON result: {e}")
                            else:
                                # Regular numeric data
                                if dataset_obj.shape == ():  # Scalar
                                    geant4_data[key] = float(dataset_obj[()])
                                else:  # Array
                                    geant4_data[key] = dataset_obj[:]

                        # Load attributes
                        for key, value in geant4_group.attrs.items():
                            geant4_data[key] = value

                        dataset['geant4_results'].append(geant4_data)

                    # Load parameters
                    if 'parameters' in sample_group:
                        params = {}
                        params_group = sample_group['parameters']

                        # Load attributes
                        for key, value in params_group.attrs.items():
                            if key.endswith('_json'):
                                # Parse JSON attribute
                                try:
                                    params[key[:-5]] = json.loads(value)
                                except:
                                    params[key] = value
                            else:
                                params[key] = value

                        # Load datasets
                        for key in params_group.keys():
                            dataset_obj = params_group[key]
                            if dataset_obj.shape == ():  # Scalar
                                params[key] = float(dataset_obj[()])
                            else:  # Array
                                params[key] = dataset_obj[:].tolist()

                        dataset['parameters'].append(params)

            logger.info(f"Loaded {len(dataset['unity_observations'])} samples from {file_path}")
            return dataset

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            import traceback
            traceback.print_exc()
            return None

    def finalize(self):
        """Finalize data collection and save metadata"""
        self.save_metadata()
        logger.info(f"Data collection finalized. Total samples: {self.metadata['total_samples']}")