"""
Data Processor
Processes and prepares collected data for training
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Processes and normalizes simulation data for ML training
    """

    def __init__(self, normalization_method: str = 'standard'):
        """
        Initialize data processor

        Args:
            normalization_method: Normalization method ('standard', 'minmax', 'none')
        """
        self.normalization_method = normalization_method.lower()
        self.scalers = {}
        self.is_fitted = False

    def fit(self, data: Dict[str, List[np.ndarray]]):
        """
        Fit normalization parameters on training data

        Args:
            data: Dictionary containing training data
        """
        logger.info("Fitting normalization parameters")

        # Fit scaler for Unity observations
        if 'unity_observations' in data:
            unity_obs = np.array(data['unity_observations'])

            if self.normalization_method == 'standard':
                self.scalers['unity'] = StandardScaler()
            elif self.normalization_method == 'minmax':
                self.scalers['unity'] = MinMaxScaler()

            if self.normalization_method != 'none':
                # Reshape if needed
                original_shape = unity_obs.shape
                unity_obs_flat = unity_obs.reshape(len(unity_obs), -1)
                self.scalers['unity'].fit(unity_obs_flat)
                logger.info(f"Fitted Unity observation scaler with shape {original_shape}")

        # Fit scaler for Geant4 results
        if 'geant4_results' in data:
            # Extract relevant features from Geant4 results
            geant4_features = self._extract_geant4_features(data['geant4_results'])

            if len(geant4_features) > 0:
                if self.normalization_method == 'standard':
                    self.scalers['geant4'] = StandardScaler()
                elif self.normalization_method == 'minmax':
                    self.scalers['geant4'] = MinMaxScaler()

                if self.normalization_method != 'none':
                    self.scalers['geant4'].fit(geant4_features)
                    logger.info(f"Fitted Geant4 result scaler with {len(geant4_features)} samples")

        self.is_fitted = True

    def transform(self, data: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Transform data using fitted normalization

        Args:
            data: Dictionary containing data to transform

        Returns:
            dict: Transformed and normalized data
        """
        if not self.is_fitted:
            raise RuntimeError("Processor not fitted. Call fit() first.")

        transformed_data = {}

        # Transform Unity observations
        if 'unity_observations' in data and 'unity' in self.scalers:
            unity_obs = np.array(data['unity_observations'])
            original_shape = unity_obs.shape
            unity_obs_flat = unity_obs.reshape(len(unity_obs), -1)

            transformed_unity = self.scalers['unity'].transform(unity_obs_flat)
            transformed_data['unity_observations'] = transformed_unity.reshape(original_shape)

        # Transform Geant4 results
        if 'geant4_results' in data and 'geant4' in self.scalers:
            geant4_features = self._extract_geant4_features(data['geant4_results'])

            if len(geant4_features) > 0:
                transformed_data['geant4_results'] = self.scalers['geant4'].transform(geant4_features)

        return transformed_data

    def fit_transform(self, data: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Fit and transform data in one step

        Args:
            data: Dictionary containing data

        Returns:
            dict: Transformed data
        """
        self.fit(data)
        return self.transform(data)

    def _extract_geant4_features(self, geant4_results: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract numerical features from Geant4 results

        Args:
            geant4_results: List of Geant4 result dictionaries

        Returns:
            np.ndarray: Extracted features
        """
        features = []

        for result in geant4_results:
            sample_features = []

            # Extract common features (customize based on your data structure)
            if 'total_energy' in result:
                sample_features.append(result['total_energy'])

            if 'mean_energy' in result:
                sample_features.append(result['mean_energy'])

            if 'std_energy' in result:
                sample_features.append(result['std_energy'])

            if 'max_energy' in result:
                sample_features.append(result['max_energy'])

            # If dose distribution exists, flatten and use as features
            if 'dose_distribution' in result:
                dose = result['dose_distribution']
                if isinstance(dose, np.ndarray):
                    sample_features.extend(dose.flatten()[:100])  # Limit to first 100 values

            features.append(sample_features)

        return np.array(features) if features else np.array([])

    def create_training_dataset(self,
                               data: Dict[str, List],
                               train_split: float = 0.8,
                               shuffle: bool = True,
                               random_seed: int = 42) -> Tuple[Dict, Dict]:
        """
        Create train/validation split

        Args:
            data: Full dataset
            train_split: Fraction of data for training
            shuffle: Whether to shuffle before splitting
            random_seed: Random seed for reproducibility

        Returns:
            tuple: (train_data, val_data) dictionaries
        """
        n_samples = len(data['unity_observations'])

        # Validate minimum samples
        if n_samples < 2:
            logger.warning(f"Too few samples ({n_samples}) for train/val split. Returning all as training data.")
            return data, {key: [] for key in data.keys()}

        indices = np.arange(n_samples)

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        split_idx = max(1, int(n_samples * train_split))  # Ensure at least 1 sample in train
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_data = {
            key: [value[i] for i in train_indices]
            for key, value in data.items()
        }

        val_data = {
            key: [value[i] for i in val_indices]
            for key, value in data.items()
        }

        logger.info(f"Created training dataset: {len(train_indices)} train, {len(val_indices)} val")

        return train_data, val_data

    def augment_data(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply data augmentation techniques

        Args:
            data: Data to augment

        Returns:
            dict: Augmented data
        """
        augmented_data = {}

        # Add noise augmentation
        if 'unity_observations' in data:
            obs = data['unity_observations']
            noise = np.random.normal(0, 0.01, obs.shape)
            augmented_data['unity_observations'] = obs + noise

        # Copy other data as-is
        for key in data:
            if key not in augmented_data:
                augmented_data[key] = data[key]

        return augmented_data

    def save_processor(self, file_path: str):
        """
        Save fitted processor to file

        Args:
            file_path: Path to save processor
        """
        if not self.is_fitted:
            logger.warning("Processor not fitted, saving empty processor")

        processor_data = {
            'normalization_method': self.normalization_method,
            'scalers': self.scalers,
            'is_fitted': self.is_fitted
        }

        with open(file_path, 'wb') as f:
            pickle.dump(processor_data, f)

        logger.info(f"Saved processor to: {file_path}")

    @classmethod
    def load_processor(cls, file_path: str) -> 'DataProcessor':
        """
        Load processor from file

        Args:
            file_path: Path to processor file

        Returns:
            DataProcessor: Loaded processor
        """
        with open(file_path, 'rb') as f:
            processor_data = pickle.load(f)

        processor = cls(normalization_method=processor_data['normalization_method'])
        processor.scalers = processor_data['scalers']
        processor.is_fitted = processor_data['is_fitted']

        logger.info(f"Loaded processor from: {file_path}")
        return processor

    def compute_statistics(self, data: Dict[str, List]) -> Dict[str, Any]:
        """
        Compute statistics on dataset

        Args:
            data: Dataset to analyze

        Returns:
            dict: Statistical information
        """
        stats = {}

        if 'unity_observations' in data:
            unity_obs = np.array(data['unity_observations'])
            stats['unity'] = {
                'shape': unity_obs.shape,
                'mean': float(np.mean(unity_obs)),
                'std': float(np.std(unity_obs)),
                'min': float(np.min(unity_obs)),
                'max': float(np.max(unity_obs))
            }

        if 'geant4_results' in data:
            geant4_features = self._extract_geant4_features(data['geant4_results'])
            if len(geant4_features) > 0:
                stats['geant4'] = {
                    'shape': geant4_features.shape,
                    'mean': float(np.mean(geant4_features)),
                    'std': float(np.std(geant4_features)),
                    'min': float(np.min(geant4_features)),
                    'max': float(np.max(geant4_features))
                }

        return stats