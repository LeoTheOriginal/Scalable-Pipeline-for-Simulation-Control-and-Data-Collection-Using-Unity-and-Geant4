"""
Configuration Manager
Centralized configuration management for the project
"""

import yaml
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages configuration for the entire pipeline
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager

        Args:
            config_file: Path to configuration file (YAML or JSON)
        """
        self.config_file = config_file
        self.config = self._load_config() if config_file else self._get_default_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file

        Returns:
            dict: Configuration dictionary
        """
        config_path = Path(self.config_file)

        if not config_path.exists():
            logger.warning(f"Config file not found: {self.config_file}")
            return self._get_default_config()

        try:
            with open(config_path, 'r') as f:
                if config_path.suffix in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif config_path.suffix == '.json':
                    config = json.load(f)
                else:
                    logger.error(f"Unsupported config format: {config_path.suffix}")
                    return self._get_default_config()

            logger.info(f"Loaded configuration from: {self.config_file}")
            return config

        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration

        Returns:
            dict: Default configuration
        """
        return {
            'project': {
                'name': 'Unity-Geant4-Pipeline',
                'version': '1.0.0',
                'description': 'Scalable pipeline for simulation control and data collection'
            },

            'unity': {
                'environment_path': None,  # None for Unity Editor
                'base_port': 5005,
                'num_parallel_envs': 1,
                'time_scale': 20.0,
                'timeout_wait': 60
            },

            'geant4': {
                'executable_path': './geant4_sim',
                'working_directory': './geant4_runs',
                'output_directory': './geant4_output',
                'default_num_events': 1000,
                'timeout': 300
            },

            'simulation': {
                'particle_types': ['gamma', 'electron', 'proton'],
                'energy_range': [1.0, 100.0],  # MeV
                'energy_unit': 'MeV',
                'phantom_materials': ['Water', 'Bone', 'Tissue'],
                'phantom_size': [10.0, 10.0, 10.0],  # cm
            },

            'data_collection': {
                'output_directory': './collected_data',
                'max_samples_per_file': 1000,
                'file_format': 'hdf5'
            },

            'data_processing': {
                'normalization_method': 'standard',
                'train_split': 0.8,
                'validation_split': 0.1,
                'test_split': 0.1,
                'shuffle': True,
                'random_seed': 42
            },

            'training': {
                'trainer_type': 'ppo',
                'max_steps': 500000,
                'batch_size': 1024,
                'buffer_size': 10240,
                'learning_rate': 3e-4,
                'num_epochs': 3,
                'hidden_units': 128,
                'num_layers': 2,
                'gamma': 0.99,
                'checkpoint_interval': 50000,
                'summary_freq': 10000
            },

            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'pipeline.log'
            },

            'paths': {
                'data_dir': './data',
                'models_dir': './models',
                'results_dir': './results',
                'logs_dir': './logs'
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with '.')

        Args:
            key: Configuration key (e.g., 'unity.base_port')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any):
        """
        Set configuration value by key (supports nested keys with '.')

        Args:
            key: Configuration key (e.g., 'unity.base_port')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config

        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the value
        config[keys[-1]] = value
        logger.debug(f"Set config {key} = {value}")

    def update(self, updates: Dict[str, Any]):
        """
        Update configuration with new values

        Args:
            updates: Dictionary of updates
        """
        self._deep_update(self.config, updates)
        logger.info("Configuration updated")

    def _deep_update(self, base: Dict, updates: Dict):
        """
        Recursively update nested dictionary

        Args:
            base: Base dictionary to update
            updates: Updates to apply
        """
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def save(self, output_path: Optional[str] = None):
        """
        Save configuration to file

        Args:
            output_path: Output file path (uses original path if None)
        """
        save_path = output_path or self.config_file

        if not save_path:
            logger.error("No output path specified")
            return

        save_path = Path(save_path)

        try:
            with open(save_path, 'w') as f:
                if save_path.suffix in ['.yaml', '.yml']:
                    yaml.dump(self.config, f, default_flow_style=False)
                elif save_path.suffix == '.json':
                    json.dump(self.config, f, indent=2)
                else:
                    logger.error(f"Unsupported format: {save_path.suffix}")
                    return

            logger.info(f"Saved configuration to: {save_path}")

        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def create_directories(self):
        """
        Create all directories specified in paths configuration
        """
        paths_config = self.get('paths', {})

        for path_key, path_value in paths_config.items():
            path = Path(path_value)
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")

    def setup_logging(self):
        """
        Setup logging based on configuration
        """
        log_config = self.get('logging', {})

        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
        log_file = log_config.get('file', None)

        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file) if log_file else logging.NullHandler()
            ]
        )

        logger.info("Logging configured")

    def validate(self) -> bool:
        """
        Validate configuration

        Returns:
            bool: True if configuration is valid
        """
        required_sections = ['project', 'unity', 'geant4', 'data_collection', 'training']

        for section in required_sections:
            if section not in self.config:
                logger.error(f"Missing required section: {section}")
                return False

        logger.info("Configuration validated successfully")
        return True

    def print_config(self):
        """Print current configuration in readable format"""
        print("\n" + "="*50)
        print("Current Configuration")
        print("="*50)
        print(yaml.dump(self.config, default_flow_style=False))
        print("="*50 + "\n")

    def export_template(self, output_path: str):
        """
        Export default configuration template

        Args:
            output_path: Path to save template
        """
        template = self._get_default_config()

        output_path = Path(output_path)

        with open(output_path, 'w') as f:
            if output_path.suffix in ['.yaml', '.yml']:
                yaml.dump(template, f, default_flow_style=False)
            elif output_path.suffix == '.json':
                json.dump(template, f, indent=2)

        logger.info(f"Exported configuration template to: {output_path}")