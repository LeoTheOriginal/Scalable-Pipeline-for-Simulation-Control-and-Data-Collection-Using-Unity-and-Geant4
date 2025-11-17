"""
Main Pipeline Example
Demonstrates how to use all components together
"""

import logging
from pathlib import Path

# Import project modules
from utils.config import ConfigManager
from unity_interface.environment_manager import UnityEnvironmentManager
from geant4_interface.simulation_runner import Geant4SimulationRunner
from geant4_interface.data_parser import Geant4DataParser
from data_collection.data_collector import DataCollector
from data_collection.data_processor import DataProcessor
from training.train_agent import AgentTrainer


def setup_logging():
    """Setup logging for the pipeline"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def example_unity_only():
    """
    Example: Test Unity environment connection
    """
    print("\n" + "="*60)
    print("Example 1: Unity Environment Test")
    print("="*60)

    # Create Unity environment manager
    # Set environment_path=None to use Unity Editor
    with UnityEnvironmentManager(environment_path=None, worker_id=0) as env:
        # Reset environment
        initial_state = env.reset()
        print(f"Environment initialized")
        print(f"Number of agents: {len(initial_state['agents'])}")

        # Get observation space info
        obs_space = env.get_observation_space()
        print(f"Observation shapes: {obs_space['observation_shapes']}")
        print(f"Action size: {obs_space['action_size']}")


def example_geant4_only():
    """
    Example: Run Geant4 simulation
    """
    print("\n" + "="*60)
    print("Example 2: Geant4 Simulation Test")
    print("="*60)

    # Create simulation runner
    runner = Geant4SimulationRunner(
        geant4_executable="./geant4_executable",  # Update with your path
        working_directory="./test_geant4_runs",
        output_directory="./test_geant4_output"
    )

    # Define simulation parameters
    parameters = {
        'particle_type': 'gamma',
        'particle_energy': 10.0,  # MeV
        'energy_unit': 'MeV',
        'particle_position': [0.0, 0.0, -5.0],
        'particle_direction': [0.0, 0.0, 1.0],
        'phantom_material': 'Water',
        'phantom_size': [10.0, 10.0, 10.0],
        'num_events': 1000
    }

    # Run simulation
    result = runner.run_simulation(parameters, timeout=300)

    if result['success']:
        print(f"Simulation {result['simulation_id']} completed successfully")
        print(f"Output file: {result['output_file']}")

        # Parse results
        parser = Geant4DataParser(output_format='root')
        data = parser.parse_file(result['output_file'])

        # Compute statistics
        stats = parser.compute_statistics(data)
        print(f"Statistics: {stats}")
    else:
        print(f"Simulation failed: {result.get('error', 'Unknown error')}")


def example_data_collection():
    """
    Example: Collect data from simulations
    """
    print("\n" + "="*60)
    print("Example 3: Data Collection")
    print("="*60)

    # Create data collector
    collector = DataCollector(
        output_directory="./example_collected_data",
        max_samples_per_file=100,
        clean_start=True  # Start fresh each time
    )

    # Simulate collecting data
    import numpy as np

    for i in range(10):
        # Mock Unity observation
        unity_obs = np.random.rand(10)

        # Mock Geant4 result
        geant4_result = {
            'total_energy': float(np.random.rand() * 100),
            'mean_energy': float(np.random.rand() * 10),
            'std_energy': float(np.random.rand() * 5),
        }

        # Mock parameters
        parameters = {
            'particle_type': 'gamma',
            'energy': 10.0 + i
        }

        # Collect sample
        collector.collect_simulation_pair(unity_obs, geant4_result, parameters)

    # Get statistics
    stats = collector.get_statistics()
    print(f"Collected {stats['total_samples']} samples")

    # Finalize and save metadata
    collector.finalize()


def example_data_processing():
    """
    Example: Process collected data
    """
    print("\n" + "="*60)
    print("Example 4: Data Processing")
    print("="*60)

    # Create mock dataset
    import numpy as np

    dataset = {
        'unity_observations': [np.random.rand(10) for _ in range(100)],
        'geant4_results': [
            {'total_energy': float(np.random.rand() * 100)}
            for _ in range(100)
        ],
        'parameters': [{'energy': 10.0} for _ in range(100)]
    }

    # Create data processor
    processor = DataProcessor(normalization_method='standard')

    # Fit and transform data
    transformed_data = processor.fit_transform(dataset)

    # Create train/val split
    train_data, val_data = processor.create_training_dataset(
        dataset,
        train_split=0.8,
        shuffle=True
    )

    print(f"Training samples: {len(train_data['unity_observations'])}")
    print(f"Validation samples: {len(val_data['unity_observations'])}")

    # Compute statistics
    stats = processor.compute_statistics(dataset)
    print(f"Dataset statistics: {stats}")


def example_training_setup():
    """
    Example: Setup training configuration
    """
    print("\n" + "="*60)
    print("Example 5: Training Setup")
    print("="*60)

    # Create agent trainer
    trainer = AgentTrainer(run_id="test_run_001")

    # Create trainer configuration
    config = trainer.create_trainer_config(behavior_name="SimulationAgent")

    print(f"Training configuration created")
    print(f"Results directory: {trainer.results_dir}")

    # Save training metadata
    metadata = {
        'description': 'Test training run',
        'num_environments': 1,
        'total_timesteps': 500000
    }
    trainer.save_training_metadata(metadata)

    print("\nTo start training, run:")
    print(f"mlagents-learn {trainer.results_dir}/trainer_config.yaml --run-id={trainer.run_id}")


def example_config_management():
    """
    Example: Configuration management
    """
    print("\n" + "="*60)
    print("Example 6: Configuration Management")
    print("="*60)

    # Create config manager
    config = ConfigManager()

    # Access configuration values
    unity_port = config.get('unity.base_port')
    print(f"Unity base port: {unity_port}")

    # Update configuration
    config.set('unity.num_parallel_envs', 4)
    print(f"Updated parallel envs: {config.get('unity.num_parallel_envs')}")

    # Create directories
    config.create_directories()

    # Save configuration
    config.save('./example_config.yaml')
    print("Configuration saved to example_config.yaml")

    # Export template
    config.export_template('./config_template.yaml')
    print("Template exported to config_template.yaml")


def example_full_pipeline():
    """
    Example: Complete pipeline workflow
    """
    print("\n" + "="*60)
    print("Example 7: Full Pipeline Workflow")
    print("="*60)

    # 1. Load configuration
    config = ConfigManager()
    config.setup_logging()

    print("Step 1: Configuration loaded")

    # 2. Initialize components
    # Note: These would need actual Unity and Geant4 setups
    print("Step 2: Components initialized (mock)")

    # 3. Data collection loop (simplified)
    print("Step 3: Data collection")
    collector = DataCollector(
        output_directory=config.get('data_collection.output_directory'),
        clean_start=True  # Start fresh each time
    )

    # Simulate data collection
    num_samples = 10
    for i in range(num_samples):
        import numpy as np
        unity_obs = np.random.rand(10)
        geant4_result = {'energy': float(np.random.rand() * 100)}
        parameters = {'particle_energy': 10.0 + i}

        collector.collect_simulation_pair(unity_obs, geant4_result, parameters)

    collector.finalize()
    print(f"Collected {num_samples} samples")

    # 4. Data processing
    print("Step 4: Data processing")
    dataset = collector.load_dataset(file_index=0)

    if dataset:
        processor = DataProcessor(
            normalization_method=config.get('data_processing.normalization_method')
        )
        train_data, val_data = processor.create_training_dataset(
            dataset,
            train_split=config.get('data_processing.train_split')
        )
        print(f"Training dataset prepared: {len(train_data['unity_observations'])} samples")

    # 5. Training setup
    print("Step 5: Training setup")
    trainer = AgentTrainer()
    trainer.create_trainer_config()

    print("\nPipeline workflow completed!")
    print(f"Results saved to: {trainer.results_dir}")


def main():
    """
    Main function - run all examples
    """
    setup_logging()

    print("\n" + "="*60)
    print("Unity-Geant4 Pipeline Examples")
    print("="*60)

    # Run examples (comment out ones you don't want to run)

    # example_unity_only()  # Requires Unity environment
    # example_geant4_only()  # Requires Geant4 setup
    example_data_collection()
    example_data_processing()
    example_training_setup()
    example_config_management()
    example_full_pipeline()

    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()