# Unity-Geant4 Pipeline for Reinforcement Learning

Scalable pipeline for simulation control and data collection using Unity and Geant4, designed for training RL agents to emulate physics simulations.

## Project Structure

```
python/
├── unity_interface/          # Unity ML-Agents communication
│   ├── __init__.py
│   ├── environment_manager.py   # Manage Unity environments
│   └── communication.py         # Custom Unity-Python communication
│
├── geant4_interface/         # Geant4 simulation control
│   ├── __init__.py
│   ├── simulation_runner.py    # Run Geant4 simulations
│   └── data_parser.py          # Parse Geant4 output
│
├── data_collection/          # Data management
│   ├── __init__.py
│   ├── data_collector.py       # Collect simulation data
│   └── data_processor.py       # Process and normalize data
│
├── training/                 # RL training
│   ├── __init__.py
│   └── train_agent.py          # Train ML-Agents
│
├── utils/                    # Utilities
│   ├── __init__.py
│   └── config.py               # Configuration management
│
├── test_environment.py       # Environment setup test
└── main_example.py          # Example usage
```

## Installation

### 1. Environment Setup

The project uses a Conda environment with ML-Agents:

```bash
# Activate your environment
conda activate ml-agents

# Verify installation
python test_environment.py
```

### 2. Dependencies

Core dependencies (already installed):
- Python 3.10.12
- NumPy 1.23.5
- ML-Agents 1.1.0
- PyTorch 2.9.1
- H5Py 3.15.1

Additional dependencies for Geant4 data parsing:
```bash
pip install uproot awkward  # For ROOT file parsing
pip install scikit-learn    # For data processing
pip install pyyaml          # For configuration
```

## Quick Start

### 1. Configuration

Create a configuration file or use defaults:

```python
from utils.config import ConfigManager

# Create config manager
config = ConfigManager()

# Export template for customization
config.export_template('my_config.yaml')

# Load custom config
config = ConfigManager('my_config.yaml')
```

### 2. Unity Environment

```python
from unity_interface.environment_manager import UnityEnvironmentManager

# Initialize Unity environment
with UnityEnvironmentManager(environment_path=None) as env:
    # Reset environment
    state = env.reset()
    
    # Get observation space
    obs_space = env.get_observation_space()
    print(f"Observation shapes: {obs_space['observation_shapes']}")
    print(f"Action size: {obs_space['action_size']}")
    
    # Step through environment
    actions = np.zeros(obs_space['action_size'])
    result = env.step(actions)
```

### 3. Geant4 Simulation

```python
from geant4_interface.simulation_runner import Geant4SimulationRunner
from geant4_interface.data_parser import Geant4DataParser

# Initialize simulation runner
runner = Geant4SimulationRunner(
    geant4_executable="path/to/geant4_executable"
)

# Define simulation parameters
parameters = {
    'particle_type': 'gamma',
    'particle_energy': 10.0,
    'energy_unit': 'MeV',
    'particle_position': [0.0, 0.0, -5.0],
    'particle_direction': [0.0, 0.0, 1.0],
    'phantom_material': 'Water',
    'phantom_size': [10.0, 10.0, 10.0],
    'num_events': 1000
}

# Run simulation
result = runner.run_simulation(parameters)

# Parse results
if result['success']:
    parser = Geant4DataParser(output_format='root')
    data = parser.parse_file(result['output_file'])
    stats = parser.compute_statistics(data)
```

### 4. Data Collection

```python
from data_collection.data_collector import DataCollector

# Initialize collector
collector = DataCollector(
    output_directory="./collected_data",
    max_samples_per_file=1000
)

# Collect data pairs
collector.collect_simulation_pair(
    unity_observation=unity_obs,
    geant4_result=geant4_data,
    parameters=sim_parameters
)

# Finalize and save
collector.finalize()
```

### 5. Data Processing

```python
from data_collection.data_processor import DataProcessor

# Load collected data
dataset = collector.load_dataset(file_index=0)

# Initialize processor
processor = DataProcessor(normalization_method='standard')

# Create train/validation split
train_data, val_data = processor.create_training_dataset(
    dataset,
    train_split=0.8,
    shuffle=True
)

# Normalize data
processor.fit(train_data)
train_normalized = processor.transform(train_data)
val_normalized = processor.transform(val_data)
```

### 6. Training

```python
from training.train_agent import AgentTrainer

# Initialize trainer
trainer = AgentTrainer(run_id="my_training_run")

# Create configuration
trainer.create_trainer_config(behavior_name="SimulationAgent")

# Start training (via command line)
# mlagents-learn results/my_training_run/trainer_config.yaml --run-id=my_training_run
```

## Complete Pipeline Example

```python
from utils.config import ConfigManager
from data_collection.data_collector import DataCollector
from data_collection.data_processor import DataProcessor
from training.train_agent import AgentTrainer

# 1. Load configuration
config = ConfigManager()
config.create_directories()

# 2. Collect data
collector = DataCollector(
    output_directory=config.get('data_collection.output_directory')
)

# ... data collection loop ...

collector.finalize()

# 3. Process data
dataset = collector.load_dataset(0)
processor = DataProcessor(
    normalization_method=config.get('data_processing.normalization_method')
)

train_data, val_data = processor.create_training_dataset(dataset)

# 4. Setup training
trainer = AgentTrainer()
trainer.create_trainer_config()

# 5. Train (via CLI)
print(f"Run: mlagents-learn {trainer.results_dir}/trainer_config.yaml --run-id={trainer.run_id}")
```

## Running Examples

Run the example script to test all components:

```bash
python main_example.py
```

## Configuration

The configuration file supports the following sections:

- **project**: Project metadata
- **unity**: Unity environment settings
- **geant4**: Geant4 simulation settings
- **simulation**: Default simulation parameters
- **data_collection**: Data collection settings
- **data_processing**: Data processing parameters
- **training**: ML-Agents training configuration
- **logging**: Logging configuration
- **paths**: Directory paths

## Data Format

Collected data is stored in HDF5 format with the following structure:

```
data_0000.hdf5
├── sample_000000/
│   ├── unity_observation     # Unity observation array
│   ├── geant4_result/        # Geant4 simulation results
│   │   ├── energy_deposition
│   │   ├── dose_distribution
│   │   └── ...
│   └── parameters/           # Simulation parameters
│       ├── particle_type
│       ├── particle_energy
│       └── ...
├── sample_000001/
└── ...
```

## Training with ML-Agents

### Create Training Configuration

```bash
# Generate configuration
python -c "from training.train_agent import AgentTrainer; t = AgentTrainer(); t.create_trainer_config()"
```

### Start Training

```bash
# Train with Unity Editor (no build required)
mlagents-learn results/<run_id>/trainer_config.yaml --run-id=<run_id>

# Train with Unity build
mlagents-learn results/<run_id>/trainer_config.yaml --run-id=<run_id> --env=path/to/unity_build
```

### Monitor Training

```bash
# View tensorboard
tensorboard --logdir results/<run_id>/summaries
```

## Advanced Usage

### Parallel Simulations

```python
# Run batch simulations
parameter_sets = [
    {'particle_energy': 10.0, 'num_events': 1000},
    {'particle_energy': 20.0, 'num_events': 1000},
    {'particle_energy': 30.0, 'num_events': 1000},
]

results = runner.run_batch_simulations(parameter_sets)
```

### Custom Communication

```python
from unity_interface.communication import UnityCommunication

# Establish custom communication
with UnityCommunication(host='localhost', port=9000) as comm:
    # Send parameters to Unity
    comm.send_simulation_parameters({'energy': 10.0})
    
    # Request data
    data = comm.request_simulation_data()
```

### Curriculum Learning

```python
# Create curriculum stages
stages = [
    {'threshold': 0.0, 'parameters': {'difficulty': 1}},
    {'threshold': 0.5, 'parameters': {'difficulty': 2}},
    {'threshold': 0.7, 'parameters': {'difficulty': 3}},
]

curriculum_path = trainer.create_curriculum(stages)
```

## Troubleshooting

### Unity Connection Issues

- Ensure Unity environment is running
- Check port availability (default: 5005)
- Verify worker_id doesn't conflict

### Geant4 Simulation Errors

- Check executable path is correct
- Verify Geant4 data files are installed
- Review simulation logs in working directory

### ML-Agents Training Issues

- Check Unity environment behavior matches config
- Verify observation/action spaces
- Review tensorboard logs

## Contributing

This is a thesis project. For questions, contact the project author.

## License

Academic use only - Part of Engineering Thesis Project

## References

- Unity ML-Agents: https://github.com/Unity-Technologies/ml-agents
- Geant4: https://geant4.web.cern.ch/
- PyTorch: https://pytorch.org/