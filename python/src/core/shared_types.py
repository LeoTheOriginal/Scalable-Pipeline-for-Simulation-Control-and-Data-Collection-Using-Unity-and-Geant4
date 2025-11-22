import numpy as np
from dataclasses import dataclass

# KONFIGURACJA PAMIĘCI WSPÓŁDZIELONEJ
SHM_NAME = "geant4_shm"
MAX_TRAJECTORIES = 2000
MAX_STEPS = 500
FLOAT_SIZE = 8

POINT_DTYPE = np.dtype([
    ('x', np.float64),
    ('y', np.float64),
    ('z', np.float64),
    ('energy', np.float64)
])

POINT_SIZE = POINT_DTYPE.itemsize
TOTAL_BUFFER_SIZE = MAX_TRAJECTORIES * MAX_STEPS * POINT_SIZE

@dataclass
class SimulationConfig:
    batch_size: int = 100
    energy_mev: float = 10.0