"""
Main Pipeline - Unity ML-Agents + Geant4 Integration
Collect training data: Unity observations → Geant4 ground truth
"""

import sys
from pathlib import Path
import numpy as np
import time
from typing import Dict, List
import logging

from unity_interface.environment_manager import UnityEnvironmentManager
from geant4_interface.simulation_runner import Geant4SimulationRunner
from data_collection.data_collector import DataCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLAgentsGeant4Pipeline:
    """
    Main pipeline integrating Unity ML-Agents with Geant4
    """

    def __init__(self,
                 geant4_executable: str,
                 output_directory: str = "./training_data"):
        """
        Initialize pipeline

        Args:
            geant4_executable: Path to Geant4 executable
            output_directory: Directory for training data
        """
        # Unity environment
        self.unity_env = UnityEnvironmentManager(
            environment_path=None,
            worker_id=0,
            base_port=5004
        )

        # Geant4 runner
        self.geant4_runner = Geant4SimulationRunner(
            geant4_executable=geant4_executable,
            output_directory=output_directory + "/geant4_runs"
        )

        # Data collector
        self.data_collector = DataCollector(
            output_directory=output_directory,
            clean_start=True
        )

        logger.info("Pipeline initialized successfully")

    def collect_training_data(self, num_samples: int = 100):
        """
        Collect training data by running Unity + Geant4 pairs

        Args:
            num_samples: Number of training samples to collect
        """
        logger.info(f"Starting data collection: {num_samples} samples")
        logger.info("="*60)

        # Initialize Unity
        logger.info("Initializing Unity environment...")
        success = self.unity_env.initialize()
        if not success:
            raise RuntimeError("Failed to initialize Unity environment")

        logger.info("✅ Unity connected!")

        # Parameter ranges for data collection
        energy_range = [1.0, 20.0]  # MeV

        collected = 0
        failed = 0

        try:
            for i in range(num_samples):
                logger.info(f"\n{'='*60}")
                logger.info(f"Sample {i+1}/{num_samples}")
                logger.info(f"{'='*60}")

                # 1. Generate random parameters
                particle_energy = np.random.uniform(*energy_range)

                parameters = {
                    'particle_type': 'e-',
                    'particle_energy': particle_energy,
                    'particle_position': [-6, 0, 0],
                    'particle_direction': [1, 0, 0],
                    'num_events': 1
                }

                logger.info(f"Parameters: Energy={particle_energy:.2f} MeV")

                # 2. Get Unity observation (agent's prediction)
                logger.info("Getting Unity observation...")
                unity_state = self.unity_env.reset()
                unity_obs = unity_state['observations'][0]  # Shape: (10,)

                logger.info(f"  Unity obs shape: {unity_obs.shape}")

                # 3. Run Geant4 simulation (ground truth)
                logger.info("Running Geant4 simulation...")
                geant4_result = self.geant4_runner.run_simulation(parameters)

                if not geant4_result['success']:
                    logger.error(f"  Geant4 failed: {geant4_result.get('error')}")
                    failed += 1
                    continue

                energy_deposit = geant4_result['total_energy_deposit']
                logger.info(f"  ✅ Geant4: {energy_deposit:.3f} MeV deposited")

                # 4. Collect data pair
                simulation_data = {
                    'unity_observation': unity_obs.tolist(),
                    'geant4_energy_deposit': energy_deposit,
                    'geant4_events': geant4_result['events'],
                    'parameters': parameters,
                    'timestamp': time.time()
                }

                self.data_collector.collect_simulation_pair(
                    unity_observation=unity_obs,
                    geant4_result=geant4_result,
                    parameters=parameters
                )

                collected += 1
                logger.info(f"  ✅ Collected: {collected}/{num_samples} "
                          f"(Failed: {failed})")

                # Small delay to avoid overwhelming system
                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("\n⚠️  Data collection interrupted by user")

        finally:
            # Finalize and save
            logger.info(f"\n{'='*60}")
            logger.info("Finalizing data collection...")
            self.data_collector.finalize()
            self.unity_env.close()

            logger.info(f"\n✅ Data collection complete!")
            logger.info(f"   Collected: {collected} samples")
            logger.info(f"   Failed: {failed} samples")
            logger.info(f"   Success rate: {collected/(collected+failed)*100:.1f}%")
            logger.info(f"={'='*60}\n")


def main():
    """Main entry point"""

    print("\n" + "🔬"*30)
    print("UNITY ML-AGENTS + GEANT4 PIPELINE")
    print("🔬"*30 + "\n")

    # Configuration
    GEANT4_EXE = r"C:\Thesis\geant4\Water-Phantom\build\Release\WaterPhantomSim.exe"
    OUTPUT_DIR = r"C:\Thesis\python\training_data"
    NUM_SAMPLES = 50  # Start with 50 samples

    print("Configuration:")
    print(f"  Geant4: {GEANT4_EXE}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Samples: {NUM_SAMPLES}")
    print()

    # Check if Geant4 exists
    if not Path(GEANT4_EXE).exists():
        print(f"❌ ERROR: Geant4 executable not found!")
        print(f"   Path: {GEANT4_EXE}")
        return 1

    print("📋 INSTRUCTIONS:")
    print("  1. Make sure Unity is OPEN")
    print("  2. TestScene is loaded")
    print("  3. Unity is NOT in Play mode")
    print("  4. Press Enter to start...")
    input()

    try:
        # Create pipeline
        pipeline = MLAgentsGeant4Pipeline(
            geant4_executable=GEANT4_EXE,
            output_directory=OUTPUT_DIR
        )

        print("\n🎮 NOW click PLAY in Unity!")
        print("Waiting for connection...\n")

        # Collect data
        pipeline.collect_training_data(num_samples=NUM_SAMPLES)

        print("\n🎉 SUCCESS! Training data collected!")
        print(f"📁 Data saved to: {OUTPUT_DIR}")

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())