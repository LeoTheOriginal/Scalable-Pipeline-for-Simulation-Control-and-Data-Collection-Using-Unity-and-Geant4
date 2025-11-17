"""
REST API Server for Unity-Geant4 Communication
Updated with batch processing for trajectory aggregation
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from geant4_interface.realtime_simulator import RealtimeGeant4Simulator
from rl_training.reward_calculator import StepRewardCalculator
from rl_training.trajectory_aggregator import TrajectoryAggregator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)

# Global state
active_simulators: Dict[int, RealtimeGeant4Simulator] = {}
reward_calculator = StepRewardCalculator()

# Trajectory aggregator for batch processing
trajectory_aggregator = None

# Configuration
GEANT4_EXECUTABLE = r"C:\Thesis\geant4\Water-Phantom\build\Release\WaterPhantomSim.exe"


def initialize_trajectory_aggregator():
    """Initialize trajectory aggregator on first use"""
    global trajectory_aggregator
    if trajectory_aggregator is None:
        trajectory_aggregator = TrajectoryAggregator(
            buffer_size=1000,
            geant4_executable=GEANT4_EXECUTABLE,
            num_workers=8,
            auto_process=False  # Manual control via endpoint
        )
        logger.info("Trajectory aggregator initialized")


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    Test with: http://localhost:5000/health
    """
    return jsonify({
        'status': 'ok',
        'active_agents': len(active_simulators),
        'geant4_version': '11.3.2',
        'server_time': time.time(),
        'trajectory_buffer_size': trajectory_aggregator.buffer_size if trajectory_aggregator else 0,
        'trajectories_buffered': len(trajectory_aggregator.pending_trajectories) if trajectory_aggregator else 0
    })


@app.route('/initialize', methods=['POST'])
def initialize_agent():
    """
    Initialize particle with initial conditions

    Request body:
    {
        "agent_id": 0,
        "particle_type": "e-",
        "initial_energy": 10.0,
        "initial_position": [-6.0, 0.0, 0.0],
        "initial_direction": [1.0, 0.0, 0.0]
    }
    """
    try:
        data = request.json
        agent_id = data['agent_id']

        logger.info(f"Initializing agent {agent_id}")

        # Create simulator if doesn't exist
        if agent_id not in active_simulators:
            active_simulators[agent_id] = RealtimeGeant4Simulator(GEANT4_EXECUTABLE)
            logger.info(f"  Created new simulator for agent {agent_id}")

        simulator = active_simulators[agent_id]

        # Initialize particle
        simulator.initialize_particle(
            particle_type=data['particle_type'],
            energy=float(data['initial_energy']),
            position=np.array(data['initial_position'], dtype=float),
            direction=np.array(data['initial_direction'], dtype=float)
        )

        logger.info(f"  Agent {agent_id} initialized: {data['particle_type']}, "
                   f"{data['initial_energy']:.2f} MeV")

        return jsonify({
            'success': True,
            'agent_id': agent_id,
            'message': 'Agent initialized successfully'
        })

    except Exception as e:
        logger.error(f"Initialize error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/step', methods=['POST'])
def execute_step():
    """
    Execute single step and return reward

    Request body:
    {
        "agent_id": 0,
        "unity_position": [x, y, z],
        "unity_direction": [dx, dy, dz],
        "unity_energy": energy,
        "energy_deposited": deposited
    }
    """
    try:
        start_time = time.time()

        data = request.json
        agent_id = data['agent_id']

        # Check if agent exists
        if agent_id not in active_simulators:
            return jsonify({
                'success': False,
                'error': 'Agent not initialized. Call /initialize first.'
            }), 400

        simulator = active_simulators[agent_id]

        # Prepare Unity state
        unity_state = {
            'position': np.array(data['unity_position'], dtype=float),
            'energy': float(data['unity_energy']),
            'direction': np.array(data['unity_direction'], dtype=float),
            'energy_deposited': float(data.get('energy_deposited', 0.0))
        }

        # Execute Geant4 step
        geant4_result = simulator.execute_step(
            unity_position=unity_state['position'],
            unity_energy=unity_state['energy'],
            unity_direction=unity_state['direction']
        )

        # Calculate reward
        reward_data = reward_calculator.calculate_step_reward(
            unity_state,
            geant4_result
        )

        processing_time = (time.time() - start_time) * 1000  # ms

        # Build response
        response = {
            'success': True,
            'agent_id': agent_id,
            'reward': float(reward_data['reward']),
            'geant4_state': {
                'position': geant4_result['position'].tolist(),
                'energy': float(geant4_result['energy']),
                'direction': geant4_result['direction'].tolist(),
                'energy_deposited': float(geant4_result['energy_deposited']),
                'step_length': float(geant4_result['step_length']),
                'process_name': geant4_result['process_name']
            },
            'metrics': {
                'position_error': reward_data['position_error'],
                'energy_error': reward_data['energy_error'],
                'direction_error': reward_data['direction_error']
            },
            'episode_done': reward_data['should_terminate'],
            'termination_reason': reward_data['termination_reason'],
            'processing_time_ms': processing_time
        }

        # Log periodically
        if simulator.step_count % 50 == 0:
            logger.info(f"Agent {agent_id} step {simulator.step_count}: "
                       f"Reward={reward_data['reward']:.3f}, "
                       f"PosErr={reward_data['position_error']:.3f}cm, "
                       f"Latency={processing_time:.1f}ms")

        return jsonify(response)

    except Exception as e:
        logger.error(f"Step execution error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# NEW ENDPOINTS - BATCH PROCESSING
# ============================================================================

@app.route('/trajectory/submit', methods=['POST'])
def submit_trajectory():
    """
    Submit completed trajectory to buffer

    Meeting requirement: "buforujemy wiele historii cząstki (>1000)"

    Request body:
    {
        "agent_id": 0,
        "initial_conditions": {
            "particle_type": "e-",
            "initial_energy": 10.0,
            "initial_position": [-6.0, 0.0, 0.0],
            "initial_direction": [1.0, 0.0, 0.0]
        },
        "steps": [
            {
                "step_number": 0,
                "position": [x, y, z],
                "direction": [dx, dy, dz],
                "energy": e,
                "energy_deposited": de
            },
            ...
        ]
    }

    Response:
    {
        "success": true,
        "trajectory_id": 123,
        "buffer_count": 456,
        "buffer_size": 1000
    }
    """
    try:
        initialize_trajectory_aggregator()

        data = request.json
        agent_id = data['agent_id']
        initial_conditions = data['initial_conditions']
        steps = data['steps']

        # Add trajectory to buffer
        trajectory_id = trajectory_aggregator.add_trajectory(
            agent_id=agent_id,
            initial_conditions=initial_conditions,
            steps=steps
        )

        buffer_stats = trajectory_aggregator.get_statistics()

        logger.info(f"Trajectory {trajectory_id} from agent {agent_id} added to buffer "
                   f"({buffer_stats['pending_trajectories']}/{buffer_stats['buffer_size']})")

        return jsonify({
            'success': True,
            'trajectory_id': trajectory_id,
            'buffer_count': buffer_stats['pending_trajectories'],
            'buffer_size': buffer_stats['buffer_size'],
            'buffer_utilization': buffer_stats['buffer_utilization']
        })

    except Exception as e:
        logger.error(f"Submit trajectory error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/trajectory/process_batch', methods=['POST'])
def process_trajectory_batch():
    """
    Process buffered trajectories with Geant4

    Meeting requirement:
    - "buforujemy wiele historii cząstki (>1000)"
    - "odpalamy na kilku agentach równolegle"

    Request body (optional):
    {
        "max_trajectories": 1000  // Optional, defaults to buffer_size
    }

    Response:
    {
        "success": true,
        "trajectories_processed": 1000,
        "processing_time_seconds": 45.2,
        "results": [
            {
                "trajectory_id": 123,
                "agent_id": 0,
                "episode_summary": {
                    "total_reward": 12.5,
                    "mean_position_error": 0.45,
                    ...
                },
                "step_rewards": [...]
            },
            ...
        ]
    }
    """
    try:
        initialize_trajectory_aggregator()

        buffer_stats = trajectory_aggregator.get_statistics()

        if buffer_stats['pending_trajectories'] == 0:
            return jsonify({
                'success': False,
                'error': 'No trajectories in buffer to process'
            }), 400

        logger.info(f"Processing trajectory batch: "
                   f"{buffer_stats['pending_trajectories']} trajectories")

        start_time = time.time()

        # Process batch
        results = trajectory_aggregator.process_buffer()

        processing_time = time.time() - start_time

        logger.info(f"✅ Batch processed: {len(results)} trajectories in {processing_time:.2f}s")

        return jsonify({
            'success': True,
            'trajectories_processed': len(results),
            'processing_time_seconds': processing_time,
            'results': results
        })

    except Exception as e:
        logger.error(f"Process batch error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/trajectory/buffer_status', methods=['GET'])
def get_buffer_status():
    """
    Get current buffer status

    Response:
    {
        "buffer_count": 456,
        "buffer_size": 1000,
        "buffer_utilization": 0.456,
        "total_received": 12345,
        "total_processed": 11000
    }
    """
    try:
        initialize_trajectory_aggregator()

        stats = trajectory_aggregator.get_statistics()

        return jsonify({
            'success': True,
            **stats
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/reset', methods=['POST'])
def reset_agent():
    """
    Reset agent simulation

    Request body:
    {
        "agent_id": 0
    }
    """
    try:
        data = request.json
        agent_id = data['agent_id']

        if agent_id in active_simulators:
            active_simulators[agent_id].reset()
            logger.info(f"Agent {agent_id} reset")

            return jsonify({
                'success': True,
                'message': f'Agent {agent_id} reset successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Agent not found'
            }), 404

    except Exception as e:
        logger.error(f"Reset error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/shutdown', methods=['POST'])
def shutdown_agent():
    """
    Remove agent from active simulators

    Request body:
    {
        "agent_id": 0
    }
    """
    try:
        data = request.json
        agent_id = data['agent_id']

        if agent_id in active_simulators:
            del active_simulators[agent_id]
            logger.info(f"Agent {agent_id} removed")

            return jsonify({
                'success': True,
                'message': f'Agent {agent_id} removed'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Agent not found'
            }), 404

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def main():
    """Start REST API server"""

    print("\n" + "=" * 70)
    print("🚀 REST API SERVER FOR UNITY-GEANT4 INTEGRATION")
    print("=" * 70)
    print(f"  Server URL: http://localhost:5000")
    print(f"  Geant4 exe: {GEANT4_EXECUTABLE}")
    print()
    print("  Available endpoints:")
    print("    GET  /health                    - Health check")
    print("    POST /initialize                - Initialize particle")
    print("    POST /step                      - Execute step + get reward")
    print("    POST /reset                     - Reset agent")
    print("    POST /shutdown                  - Remove agent")
    print()
    print("    === NEW: Batch Processing ===")
    print("    POST /trajectory/submit         - Submit completed trajectory")
    print("    POST /trajectory/process_batch  - Process buffered trajectories")
    print("    GET  /trajectory/buffer_status  - Get buffer status")
    print()
    print("  Press Ctrl+C to stop")
    print("=" * 70 + "\n")

    # Check if Geant4 exists
    if not Path(GEANT4_EXECUTABLE).exists():
        logger.warning(f"⚠️  Geant4 executable not found: {GEANT4_EXECUTABLE}")
        logger.warning(f"   Server will start but simulations will fail!")

    # Start Flask server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )


if __name__ == '__main__':
    main()