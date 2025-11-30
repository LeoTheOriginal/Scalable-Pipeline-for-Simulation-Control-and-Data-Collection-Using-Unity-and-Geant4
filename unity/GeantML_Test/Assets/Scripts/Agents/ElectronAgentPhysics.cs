using Core;
using Physics;
using System;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using Random = UnityEngine.Random;

namespace Agents
{
    /// <summary>
    /// Training mode for the electron agent.
    /// </summary>
    public enum TrainingMode
    {
        /// <summary>
        /// Physics-based training: rewards based on physical constraints only,
        /// no Geant4 reference. Best for learning physics from scratch.
        /// </summary>
        PhysicsBased,

        /// <summary>
        /// Geant4 Statistical: runs Geant4 each episode, compares trajectory
        /// STATISTICS (not step-by-step). This is the recommended mode for thesis.
        /// </summary>
        Geant4Statistical,

        /// <summary>
        /// Hybrid: physics constraints + Geant4 statistical validation.
        /// </summary>
        Hybrid,

        /// <summary>
        /// Inference only: no training, no Geant4, just run the learned policy.
        /// </summary>
        Inference
    }

    /// <summary>
    /// Improved ElectronAgent using physics-based rewards and statistical Geant4 validation.
    /// 
    /// Key features:
    /// 1. No teacher forcing or scheduled sampling
    /// 2. Rewards based on physics consistency (Bethe-Bloch, Highland, CSDA range)
    /// 3. Statistical validation against Geant4 (not step-by-step matching)
    /// 4. Trajectory-level rewards computed at episode end
    /// </summary>
    public class ElectronAgentPhysics : Agent
    {
        // ====================================================================
        // INSPECTOR SETTINGS
        // ====================================================================

        [Header("Training Configuration")]
        [Tooltip("Training approach to use")]
        public TrainingMode Mode = TrainingMode.Geant4Statistical;

        [Tooltip("Agent index for multi-agent training (0, 1, 2, ...)")]
        public int AgentIndex = 0;

        [Header("Simulation Settings")]
        [Tooltip("Maximum steps per episode")]
        public int MaxSteps = 500;

        [Tooltip("Show trajectory visualization")]
        public bool ShowVisualization = true;

        [Header("Physics Constraints")]
        [Tooltip("Maximum step size in cm (recommended: 0.02-0.05)")]
        public float MaxStepSize = 0.03f;

        [Tooltip("Minimum step size in cm")]
        public float MinStepSize = 0.005f;

        [Header("Reward Weights (Physics-Based)")]
        [Tooltip("Weight for scattering angle validity")]
        public float W_Scattering = 15f;

        [Tooltip("Weight for forward progress")]
        public float W_Forward = 5f;

        [Tooltip("Weight for trajectory smoothness")]
        public float W_Smoothness = 8f;

        [Tooltip("Weight for range consistency")]
        public float W_Range = 20f;

        [Tooltip("Weight for staying in phantom")]
        public float W_Boundary = 50f;

        [Header("Trajectory Reward (End of Episode)")]
        [Tooltip("Weight for total path length vs CSDA range")]
        public float W_TotalRange = 30f;

        [Tooltip("Weight for proper energy depletion")]
        public float W_EnergyDepletion = 25f;

        [Tooltip("Weight for trajectory coherence")]
        public float W_Coherence = 15f;

        [Header("Geant4 Statistical Comparison Weights")]
        [Tooltip("Weight for path length match with Geant4")]
        public float W_Geant4Path = 25f;

        [Tooltip("Weight for final depth match with Geant4")]
        public float W_Geant4Depth = 20f;

        [Tooltip("Weight for lateral spread match with Geant4")]
        public float W_Geant4Lateral = 15f;

        [Tooltip("Tolerance for statistical match (0.2 = 20%)")]
        [Range(0.1f, 0.5f)]
        public float StatisticalTolerance = 0.25f;

        [Header("Debug")]
        [Tooltip("Log detailed step information")]
        public bool VerboseLogging = false;

        [Tooltip("Log every N steps")]
        public int LogInterval = 50;

        // ====================================================================
        // PRIVATE STATE
        // ====================================================================

        // Current particle state
        private Vector3 _position;
        private Vector3 _momentumDirection;
        private float _energy;

        // Previous step state (for smoothness calculation)
        private Vector3 _previousPosition;
        private Vector3 _previousDirection;
        private float _previousEnergy;

        // Initial state (for forward direction reference)
        private Vector3 _initialPosition;
        private Vector3 _initialDirection;
        private float _initialEnergy;

        // Trajectory tracking
        private int _currentStep;
        private float _cumulativePathLength;
        private float _totalEnergyDeposited;
        private List<Vector3> _trajectoryPositions;
        private List<float> _trajectoryEnergies;
        private List<float> _scatteringAngles;

        // Physics reference values
        private float _expectedCSDARange;
        private float _remainingRange;

        // Geant4 data
        private float[] _geant4Buffer;
        private int _geant4TrajectoryLength;

        // Geant4 statistics (computed once per episode)
        private float _geant4PathLength;
        private float _geant4FinalDepth;
        private float _geant4LateralSpread;
        private float _geant4FinalEnergy;
        private bool _geant4DataValid;

        // Statistics collection
        private float _episodeRewardSum;

        // ====================================================================
        // INITIALIZATION
        // ====================================================================

        public override void Initialize()
        {
            _trajectoryPositions = new List<Vector3>(MaxSteps);
            _trajectoryEnergies = new List<float>(MaxSteps);
            _scatteringAngles = new List<float>(MaxSteps);
            _geant4Buffer = new float[MaxSteps * 7];

            _expectedCSDARange = ElectronPhysics.GetInitialCSDARange();

            Debug.Log($"[ElectronAgent #{AgentIndex}] Initialized");
            Debug.Log($"  Mode: {Mode}");
            Debug.Log($"  Expected CSDA Range: {_expectedCSDARange:F3} cm");
            Debug.Log($"  Max Step Size: {MaxStepSize} cm");

            if (Mode == TrainingMode.Geant4Statistical || Mode == TrainingMode.Hybrid)
            {
                Debug.Log($"  Geant4 Statistical Comparison: ENABLED");
                Debug.Log($"  Statistical Tolerance: {StatisticalTolerance:P0}");
            }
        }

        // ====================================================================
        // EPISODE LIFECYCLE
        // ====================================================================

        public override void OnEpisodeBegin()
        {
            _currentStep = 0;
            _cumulativePathLength = 0f;
            _totalEnergyDeposited = 0f;
            _episodeRewardSum = 0f;
            _geant4DataValid = false;

            _trajectoryPositions.Clear();
            _trajectoryEnergies.Clear();
            _scatteringAngles.Clear();

            // Standard initial conditions (matching Geant4)
            _initialPosition = new Vector3(-6f, 0f, 0f);
            _initialDirection = new Vector3(1f, 0f, 0f);
            _initialEnergy = ElectronPhysics.INITIAL_ENERGY;

            // Set current state
            _position = _initialPosition;
            _momentumDirection = _initialDirection;
            _energy = _initialEnergy;

            _previousPosition = _position;
            _previousDirection = _momentumDirection;
            _previousEnergy = _energy;

            // Calculate physics references
            _expectedCSDARange = ElectronPhysics.CalculateCSDARange(_energy);
            _remainingRange = _expectedCSDARange;

            // Get Geant4 reference trajectory for statistical validation
            if (Mode == TrainingMode.Geant4Statistical || Mode == TrainingMode.Hybrid)
            {
                FetchGeant4Reference();
            }

            // Record initial state
            _trajectoryPositions.Add(_position);
            _trajectoryEnergies.Add(_energy);

            // Update visualization
            if (ShowVisualization)
            {
                transform.localPosition = _position;
            }

            if (VerboseLogging)
            {
                Debug.Log($"[Agent #{AgentIndex}] Episode {CompletedEpisodes} started");
                if (_geant4DataValid)
                {
                    Debug.Log($"  Geant4 reference: {_geant4TrajectoryLength} steps, " +
                             $"path={_geant4PathLength:F2}cm, depth={_geant4FinalDepth:F2}cm");
                }
            }
        }

        private void FetchGeant4Reference()
        {
            if (Mode == TrainingMode.Inference) return;

            try
            {
                _geant4TrajectoryLength = Geant4Interface.RunSimulationBatch(_geant4Buffer, MaxSteps);

                if (_geant4TrajectoryLength >= 2)
                {
                    // Calculate Geant4 statistics
                    _geant4PathLength = CalculateGeant4PathLength();
                    _geant4FinalDepth = GetGeant4FinalDepth();
                    _geant4LateralSpread = GetGeant4LateralSpread();
                    _geant4FinalEnergy = GetGeant4FinalEnergy();
                    _geant4DataValid = true;

                    if (VerboseLogging)
                    {
                        Debug.Log($"[Agent #{AgentIndex}] Geant4 stats: " +
                                 $"path={_geant4PathLength:F3}cm, " +
                                 $"depth={_geant4FinalDepth:F3}cm, " +
                                 $"lateral={_geant4LateralSpread:F3}cm, " +
                                 $"finalE={_geant4FinalEnergy:F3}MeV");
                    }
                }
                else
                {
                    Debug.LogWarning($"[Agent #{AgentIndex}] Geant4 returned insufficient data ({_geant4TrajectoryLength} steps)");
                    _geant4DataValid = false;
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[Agent #{AgentIndex}] Geant4 fetch failed: {e.Message}");
                _geant4DataValid = false;
            }
        }

        // ====================================================================
        // GEANT4 STATISTICS CALCULATION
        // ====================================================================

        private float CalculateGeant4PathLength()
        {
            float total = 0f;
            for (int i = 0; i < _geant4TrajectoryLength - 1; i++)
            {
                int idx = i * 7;
                int nextIdx = (i + 1) * 7;
                Vector3 pos1 = new Vector3(_geant4Buffer[idx], _geant4Buffer[idx + 1], _geant4Buffer[idx + 2]);
                Vector3 pos2 = new Vector3(_geant4Buffer[nextIdx], _geant4Buffer[nextIdx + 1], _geant4Buffer[nextIdx + 2]);
                total += Vector3.Distance(pos1, pos2);
            }
            return total;
        }

        private float GetGeant4FinalDepth()
        {
            if (_geant4TrajectoryLength < 1) return 0f;
            int idx = (_geant4TrajectoryLength - 1) * 7;
            // Depth = how far into phantom (phantom starts at x = -5)
            return _geant4Buffer[idx] - ElectronPhysics.PHANTOM_ENTRY_X;
        }

        private float GetGeant4LateralSpread()
        {
            if (_geant4TrajectoryLength < 1) return 0f;
            int idx = (_geant4TrajectoryLength - 1) * 7;
            return Mathf.Sqrt(_geant4Buffer[idx + 1] * _geant4Buffer[idx + 1] +
                             _geant4Buffer[idx + 2] * _geant4Buffer[idx + 2]);
        }

        private float GetGeant4FinalEnergy()
        {
            if (_geant4TrajectoryLength < 1) return 0f;
            int idx = (_geant4TrajectoryLength - 1) * 7;
            return _geant4Buffer[idx + 6];
        }

        // ====================================================================
        // OBSERVATIONS
        // ====================================================================

        /// <summary>
        /// Observation space (11 values):
        /// - Position (3): x, y, z in cm (normalized)
        /// - Momentum direction (3): normalized direction vector
        /// - Energy (1): kinetic energy in MeV (normalized by initial)
        /// - Remaining range (1): expected remaining path length (normalized)
        /// - Depth in phantom (1): how far into phantom (normalized)
        /// - Path length fraction (1): cumulative path / CSDA range
        /// - Forward progress (1): dot product of current vs initial direction
        /// </summary>
        public override void CollectObservations(VectorSensor sensor)
        {
            // Position (normalized to phantom size)
            sensor.AddObservation(_position.x / ElectronPhysics.PHANTOM_HALF_SIZE);
            sensor.AddObservation(_position.y / ElectronPhysics.PHANTOM_HALF_SIZE);
            sensor.AddObservation(_position.z / ElectronPhysics.PHANTOM_HALF_SIZE);

            // Momentum direction (already normalized)
            sensor.AddObservation(_momentumDirection.x);
            sensor.AddObservation(_momentumDirection.y);
            sensor.AddObservation(_momentumDirection.z);

            // Energy (normalized by initial energy)
            sensor.AddObservation(_energy / ElectronPhysics.INITIAL_ENERGY);

            // Remaining range (normalized by CSDA range)
            _remainingRange = ElectronPhysics.CalculateRemainingRange(_energy);
            sensor.AddObservation(_remainingRange / _expectedCSDARange);

            // Depth in phantom (0 = entry, 1 = crossed full phantom)
            float depthInPhantom = (_position.x - ElectronPhysics.PHANTOM_ENTRY_X) / (2f * ElectronPhysics.PHANTOM_HALF_SIZE);
            sensor.AddObservation(Mathf.Clamp01(depthInPhantom));

            // Path length fraction (how much of expected range traveled)
            float pathFraction = _cumulativePathLength / _expectedCSDARange;
            sensor.AddObservation(Mathf.Clamp(pathFraction, 0f, 2f) / 2f);

            // Forward progress (dot product with initial direction)
            float forwardness = Vector3.Dot(_momentumDirection, _initialDirection);
            sensor.AddObservation(forwardness);
        }

        // ====================================================================
        // ACTIONS
        // ====================================================================

        /// <summary>
        /// Action space (4 continuous values):
        /// - Delta direction (3): direction change vector
        /// - Step size factor (1): 0-1 mapping to [MinStepSize, MaxStepSize]
        /// 
        /// Energy change is computed from physics (Bethe-Bloch), not predicted!
        /// </summary>
        public override void OnActionReceived(ActionBuffers actions)
        {
            // Check for episode end conditions
            if (ShouldEndEpisode())
            {
                ProcessEpisodeEnd();
                return;
            }

            var act = actions.ContinuousActions;

            // Parse actions
            Vector3 deltaDirection = new Vector3(act[0], act[1], act[2]);
            float stepSizeFactor = (act[3] + 1f) / 2f;

            // Apply action with physics constraints
            ApplyAction(deltaDirection, stepSizeFactor);

            // Calculate step reward
            float reward = CalculateStepReward();
            AddReward(reward);
            _episodeRewardSum += reward;

            // Update visualization
            if (ShowVisualization)
            {
                transform.localPosition = _position;
            }

            // Logging
            if (VerboseLogging && _currentStep % LogInterval == 0)
            {
                LogStepInfo(reward);
            }

            _currentStep++;
        }

        private void ApplyAction(Vector3 deltaDirection, float stepSizeFactor)
        {
            // Store previous state
            _previousPosition = _position;
            _previousDirection = _momentumDirection;
            _previousEnergy = _energy;

            // Calculate step size (bounded)
            float stepSize = Mathf.Lerp(MinStepSize, MaxStepSize, stepSizeFactor);

            // Calculate new direction (blend current direction with action)
            Vector3 directionDelta = deltaDirection * 0.3f;
            Vector3 newDirection = (_momentumDirection + directionDelta).normalized;

            // Ensure we don't get zero vector
            if (newDirection.magnitude < 0.001f)
            {
                newDirection = _momentumDirection;
            }

            // Calculate scattering angle
            float scatterAngle = Vector3.Angle(_momentumDirection, newDirection);
            _scatteringAngles.Add(scatterAngle);

            // Update direction
            _momentumDirection = newDirection;

            // Calculate position change
            Vector3 deltaPos = _momentumDirection * stepSize;
            _position += deltaPos;

            // Update path length
            _cumulativePathLength += stepSize;

            // Calculate energy loss using Bethe-Bloch (physics-based, not learned!)
            float energyLoss = ElectronPhysics.CalculateEnergyLoss(_energy, stepSize);

            // Add stochastic fluctuation (Landau fluctuations)
            float fluctuation = Random.Range(0.8f, 1.2f);
            energyLoss *= fluctuation;

            _energy -= energyLoss;
            _energy = Mathf.Max(0f, _energy);
            _totalEnergyDeposited += energyLoss;

            // Update remaining range
            _remainingRange = ElectronPhysics.CalculateRemainingRange(_energy);

            // Record trajectory
            _trajectoryPositions.Add(_position);
            _trajectoryEnergies.Add(_energy);
        }

        // ====================================================================
        // REWARD CALCULATION (PHYSICS-BASED)
        // ====================================================================

        private float CalculateStepReward()
        {
            float reward = 0f;

            // 1. SCATTERING ANGLE VALIDITY
            float lastScatterAngle = _scatteringAngles[_scatteringAngles.Count - 1];
            float maxAllowedAngle = ElectronPhysics.CalculateMaxScatteringAngle(_previousEnergy, MaxStepSize) * Mathf.Rad2Deg;
            maxAllowedAngle = Mathf.Clamp(maxAllowedAngle, 10f, 45f);

            if (lastScatterAngle <= maxAllowedAngle)
            {
                reward += W_Scattering * 0.1f;
            }
            else if (lastScatterAngle <= maxAllowedAngle * 2f)
            {
                float excess = (lastScatterAngle - maxAllowedAngle) / maxAllowedAngle;
                reward -= W_Scattering * excess * 0.5f;
            }
            else
            {
                float excess = (lastScatterAngle - maxAllowedAngle) / maxAllowedAngle;
                reward -= W_Scattering * excess;
            }

            // 2. FORWARD PROGRESS
            float forwardness = Vector3.Dot(_momentumDirection, _initialDirection);

            if (forwardness > 0.5f)
            {
                reward += W_Forward * 0.2f;
            }
            else if (forwardness > 0f)
            {
                reward += W_Forward * 0.1f * forwardness;
            }
            else if (forwardness > -0.3f)
            {
                reward -= W_Forward * 0.3f;
            }
            else
            {
                reward -= W_Forward * (1f - forwardness);
            }

            // 3. TRAJECTORY SMOOTHNESS
            if (_currentStep > 1 && _trajectoryPositions.Count >= 3)
            {
                Vector3 prevDelta = _previousPosition - _trajectoryPositions[_trajectoryPositions.Count - 3];
                Vector3 currDelta = _position - _previousPosition;

                float smoothnessAngle = Vector3.Angle(prevDelta, currDelta);

                if (smoothnessAngle < 15f)
                {
                    reward += W_Smoothness * 0.1f;
                }
                else if (smoothnessAngle > 60f)
                {
                    reward -= W_Smoothness * (smoothnessAngle - 60f) / 60f;
                }
            }

            // 4. BOUNDARY AWARENESS
            if (!IsInPhantom(_position))
            {
                float distanceOutside = GetDistanceOutsidePhantom(_position);
                reward -= W_Boundary * (1f + distanceOutside);
            }

            // 5. RANGE CONSISTENCY
            float pathVsRangeFraction = _cumulativePathLength / _expectedCSDARange;
            float energyFraction = _energy / _initialEnergy;
            float expectedEnergyRemaining = 1f - pathVsRangeFraction;
            float energyConsistency = 1f - Mathf.Abs(energyFraction - Mathf.Max(0f, expectedEnergyRemaining));

            if (energyConsistency > 0.8f)
            {
                reward += W_Range * 0.05f;
            }
            else if (energyConsistency < 0.5f)
            {
                reward -= W_Range * (0.5f - energyConsistency) * 0.2f;
            }

            // 6. SURVIVAL BONUS
            reward += 0.01f;

            return reward;
        }

        // ====================================================================
        // EPISODE END HANDLING
        // ====================================================================

        private bool ShouldEndEpisode()
        {
            if (_currentStep >= MaxSteps - 1) return true;
            if (_energy <= 0.01f) return true;
            if (GetDistanceOutsidePhantom(_position) > 2f) return true;
            return false;
        }

        private void ProcessEpisodeEnd()
        {
            // Calculate trajectory-level rewards (physics-based)
            float trajectoryReward = CalculateTrajectoryReward();

            // Calculate Geant4 statistical comparison reward
            float geant4Reward = 0f;
            if ((Mode == TrainingMode.Geant4Statistical || Mode == TrainingMode.Hybrid) && _geant4DataValid)
            {
                geant4Reward = CalculateGeant4StatisticalReward();
            }

            float totalReward = trajectoryReward + geant4Reward;
            AddReward(totalReward);
            _episodeRewardSum += totalReward;

            // Logging
            if (CompletedEpisodes % 100 == 0 || VerboseLogging)
            {
                LogEpisodeSummary(trajectoryReward, geant4Reward);
            }

            EndEpisode();
        }

        private float CalculateTrajectoryReward()
        {
            float reward = 0f;

            // 1. TOTAL PATH LENGTH vs CSDA RANGE
            float pathRatio = _cumulativePathLength / _expectedCSDARange;
            float expectedDetour = ElectronPhysics.GetExpectedDetourFactor(_initialEnergy);
            float idealPathLength = _expectedCSDARange * expectedDetour;
            float pathError = Mathf.Abs(_cumulativePathLength - idealPathLength) / idealPathLength;

            if (pathError < 0.1f)
            {
                reward += W_TotalRange * (1f - pathError);
            }
            else if (pathError < 0.3f)
            {
                reward += W_TotalRange * 0.5f * (1f - pathError);
            }
            else
            {
                reward -= W_TotalRange * pathError * 0.5f;
            }

            // 2. ENERGY DEPLETION
            float remainingEnergyFraction = _energy / _initialEnergy;

            if (remainingEnergyFraction < 0.05f)
            {
                reward += W_EnergyDepletion * (1f - remainingEnergyFraction);
            }
            else if (remainingEnergyFraction < 0.2f)
            {
                reward += W_EnergyDepletion * 0.5f * (1f - remainingEnergyFraction);
            }
            else
            {
                reward -= W_EnergyDepletion * remainingEnergyFraction;
            }

            // Penalize if energy gone but path too short
            if (_energy <= 0.01f && pathRatio < 0.7f)
            {
                reward -= W_TotalRange * 0.5f;
            }

            // 3. TRAJECTORY COHERENCE
            if (_scatteringAngles.Count > 0)
            {
                float meanScatterAngle = 0f;
                foreach (float angle in _scatteringAngles)
                {
                    meanScatterAngle += angle;
                }
                meanScatterAngle /= _scatteringAngles.Count;

                float expectedMeanAngle = ElectronPhysics.CalculateRMSScatteringAngle(
                    _initialEnergy / 2f, MaxStepSize) * Mathf.Rad2Deg;
                expectedMeanAngle = Mathf.Max(5f, expectedMeanAngle);

                float angleRatio = meanScatterAngle / expectedMeanAngle;

                if (angleRatio > 0.5f && angleRatio < 2f)
                {
                    reward += W_Coherence * 0.5f;
                }
                else if (angleRatio >= 2f)
                {
                    reward -= W_Coherence * (angleRatio - 2f) * 0.5f;
                }
            }

            // 4. COMPLETION BONUS
            if (_energy <= 0.01f && pathRatio > 0.8f && pathRatio < 1.5f)
            {
                reward += 50f;
            }

            return reward;
        }

        /// <summary>
        /// Statistical comparison with Geant4 trajectory.
        /// NOT step-by-step matching - compares overall statistics!
        /// </summary>
        private float CalculateGeant4StatisticalReward()
        {
            if (!_geant4DataValid) return 0f;

            float reward = 0f;

            // Agent's statistics
            float agentFinalDepth = _position.x - ElectronPhysics.PHANTOM_ENTRY_X;
            float agentLateralSpread = Mathf.Sqrt(_position.y * _position.y + _position.z * _position.z);

            // 1. PATH LENGTH COMPARISON
            // Agent's path should be statistically similar to Geant4's path
            if (_geant4PathLength > 0.1f)
            {
                float pathError = Mathf.Abs(_cumulativePathLength - _geant4PathLength) / _geant4PathLength;

                if (pathError < StatisticalTolerance)
                {
                    // Good match - reward proportional to accuracy
                    reward += W_Geant4Path * (1f - pathError / StatisticalTolerance);
                }
                else if (pathError < StatisticalTolerance * 2f)
                {
                    // Partial match
                    reward += W_Geant4Path * 0.3f * (1f - pathError / (StatisticalTolerance * 2f));
                }
                else
                {
                    // Poor match - small penalty
                    reward -= W_Geant4Path * 0.2f;
                }
            }

            // 2. FINAL DEPTH COMPARISON
            // How deep did the electron penetrate?
            if (_geant4FinalDepth > 0.1f)
            {
                float depthError = Mathf.Abs(agentFinalDepth - _geant4FinalDepth) / _geant4FinalDepth;

                if (depthError < StatisticalTolerance)
                {
                    reward += W_Geant4Depth * (1f - depthError / StatisticalTolerance);
                }
                else if (depthError < StatisticalTolerance * 2f)
                {
                    reward += W_Geant4Depth * 0.3f * (1f - depthError / (StatisticalTolerance * 2f));
                }
                else
                {
                    reward -= W_Geant4Depth * 0.2f;
                }
            }

            // 3. LATERAL SPREAD COMPARISON
            // How much did the electron spread sideways?
            // Note: This is more variable, so we're more lenient
            float lateralTolerance = StatisticalTolerance * 1.5f;
            float maxLateral = Mathf.Max(_geant4LateralSpread, agentLateralSpread, 0.1f);
            float lateralError = Mathf.Abs(agentLateralSpread - _geant4LateralSpread) / maxLateral;

            if (lateralError < lateralTolerance)
            {
                reward += W_Geant4Lateral * (1f - lateralError / lateralTolerance);
            }
            else if (lateralError < lateralTolerance * 2f)
            {
                reward += W_Geant4Lateral * 0.2f;
            }

            // 4. BONUS: Good overall match
            float overallMatch = 0f;
            if (_geant4PathLength > 0.1f)
                overallMatch += (1f - Mathf.Abs(_cumulativePathLength - _geant4PathLength) / _geant4PathLength) / 3f;
            if (_geant4FinalDepth > 0.1f)
                overallMatch += (1f - Mathf.Abs(agentFinalDepth - _geant4FinalDepth) / _geant4FinalDepth) / 3f;
            overallMatch += (1f - lateralError) / 3f;

            if (overallMatch > 0.7f)
            {
                reward += 30f * (overallMatch - 0.7f) / 0.3f;
            }

            return reward;
        }

        // ====================================================================
        // UTILITY METHODS
        // ====================================================================

        private bool IsInPhantom(Vector3 pos)
        {
            float halfSize = ElectronPhysics.PHANTOM_HALF_SIZE;
            return Mathf.Abs(pos.x) <= halfSize &&
                   Mathf.Abs(pos.y) <= halfSize &&
                   Mathf.Abs(pos.z) <= halfSize;
        }

        private float GetDistanceOutsidePhantom(Vector3 pos)
        {
            float halfSize = ElectronPhysics.PHANTOM_HALF_SIZE;
            float dx = Mathf.Max(0f, Mathf.Abs(pos.x) - halfSize);
            float dy = Mathf.Max(0f, Mathf.Abs(pos.y) - halfSize);
            float dz = Mathf.Max(0f, Mathf.Abs(pos.z) - halfSize);
            return Mathf.Sqrt(dx * dx + dy * dy + dz * dz);
        }

        private void LogStepInfo(float reward)
        {
            Debug.Log($"[Agent #{AgentIndex} Step {_currentStep}] " +
                     $"Pos: ({_position.x:F2}, {_position.y:F2}, {_position.z:F2}), " +
                     $"E: {_energy:F2} MeV, " +
                     $"Path: {_cumulativePathLength:F2} cm, " +
                     $"Reward: {reward:F2}");
        }

        private void LogEpisodeSummary(float physicsReward, float geant4Reward)
        {
            float pathRatio = _cumulativePathLength / _expectedCSDARange;
            float remainingEnergy = _energy / _initialEnergy * 100f;

            Debug.Log($"[Agent #{AgentIndex}] Episode {CompletedEpisodes} Summary:");
            Debug.Log($"  Steps: {_currentStep}");
            Debug.Log($"  Agent Path: {_cumulativePathLength:F2} cm ({pathRatio:P0} of CSDA)");
            Debug.Log($"  Remaining Energy: {remainingEnergy:F1}%");
            Debug.Log($"  Physics Reward: {physicsReward:F2}");

            if (_geant4DataValid)
            {
                float agentDepth = _position.x - ElectronPhysics.PHANTOM_ENTRY_X;
                Debug.Log($"  Geant4 Path: {_geant4PathLength:F2} cm");
                Debug.Log($"  Geant4 Depth: {_geant4FinalDepth:F2} cm | Agent Depth: {agentDepth:F2} cm");
                Debug.Log($"  Geant4 Reward: {geant4Reward:F2}");
            }

            Debug.Log($"  Total Reward: {_episodeRewardSum:F2}");
        }

        // ====================================================================
        // HEURISTIC (for testing)
        // ====================================================================

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var actions = actionsOut.ContinuousActions;

            actions[0] = 0.8f + Random.Range(-0.1f, 0.1f);
            actions[1] = Random.Range(-0.2f, 0.2f);
            actions[2] = Random.Range(-0.2f, 0.2f);
            actions[3] = 0.5f;
        }
    }
}