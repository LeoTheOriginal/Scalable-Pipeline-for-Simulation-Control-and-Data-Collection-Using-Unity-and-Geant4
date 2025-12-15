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
        /// Combines physics constraints with Geant4 statistical validation.
        /// </summary>
        Geant4Statistical,

        /// <summary>
        /// Inference only: no training, no Geant4, just run the learned policy.
        /// Use this mode after training to evaluate the model.
        /// </summary>
        Inference
    }

    /// <summary>
    /// Electron transport RL agent with REALISTIC SCATTERING requirements.
    /// 
    /// KEY FIX: Agent MUST scatter realistically, not just go straight!
    /// - Anti-spiral detection: penalizes consistent same-direction turns
    /// - Scattering distribution matching: requires variance in angles
    /// - Minimum deflection requirement: can't just go straight
    /// 
    /// Multi-agent support:
    /// - Each agent has unique AgentIndex for identification
    /// - Agents can have different reward weights for comparison
    /// - TensorBoard logs are tagged by agent index
    /// </summary>
    public class ElectronAgentPhysics : Agent
    {
        // ====================================================================
        // INSPECTOR SETTINGS
        // ====================================================================

        [Header("Training Configuration")]
        [Tooltip("Training approach to use")]
        public TrainingMode Mode = TrainingMode.Geant4Statistical;

        [Tooltip("Agent index for multi-agent training (0, 1, 2, ...).")]
        public int AgentIndex = 0;

        [Header("Simulation Settings")]
        [Tooltip("Maximum steps per episode. Typical trajectories complete in 250-350 steps.")]
        public int MaxSteps = 500;

        [Tooltip("Show trajectory visualization (position updates on Transform)")]
        public bool ShowVisualization = true;

        [Header("Physics Constraints")]
        [Tooltip("Maximum step size in cm")]
        public float MaxStepSize = 0.03f;

        [Tooltip("Minimum step size in cm")]
        public float MinStepSize = 0.005f;

        [Header("=== CRITICAL: Scattering Requirements ===")]
        [Tooltip("Weight for scattering angle being within Highland bounds")]
        public float W_ScatteringBounds = 10f;

        [Tooltip("Weight for having VARIANCE in scattering (anti-straight-line)")]
        public float W_ScatteringVariance = 25f;

        [Tooltip("Weight for anti-spiral (penalize consistent same-direction turns)")]
        public float W_AntiSpiral = 30f;

        [Tooltip("Weight for matching expected mean scattering angle")]
        public float W_MeanScattering = 20f;

        [Tooltip("Minimum required scattering angle std dev (degrees)")]
        public float MinScatteringStdDev = 2f;

        [Header("Step Reward Weights")]
        [Tooltip("Weight for energy-range consistency (CSDA relationship)")]
        public float W_Range = 15f;

        [Tooltip("Small survival bonus per step")]
        public float SurvivalBonus = 0.01f;

        [Header("Episode-End Reward Weights")]
        [Tooltip("Weight for total path length vs CSDA range")]
        public float W_TotalRange = 20f;

        [Tooltip("Weight for proper energy depletion")]
        public float W_EnergyDepletion = 20f;

        [Tooltip("Fixed penalty for exiting phantom boundaries")]
        public float BoundaryExitPenalty = 50f;

        [Header("Geant4 Statistical Comparison Weights")]
        [Tooltip("Weight for path length match with Geant4")]
        public float W_Geant4Path = 20f;

        [Tooltip("Weight for final depth match with Geant4")]
        public float W_Geant4Depth = 15f;

        [Tooltip("Weight for lateral spread match with Geant4")]
        public float W_Geant4Lateral = 15f;

        [Tooltip("Weight for scattering distribution match with Geant4")]
        public float W_Geant4Scattering = 25f;

        [Tooltip("Tolerance for statistical match (0.25 = 25%)")]
        [Range(0.1f, 0.5f)]
        public float StatisticalTolerance = 0.25f;

        [Header("Lateral Distribution Rewards (Normal Distribution)")]
        [Tooltip("Weight for lateral deviation matching normal distribution")]
        public float W_LateralDistribution = 15f;

        [Tooltip("Weight for step-level lateral change")]
        public float W_StepLateral = 5f;

        [Tooltip("Enable normal distribution rewards")]
        public bool UseNormalDistributionRewards = true;

        [Header("Debug")]
        [Tooltip("Log detailed step information")]
        public bool VerboseLogging = false;

        [Tooltip("Log every N steps")]
        public int LogInterval = 50;

        // ====================================================================
        // EVENTS (for visualization)
        // ====================================================================

        /// <summary>
        /// Event fired when agent takes a step. Use for trajectory visualization.
        /// </summary>
        public event Action<Vector3> OnStepTaken;

        /// <summary>
        /// Event fired when episode begins. Use to clear trajectory visualization.
        /// </summary>
        public event Action OnEpisodeReset;

        // ====================================================================
        // PRIVATE STATE
        // ====================================================================

        // Current particle state
        private Vector3 _position;
        private Vector3 _momentumDirection;
        private float _energy;

        // Previous step state
        private Vector3 _previousPosition;
        private Vector3 _previousDirection;
        private float _previousEnergy;

        // Initial state
        private Vector3 _initialPosition;
        private Vector3 _initialDirection;
        private float _initialEnergy;

        // Trajectory tracking
        private int _currentStep;
        private float _cumulativePathLength;
        private float _totalEnergyDeposited;
        private List<Vector3> _trajectoryPositions;
        private List<float> _trajectoryEnergies;

        // Lateral position tracking for distribution rewards
        private List<float> _lateralPositionsY;
        private List<float> _lateralPositionsZ;
        private float _previousLateralY;
        private float _previousLateralZ;

        // SCATTERING TRACKING (critical for anti-straight-line)
        private List<float> _scatteringAngles;          // Magnitude of each scatter
        private List<Vector3> _scatteringAxes;          // Axis of each scatter (for spiral detection)
        private Vector3 _cumulativeAngularMomentum;     // Detects spiraling
        private float _totalAbsoluteScattering;         // Sum of |angle|

        // Physics reference values
        private float _expectedCSDARange;
        private float _remainingRange;

        // Episode termination state
        private bool _exitedBoundary;

        // Geant4 data
        private float[] _geant4Buffer;
        private int _geant4TrajectoryLength;
        private float _geant4PathLength;
        private float _geant4FinalDepth;
        private float _geant4LateralSpread;
        private float _geant4FinalEnergy;
        private float _geant4MeanScatterAngle;
        private float _geant4ScatterStdDev;
        private bool _geant4DataValid;

        // Statistics
        private float _episodeRewardSum;
        private int _totalEpisodes;
        private int _boundaryExitCount;
        private int _straightLineCount; // Episodes that were too straight

        // ====================================================================
        // INITIALIZATION
        // ====================================================================

        public override void Initialize()
        {
            _trajectoryPositions = new List<Vector3>(MaxSteps);
            _trajectoryEnergies = new List<float>(MaxSteps);
            _scatteringAngles = new List<float>(MaxSteps);
            _scatteringAxes = new List<Vector3>(MaxSteps);
            _lateralPositionsY = new List<float>(MaxSteps);
            _lateralPositionsZ = new List<float>(MaxSteps);
            _geant4Buffer = new float[MaxSteps * 7];

            _expectedCSDARange = ElectronPhysics.GetInitialCSDARange();
            _totalEpisodes = 0;
            _boundaryExitCount = 0;
            _straightLineCount = 0;

            Debug.Log($"[ElectronAgent #{AgentIndex}] Initialized with ANTI-SPIRAL protection");
            Debug.Log($"  Mode: {Mode}");
            Debug.Log($"  Initial Position: {ElectronPhysics.GetInitialPosition()}");
            Debug.Log($"  Min Scattering StdDev: {MinScatteringStdDev}°");
            Debug.Log($"  Anti-Spiral Weight: {W_AntiSpiral}");
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
            _exitedBoundary = false;
            _cumulativeAngularMomentum = Vector3.zero;
            _totalAbsoluteScattering = 0f;

            _trajectoryPositions.Clear();
            _trajectoryEnergies.Clear();

            _lateralPositionsY.Clear();
            _lateralPositionsZ.Clear();
            _previousLateralY = 0f;
            _previousLateralZ = 0f;

            _scatteringAngles.Clear();
            _scatteringAxes.Clear();

            // Standard initial conditions at phantom boundary
            _initialPosition = ElectronPhysics.GetInitialPosition();
            _initialDirection = ElectronPhysics.GetInitialDirection();
            _initialEnergy = ElectronPhysics.INITIAL_ENERGY;

            _position = _initialPosition;
            _momentumDirection = _initialDirection;
            _energy = _initialEnergy;

            _previousPosition = _position;
            _previousDirection = _momentumDirection;
            _previousEnergy = _energy;

            _expectedCSDARange = ElectronPhysics.CalculateCSDARange(_energy);
            _remainingRange = _expectedCSDARange;

            // Get Geant4 reference
            if (Mode == TrainingMode.Geant4Statistical)
            {
                FetchGeant4Reference();
            }

            _trajectoryPositions.Add(_position);
            _trajectoryEnergies.Add(_energy);

            // Fire event for visualization
            OnEpisodeReset?.Invoke();

            if (ShowVisualization)
            {
                transform.localPosition = _position;
            }

            _totalEpisodes++;

            if (VerboseLogging)
            {
                Debug.Log($"[Agent #{AgentIndex}] Episode {CompletedEpisodes} started");
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
                    _geant4PathLength = CalculateGeant4PathLength();
                    _geant4FinalDepth = GetGeant4FinalDepth();
                    _geant4LateralSpread = GetGeant4LateralSpread();
                    _geant4FinalEnergy = GetGeant4FinalEnergy();

                    // Calculate Geant4 scattering statistics
                    CalculateGeant4ScatteringStats();

                    _geant4DataValid = true;

                    if (VerboseLogging)
                    {
                        Debug.Log($"[Agent #{AgentIndex}] Geant4: path={_geant4PathLength:F2}cm, " +
                                 $"meanScatter={_geant4MeanScatterAngle:F2}°, stdDev={_geant4ScatterStdDev:F2}°");
                    }
                }
                else
                {
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
        // GEANT4 SCATTERING STATISTICS
        // ====================================================================

        private void CalculateGeant4ScatteringStats()
        {
            List<float> g4Angles = new List<float>();

            for (int i = 0; i < _geant4TrajectoryLength - 2; i++)
            {
                int idx0 = i * 7;
                int idx1 = (i + 1) * 7;
                int idx2 = (i + 2) * 7;

                Vector3 pos0 = new Vector3(_geant4Buffer[idx0], _geant4Buffer[idx0 + 1], _geant4Buffer[idx0 + 2]);
                Vector3 pos1 = new Vector3(_geant4Buffer[idx1], _geant4Buffer[idx1 + 1], _geant4Buffer[idx1 + 2]);
                Vector3 pos2 = new Vector3(_geant4Buffer[idx2], _geant4Buffer[idx2 + 1], _geant4Buffer[idx2 + 2]);

                Vector3 dir1 = (pos1 - pos0).normalized;
                Vector3 dir2 = (pos2 - pos1).normalized;

                if (dir1.magnitude > 0.001f && dir2.magnitude > 0.001f)
                {
                    float angle = Vector3.Angle(dir1, dir2);
                    g4Angles.Add(angle);
                }
            }

            if (g4Angles.Count > 0)
            {
                // Calculate mean
                float sum = 0f;
                foreach (float a in g4Angles) sum += a;
                _geant4MeanScatterAngle = sum / g4Angles.Count;

                // Calculate std dev
                float variance = 0f;
                foreach (float a in g4Angles)
                {
                    float diff = a - _geant4MeanScatterAngle;
                    variance += diff * diff;
                }
                _geant4ScatterStdDev = Mathf.Sqrt(variance / g4Angles.Count);
            }
            else
            {
                _geant4MeanScatterAngle = 5f; // Default expected value
                _geant4ScatterStdDev = 3f;
            }
        }

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
        /// Observation space (14 values):
        /// - Position (3): x, y, z normalized
        /// - Momentum direction (3): normalized direction vector
        /// - Energy (1): normalized by initial
        /// - Remaining range (1): normalized by CSDA
        /// - Depth in phantom (1): normalized
        /// - Path length fraction (1): cumulative / CSDA
        /// - Recent scattering mean (1): average of last 10 scattering angles
        /// - Recent scattering variance (1): variance of last 10 angles
        /// - Spiral indicator (1): magnitude of cumulative angular momentum
        /// </summary>
        public override void CollectObservations(VectorSensor sensor)
        {
            // Position (3)
            sensor.AddObservation(_position.x / ElectronPhysics.PHANTOM_HALF_SIZE);
            sensor.AddObservation(_position.y / ElectronPhysics.PHANTOM_HALF_SIZE);
            sensor.AddObservation(_position.z / ElectronPhysics.PHANTOM_HALF_SIZE);

            // Momentum direction (3)
            sensor.AddObservation(_momentumDirection.x);
            sensor.AddObservation(_momentumDirection.y);
            sensor.AddObservation(_momentumDirection.z);

            // Energy (1)
            sensor.AddObservation(_energy / ElectronPhysics.INITIAL_ENERGY);

            // Remaining range (1)
            _remainingRange = ElectronPhysics.CalculateRemainingRange(_energy);
            sensor.AddObservation(_remainingRange / _expectedCSDARange);

            // Depth in phantom (1)
            float depthInPhantom = (_position.x - ElectronPhysics.PHANTOM_ENTRY_X) / (2f * ElectronPhysics.PHANTOM_HALF_SIZE);
            sensor.AddObservation(Mathf.Clamp01(depthInPhantom));

            // Path length fraction (1)
            float pathFraction = _cumulativePathLength / _expectedCSDARange;
            sensor.AddObservation(Mathf.Clamp(pathFraction, 0f, 2f) / 2f);

            // Recent scattering statistics (helps agent know if it's scattering enough)
            float recentMean = 0f;
            float recentVariance = 0f;
            int lookback = Mathf.Min(10, _scatteringAngles.Count);

            if (lookback > 0)
            {
                // Calculate mean of recent angles
                for (int i = _scatteringAngles.Count - lookback; i < _scatteringAngles.Count; i++)
                {
                    recentMean += _scatteringAngles[i];
                }
                recentMean /= lookback;

                // Calculate variance
                for (int i = _scatteringAngles.Count - lookback; i < _scatteringAngles.Count; i++)
                {
                    float diff = _scatteringAngles[i] - recentMean;
                    recentVariance += diff * diff;
                }
                recentVariance /= lookback;
            }

            // Recent scattering mean (1) - normalized to ~[0,1] assuming max ~30 degrees
            sensor.AddObservation(recentMean / 30f);

            // Recent scattering variance (1) - normalized
            sensor.AddObservation(Mathf.Sqrt(recentVariance) / 15f);

            // Spiral indicator (1) - how much is the agent spiraling?
            float spiralMagnitude = _cumulativeAngularMomentum.magnitude / Mathf.Max(1f, _currentStep);
            sensor.AddObservation(Mathf.Clamp01(spiralMagnitude * 10f));
        }

        // ====================================================================
        // ACTIONS
        // ====================================================================

        public override void OnActionReceived(ActionBuffers actions)
        {
            if (ShouldEndEpisode())
            {
                ProcessEpisodeEnd();
                return;
            }

            var act = actions.ContinuousActions;

            Vector3 deltaDirection = new Vector3(act[0], act[1], act[2]);
            float stepSizeFactor = (act[3] + 1f) / 2f;

            ApplyAction(deltaDirection, stepSizeFactor);

            if (!IsInPhantom(_position))
            {
                _exitedBoundary = true;
                ProcessEpisodeEnd();
                return;
            }

            float reward = CalculateStepReward();
            AddReward(reward);
            _episodeRewardSum += reward;

            // ====================================================================
            // ❌ BŁĄD: TĘ LINIĘ PONIŻEJ MUSISZ USUNĄĆ! 
            // Ona wysyła pozycję lokalną, co psuje wizualizację (tworzy "gwiazdę" na środku).
            // OnStepTaken?.Invoke(_position);  <-- USUŃ TO
            // ====================================================================

            // 1. Najpierw zaktualizuj fizyczną pozycję Unity
            if (ShowVisualization)
            {
                // _position to pozycja lokalna względem TrainingEnvironment
                transform.localPosition = _position;
            }

            // ✅ 2. To jest poprawne wywołanie (Global Position)
            // Dzięki temu LineRenderer dostanie koordynaty świata.
            OnStepTaken?.Invoke(transform.position);

            if (VerboseLogging && _currentStep % LogInterval == 0)
            {
                LogStepInfo(reward);
            }

            _currentStep++;
        }

        private void ApplyAction(Vector3 deltaDirection, float stepSizeFactor)
        {
            _previousPosition = _position;
            _previousDirection = _momentumDirection;
            _previousEnergy = _energy;

            float stepSize = Mathf.Lerp(MinStepSize, MaxStepSize, stepSizeFactor);

            // Calculate new direction
            Vector3 directionDelta = deltaDirection * 0.3f;
            Vector3 newDirection = (_momentumDirection + directionDelta).normalized;

            if (newDirection.magnitude < 0.001f)
            {
                newDirection = _momentumDirection;
            }

            // Calculate scattering angle (magnitude)
            float scatterAngle = Vector3.Angle(_momentumDirection, newDirection);
            _scatteringAngles.Add(scatterAngle);
            _totalAbsoluteScattering += scatterAngle;

            // Calculate scattering axis (for spiral detection)
            Vector3 scatterAxis = Vector3.Cross(_momentumDirection, newDirection);
            _scatteringAxes.Add(scatterAxis);

            // Accumulate angular momentum (spiral detection)
            // If agent always turns the same way, this will grow large
            _cumulativeAngularMomentum += scatterAxis;

            // Update direction
            _momentumDirection = newDirection;

            // Update position
            Vector3 deltaPos = _momentumDirection * stepSize;
            _position += deltaPos;
            _cumulativePathLength += stepSize;

            // Physics-based energy loss
            float energyLoss = ElectronPhysics.CalculateEnergyLoss(_energy, stepSize);
            float fluctuation = Random.Range(0.8f, 1.2f);
            energyLoss *= fluctuation;

            _energy -= energyLoss;
            _energy = Mathf.Max(0f, _energy);
            _totalEnergyDeposited += energyLoss;

            _remainingRange = ElectronPhysics.CalculateRemainingRange(_energy);

            _trajectoryPositions.Add(_position);
            _trajectoryEnergies.Add(_energy);
            // Track lateral positions for distribution rewards
            _lateralPositionsY.Add(_position.y);
            _lateralPositionsZ.Add(_position.z);
        }

        // ====================================================================
        // STEP REWARD (with anti-straight-line and anti-spiral)
        // ====================================================================

        private float CalculateStepReward()
        {
            float reward = 0f;

            // 1. SCATTERING BOUNDS CHECK
            float lastScatterAngle = _scatteringAngles[_scatteringAngles.Count - 1];
            float expectedRMS = ElectronPhysics.CalculateRMSScatteringAngle(_previousEnergy, MaxStepSize) * Mathf.Rad2Deg;
            expectedRMS = Mathf.Max(3f, expectedRMS);

            float maxAllowed = expectedRMS * 4f;
            float minExpected = expectedRMS * 0.3f;

            if (lastScatterAngle >= minExpected && lastScatterAngle <= maxAllowed)
            {
                reward += W_ScatteringBounds * 0.1f;
            }
            else if (lastScatterAngle < minExpected)
            {
                float straightPenalty = (minExpected - lastScatterAngle) / minExpected;
                reward -= W_ScatteringBounds * straightPenalty * 0.3f;
            }
            else
            {
                float excessAngle = (lastScatterAngle - maxAllowed) / maxAllowed;
                reward -= W_ScatteringBounds * excessAngle * 0.2f;
            }

            // 2. ANTI-SPIRAL CHECK (every 10 steps)
            if (_currentStep > 0 && _currentStep % 10 == 0 && _scatteringAxes.Count >= 10)
            {
                Vector3 recentAxisSum = Vector3.zero;
                int checkSteps = Mathf.Min(10, _scatteringAxes.Count);

                for (int i = _scatteringAxes.Count - checkSteps; i < _scatteringAxes.Count; i++)
                {
                    recentAxisSum += _scatteringAxes[i].normalized;
                }

                float spiralIndicator = recentAxisSum.magnitude / checkSteps;

                if (spiralIndicator > 0.5f)
                {
                    reward -= W_AntiSpiral * (spiralIndicator - 0.5f) * 2f;
                }
                else
                {
                    reward += W_AntiSpiral * 0.05f;
                }
            }

            // 3. SCATTERING VARIANCE CHECK (every 20 steps)
            if (_currentStep > 0 && _currentStep % 20 == 0 && _scatteringAngles.Count >= 20)
            {
                float sum = 0f;
                int lookback = 20;
                for (int i = _scatteringAngles.Count - lookback; i < _scatteringAngles.Count; i++)
                {
                    sum += _scatteringAngles[i];
                }
                float mean = sum / lookback;

                float variance = 0f;
                for (int i = _scatteringAngles.Count - lookback; i < _scatteringAngles.Count; i++)
                {
                    float diff = _scatteringAngles[i] - mean;
                    variance += diff * diff;
                }
                float stdDev = Mathf.Sqrt(variance / lookback);

                if (stdDev < MinScatteringStdDev)
                {
                    float consistencyPenalty = (MinScatteringStdDev - stdDev) / MinScatteringStdDev;
                    reward -= W_ScatteringVariance * consistencyPenalty * 0.5f;
                }
                else
                {
                    reward += W_ScatteringVariance * 0.1f;
                }
            }

            // 4. RANGE CONSISTENCY
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

            // 5. SURVIVAL BONUS
            reward += SurvivalBonus;

            // ================================================================
            // 6. NEW: NORMAL DISTRIBUTION LATERAL REWARDS
            // ================================================================
            if (UseNormalDistributionRewards)
            {
                // Calculate depth fraction (how far into phantom)
                float depthFraction = (_position.x - ElectronPhysics.PHANTOM_ENTRY_X) /
                                     (2f * ElectronPhysics.PHANTOM_HALF_SIZE);
                depthFraction = Mathf.Clamp01(depthFraction);

                // a) Step-level lateral change reward
                if (_lateralPositionsY.Count >= 2)
                {
                    float deltaY = _position.y - _previousLateralY;
                    float deltaZ = _position.z - _previousLateralZ;

                    float stepSize = Vector3.Distance(_position, _previousPosition);

                    // Each step's lateral change should follow Highland distribution
                    float stepLateralRewardY = NormalDistributionRewards.CalculateStepLateralReward(
                        deltaY, stepSize, _previousEnergy, W_StepLateral * 0.5f);
                    float stepLateralRewardZ = NormalDistributionRewards.CalculateStepLateralReward(
                        deltaZ, stepSize, _previousEnergy, W_StepLateral * 0.5f);

                    reward += (stepLateralRewardY + stepLateralRewardZ) * 0.1f;
                }

                // b) Overall lateral position reward (encourage normal distribution)
                // Check every 10 steps to avoid excessive computation
                if (_currentStep > 0 && _currentStep % 10 == 0)
                {
                    float lateralReward = NormalDistributionRewards.CalculateLateralDeviationReward(
                        _position.y, _position.z, depthFraction, W_LateralDistribution * 0.1f);
                    reward += lateralReward;
                }

                // c) Scattering angle reward using ±2σ/±3σ bounds
                float scatterReward = NormalDistributionRewards.CalculateScatteringAngleReward(
                    lastScatterAngle, _previousEnergy, MaxStepSize, W_MeanScattering * 0.05f);
                reward += scatterReward;

                // Update previous lateral positions
                _previousLateralY = _position.y;
                _previousLateralZ = _position.z;
            }

            return reward;
        }

        // ====================================================================
        // EPISODE END
        // ====================================================================

        private bool ShouldEndEpisode()
        {
            if (_currentStep >= MaxSteps - 1) return true;
            if (_energy <= 0.01f) return true;
            return false;
        }

        private void ProcessEpisodeEnd()
        {
            float trajectoryReward = 0f;
            float geant4Reward = 0f;

            if (_exitedBoundary)
            {
                trajectoryReward = -BoundaryExitPenalty;
                _boundaryExitCount++;
            }
            else
            {
                trajectoryReward = CalculateTrajectoryReward();

                if (Mode == TrainingMode.Geant4Statistical && _geant4DataValid)
                {
                    geant4Reward = CalculateGeant4StatisticalReward();
                }
            }

            float totalReward = trajectoryReward + geant4Reward;
            AddReward(totalReward);
            _episodeRewardSum += totalReward;

            if (CompletedEpisodes % 100 == 0 || VerboseLogging)
            {
                LogEpisodeSummary(trajectoryReward, geant4Reward);
            }

            EndEpisode();
        }

        private float CalculateTrajectoryReward()
        {
            float reward = 0f;

            // 1. PATH LENGTH vs CSDA
            float pathRatio = _cumulativePathLength / _expectedCSDARange;
            float expectedDetour = ElectronPhysics.GetExpectedDetourFactor(_initialEnergy);
            float idealPathLength = _expectedCSDARange * expectedDetour;
            float pathError = Mathf.Abs(_cumulativePathLength - idealPathLength) / idealPathLength;

            if (pathError < 0.15f)
            {
                reward += W_TotalRange * (1f - pathError);
            }
            else if (pathError < 0.3f)
            {
                reward += W_TotalRange * 0.5f * (1f - pathError);
            }
            else
            {
                reward -= W_TotalRange * pathError * 0.3f;
            }

            // 2. ENERGY DEPLETION
            float remainingEnergyFraction = _energy / _initialEnergy;
            if (remainingEnergyFraction < 0.05f)
            {
                reward += W_EnergyDepletion * (1f - remainingEnergyFraction);
            }
            else if (remainingEnergyFraction < 0.2f)
            {
                reward += W_EnergyDepletion * 0.5f;
            }
            else
            {
                reward -= W_EnergyDepletion * remainingEnergyFraction;
            }

            // 3. OVERALL SCATTERING QUALITY
            if (_scatteringAngles.Count > 10)
            {
                // Calculate overall statistics
                float sum = 0f;
                foreach (float a in _scatteringAngles) sum += a;
                float meanAngle = sum / _scatteringAngles.Count;

                float variance = 0f;
                foreach (float a in _scatteringAngles)
                {
                    float diff = a - meanAngle;
                    variance += diff * diff;
                }
                float stdDev = Mathf.Sqrt(variance / _scatteringAngles.Count);

                // Check mean scattering
                float expectedMean = ElectronPhysics.CalculateRMSScatteringAngle(_initialEnergy / 2f, MaxStepSize) * Mathf.Rad2Deg;
                expectedMean = Mathf.Max(3f, expectedMean);

                float meanError = Mathf.Abs(meanAngle - expectedMean) / expectedMean;
                if (meanError < 0.5f)
                {
                    reward += W_MeanScattering * (1f - meanError);
                }
                else
                {
                    reward -= W_MeanScattering * meanError * 0.3f;
                }

                // Check variance (must have variance!)
                if (stdDev < MinScatteringStdDev)
                {
                    // TOO STRAIGHT! Big penalty!
                    reward -= W_ScatteringVariance * 2f;
                    _straightLineCount++;

                    if (VerboseLogging)
                    {
                        Debug.LogWarning($"[Agent #{AgentIndex}] TOO STRAIGHT! StdDev={stdDev:F2}° < {MinScatteringStdDev}°");
                    }
                }
                else
                {
                    reward += W_ScatteringVariance * 0.5f;
                }

                // Check spiral (cumulative angular momentum)
                float spiralMagnitude = _cumulativeAngularMomentum.magnitude / _scatteringAngles.Count;
                if (spiralMagnitude > 0.3f)
                {
                    // SPIRALING! Penalty
                    reward -= W_AntiSpiral * spiralMagnitude;
                }
                else
                {
                    reward += W_AntiSpiral * 0.3f;
                }
            }

            // 4. COMPLETION BONUS (only if proper physics!)
            if (_energy <= 0.01f && pathRatio > 0.8f && pathRatio < 1.5f)
            {
                // Calculate final scattering quality
                float sum = 0f;
                foreach (float a in _scatteringAngles) sum += a;
                float meanAngle = sum / _scatteringAngles.Count;

                float variance = 0f;
                foreach (float a in _scatteringAngles)
                {
                    float diff = a - meanAngle;
                    variance += diff * diff;
                }
                float stdDev = Mathf.Sqrt(variance / _scatteringAngles.Count);

                float spiralMag = _cumulativeAngularMomentum.magnitude / _scatteringAngles.Count;

                // Only bonus if good scattering behavior
                if (stdDev >= MinScatteringStdDev && spiralMag < 0.3f)
                {
                    reward += 50f;
                }
            }

            // ================================================================
            // NEW: Distribution quality reward at episode end
            // ================================================================
            if (UseNormalDistributionRewards && _lateralPositionsY.Count > 10)
            {
                // Convert lists to arrays for reward calculation
                float[] lateralY = _lateralPositionsY.ToArray();
                float[] lateralZ = _lateralPositionsZ.ToArray();

                // Reward for Y distribution quality
                float distQualityY = NormalDistributionRewards.CalculateDistributionQualityReward(
                    lateralY, W_LateralDistribution * 0.3f);

                // Reward for Z distribution quality  
                float distQualityZ = NormalDistributionRewards.CalculateDistributionQualityReward(
                    lateralZ, W_LateralDistribution * 0.3f);

                reward += distQualityY + distQualityZ;

                // Bonus for symmetric distribution (Y and Z should have similar spread)
                float sumY = 0f, sumZ = 0f;
                for (int i = 0; i < lateralY.Length; i++)
                {
                    sumY += lateralY[i] * lateralY[i];
                    sumZ += lateralZ[i] * lateralZ[i];
                }
                float rmsY = Mathf.Sqrt(sumY / lateralY.Length);
                float rmsZ = Mathf.Sqrt(sumZ / lateralZ.Length);

                float symmetryError = Mathf.Abs(rmsY - rmsZ) / Mathf.Max(rmsY, rmsZ, 0.01f);
                if (symmetryError < 0.2f)
                {
                    reward += W_LateralDistribution * 0.2f; // Bonus for symmetry
                }
            }

            return reward;
        }

        private float CalculateGeant4StatisticalReward()
        {
            if (!_geant4DataValid) return 0f;

            float reward = 0f;

            float agentFinalDepth = _position.x - ElectronPhysics.PHANTOM_ENTRY_X;
            float agentLateralSpread = Mathf.Sqrt(_position.y * _position.y + _position.z * _position.z);

            // 1. PATH LENGTH
            if (_geant4PathLength > 0.1f)
            {
                float pathError = Mathf.Abs(_cumulativePathLength - _geant4PathLength) / _geant4PathLength;
                if (pathError < StatisticalTolerance)
                {
                    reward += W_Geant4Path * (1f - pathError / StatisticalTolerance);
                }
                else if (pathError < StatisticalTolerance * 2f)
                {
                    reward += W_Geant4Path * 0.3f;
                }
                else
                {
                    reward -= W_Geant4Path * 0.2f;
                }
            }

            // 2. FINAL DEPTH
            if (_geant4FinalDepth > 0.1f)
            {
                float depthError = Mathf.Abs(agentFinalDepth - _geant4FinalDepth) / _geant4FinalDepth;
                if (depthError < StatisticalTolerance)
                {
                    reward += W_Geant4Depth * (1f - depthError / StatisticalTolerance);
                }
                else
                {
                    reward -= W_Geant4Depth * 0.1f;
                }
            }

            // 3. LATERAL SPREAD
            float lateralTolerance = StatisticalTolerance * 1.5f;
            float maxLateral = Mathf.Max(_geant4LateralSpread, agentLateralSpread, 0.1f);
            float lateralError = Mathf.Abs(agentLateralSpread - _geant4LateralSpread) / maxLateral;

            if (lateralError < lateralTolerance)
            {
                reward += W_Geant4Lateral * (1f - lateralError / lateralTolerance);
            }

            // 4. SCATTERING DISTRIBUTION MATCH (NEW - critical!)
            if (_scatteringAngles.Count > 10 && _geant4MeanScatterAngle > 0.1f)
            {
                // Calculate agent's scattering stats
                float sum = 0f;
                foreach (float a in _scatteringAngles) sum += a;
                float agentMean = sum / _scatteringAngles.Count;

                float variance = 0f;
                foreach (float a in _scatteringAngles)
                {
                    float diff = a - agentMean;
                    variance += diff * diff;
                }
                float agentStdDev = Mathf.Sqrt(variance / _scatteringAngles.Count);

                // Compare means
                float meanError = Mathf.Abs(agentMean - _geant4MeanScatterAngle) / _geant4MeanScatterAngle;
                if (meanError < 0.3f)
                {
                    reward += W_Geant4Scattering * 0.5f * (1f - meanError / 0.3f);
                }

                // Compare std devs
                if (_geant4ScatterStdDev > 0.1f)
                {
                    float stdError = Mathf.Abs(agentStdDev - _geant4ScatterStdDev) / _geant4ScatterStdDev;
                    if (stdError < 0.5f)
                    {
                        reward += W_Geant4Scattering * 0.5f * (1f - stdError / 0.5f);
                    }
                }
            }

            return reward;
        }

        // ====================================================================
        // UTILITY
        // ====================================================================

        private bool IsInPhantom(Vector3 pos)
        {
            return ElectronPhysics.IsInsidePhantom(pos);
        }

        private void LogStepInfo(float reward)
        {
            float spiralMag = _cumulativeAngularMomentum.magnitude / Mathf.Max(1f, _currentStep);
            Debug.Log($"[Agent #{AgentIndex} Step {_currentStep}] " +
                     $"E={_energy:F2}MeV, Spiral={spiralMag:F3}, R={reward:F2}");
        }

        private void LogEpisodeSummary(float physicsReward, float geant4Reward)
        {
            float pathRatio = _cumulativePathLength / _expectedCSDARange;

            // Calculate scattering stats
            float meanAngle = 0f;
            float stdDev = 0f;
            if (_scatteringAngles.Count > 0)
            {
                float sum = 0f;
                foreach (float a in _scatteringAngles) sum += a;
                meanAngle = sum / _scatteringAngles.Count;

                float variance = 0f;
                foreach (float a in _scatteringAngles)
                {
                    float diff = a - meanAngle;
                    variance += diff * diff;
                }
                stdDev = Mathf.Sqrt(variance / _scatteringAngles.Count);
            }

            float spiralMag = _cumulativeAngularMomentum.magnitude / Mathf.Max(1f, _scatteringAngles.Count);

            Debug.Log($"[Agent #{AgentIndex}] Episode {CompletedEpisodes}:");
            Debug.Log($"  Path: {_cumulativePathLength:F2}cm ({pathRatio:P0} CSDA)");
            Debug.Log($"  Scattering: mean={meanAngle:F2}°, std={stdDev:F2}°, spiral={spiralMag:F3}");
            Debug.Log($"  Rewards: physics={physicsReward:F1}, g4={geant4Reward:F1}, total={_episodeRewardSum:F1}");
            Debug.Log($"  Stats: exits={_boundaryExitCount}, straight={_straightLineCount}");
        }

        // ====================================================================
        // PUBLIC ACCESSORS
        // ====================================================================

        public List<Vector3> GetTrajectoryPositions() => new List<Vector3>(_trajectoryPositions);
        public float GetCurrentEnergy() => _energy;
        public Vector3 GetCurrentPosition() => _position;
        public float GetCumulativePathLength() => _cumulativePathLength;
        public bool DidExitBoundary() => _exitedBoundary;
        public int GetStraightLineCount() => _straightLineCount;

        // ====================================================================
        // HEURISTIC
        // ====================================================================

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var actions = actionsOut.ContinuousActions;

            // Random but physically plausible scattering
            float expectedAngle = ElectronPhysics.CalculateRMSScatteringAngle(_energy, MaxStepSize);

            // Random direction change
            actions[0] = 0.7f + Random.Range(-0.3f, 0.3f);
            actions[1] = Random.Range(-0.4f, 0.4f);
            actions[2] = Random.Range(-0.4f, 0.4f);
            actions[3] = Random.Range(-0.5f, 0.5f);
        }
    }
}