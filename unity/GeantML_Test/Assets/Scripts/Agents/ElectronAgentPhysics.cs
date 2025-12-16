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
    public enum TrainingMode
    {
        PhysicsBased,
        Geant4Statistical,
        Inference
    }

    /// <summary>
    /// VERSION 6: Angular Diversity + Progressive Boundaries
    /// 
    /// KEY INSIGHT from Geant4:
    /// - Electrons scatter in ALL directions (full 360° coverage)
    /// - Scattering creates OUTWARD curving arcs (not just random)
    /// - Deep penetration + wide spread = "dandelion" shape
    /// 
    /// FIXES:
    /// - Reward for angular diversity (prevent mode collapse)
    /// - Progressive boundary penalty (not hard cutoff)
    /// - Encourage exploration of ALL directions
    /// - Higher entropy to prevent premature convergence
    /// </summary>
    public class ElectronAgentPhysics : Agent
    {
        // ====================================================================
        // INSPECTOR SETTINGS
        // ====================================================================

        [Header("Training Configuration")]
        public TrainingMode Mode = TrainingMode.Geant4Statistical;
        public int AgentIndex = 0;

        [Header("Simulation Settings")]
        public int MaxSteps = 500;
        public bool ShowVisualization = true;

        [Header("Physics Constraints")]
        public float MaxStepSize = 0.03f;
        public float MinStepSize = 0.005f;
        public float MaxDirectionChange = 0.5f;

        [Header("=== Scattering Reward Weights ===")]
        public float W_ScatteringBounds = 25f;
        public float W_ScatteringVariance = 40f;
        public float W_AntiSpiral = 35f;
        public float MinScatteringStdDev = 2.5f;
        public float TargetMeanScattering = 5f;
        public float W_MeanScattering = 20f;

        [Header("=== Lateral Spread Rewards ===")]
        public float W_LateralSpread = 40f;
        public float TargetLateralSpread = 0.35f;

        [Header("=== Energy/Path Rewards ===")]
        public float W_EnergyDepletion = 20f;
        public float W_PathLength = 15f;
        public float SurvivalBonus = 0.01f;
        public float BoundaryExitPenalty = 30f;

        [Header("=== Geant4 Statistical Comparison ===")]
        public float W_Geant4Match = 50f;
        [Range(0.1f, 0.5f)]
        public float StatisticalTolerance = 0.30f;

        [Header("=== Exploration Bonuses ===")]
        public float W_ExplorationBonus = 10f;
        public float W_StraightLinePenalty = 30f;
        public float MinAnglePerStep = 1.0f;

        [Header("=== Forward Progress (Critical!) ===")]
        public float W_ForwardProgress = 100f;
        public float W_BackscatterPenalty = 25f;
        public float MaxScatterAnglePerStep = 20f;

        [Header("=== Initial Straight Section ===")]
        public float W_InitialStraight = 80f;
        public float InitialStraightEnergyThreshold = 0.75f;

        [Header("=== V6: Angular Diversity (Prevent Mode Collapse!) ===")]
        [Tooltip("Weight for exploring all angular directions")]
        public float W_AngularDiversity = 60f;

        [Tooltip("Number of angular sectors to track (8 = 45° each)")]
        public int AngularSectors = 8;

        [Tooltip("Bonus for hitting unexplored angular sector")]
        public float UnexploredSectorBonus = 15f;

        [Header("=== V6: Progressive Boundary Penalty ===")]
        [Tooltip("Soft boundary starts at this fraction of phantom size")]
        public float SoftBoundaryStart = 0.8f;

        [Tooltip("Maximum penalty at phantom edge (not death)")]
        public float MaxBoundaryPenalty = 5f;

        [Header("=== V6: Outward Arc Reward ===")]
        [Tooltip("Reward for scattering AWAY from center axis")]
        public float W_OutwardScatter = 30f;

        [Header("Debug")]
        public bool VerboseLogging = false;
        public int LogInterval = 50;

        // ====================================================================
        // EVENTS
        // ====================================================================

        public event Action<Vector3> OnStepTaken;
        public event Action OnEpisodeReset;

        // ====================================================================
        // PRIVATE STATE
        // ====================================================================

        private Vector3 _position;
        private Vector3 _momentumDirection;
        private float _energy;

        private Vector3 _previousPosition;
        private Vector3 _previousDirection;
        private float _previousEnergy;

        private Vector3 _initialPosition;
        private Vector3 _initialDirection;
        private float _initialEnergy;
        private float _expectedCSDARange;

        private int _currentStep;
        private float _cumulativePathLength;
        private float _totalEnergyDeposited;
        private List<Vector3> _trajectoryPositions;
        private List<float> _trajectoryEnergies;

        private List<float> _scatteringAngles;
        private List<Vector3> _scatteringAxes;
        private Vector3 _cumulativeAngularMomentum;

        private List<float> _lateralPositionsY;
        private List<float> _lateralPositionsZ;

        private List<Vector3> _recentDirections;
        private const int DIRECTION_HISTORY_SIZE = 20;

        // V6: Angular sector tracking
        private int[] _angularSectorHits;
        private float _totalAngularCoverage;

        private bool _exitedBoundary;
        private float _episodeRewardSum;

        private float[] _geant4Buffer;
        private int _geant4TrajectoryLength;
        private float _geant4PathLength;
        private float _geant4FinalDepth;
        private float _geant4LateralSpread;
        private float _geant4MeanScatterAngle;
        private float _geant4ScatterStdDev;
        private bool _geant4DataValid;

        private int _totalEpisodes;
        private int _boundaryExitCount;
        private int _goodScatteringCount;
        private int _straightLineCount;

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
            _recentDirections = new List<Vector3>(DIRECTION_HISTORY_SIZE);
            _geant4Buffer = new float[MaxSteps * 7];
            _angularSectorHits = new int[AngularSectors];

            Debug.Log($"[ElectronAgent #{AgentIndex}] VERSION 6 - Angular Diversity + Progressive Boundaries!");
            Debug.Log($"  Angular Sectors: {AngularSectors}");
            Debug.Log($"  W_AngularDiversity: {W_AngularDiversity}");
            Debug.Log($"  W_OutwardScatter: {W_OutwardScatter}");
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
            _totalAngularCoverage = 0f;

            _trajectoryPositions.Clear();
            _trajectoryEnergies.Clear();
            _scatteringAngles.Clear();
            _scatteringAxes.Clear();
            _lateralPositionsY.Clear();
            _lateralPositionsZ.Clear();
            _recentDirections.Clear();

            // Reset angular sector tracking
            for (int i = 0; i < AngularSectors; i++)
            {
                _angularSectorHits[i] = 0;
            }

            _initialPosition = ElectronPhysics.GetInitialPosition();
            _initialDirection = ElectronPhysics.GetInitialDirection();
            _initialEnergy = ElectronPhysics.INITIAL_ENERGY;
            _expectedCSDARange = ElectronPhysics.CalculateCSDARange(_initialEnergy);

            _position = _initialPosition;
            _momentumDirection = _initialDirection;
            _energy = _initialEnergy;

            _previousPosition = _position;
            _previousDirection = _momentumDirection;
            _previousEnergy = _energy;

            if (Mode == TrainingMode.Geant4Statistical)
            {
                FetchGeant4Reference();
            }

            _trajectoryPositions.Add(_position);
            _trajectoryEnergies.Add(_energy);

            OnEpisodeReset?.Invoke();

            if (ShowVisualization)
            {
                transform.localPosition = _position;
            }

            _totalEpisodes++;
        }

        private void FetchGeant4Reference()
        {
            try
            {
                _geant4TrajectoryLength = Geant4Interface.RunSimulationBatch(_geant4Buffer, MaxSteps);

                if (_geant4TrajectoryLength >= 2)
                {
                    _geant4PathLength = CalculateGeant4PathLength();
                    _geant4FinalDepth = GetGeant4FinalDepth();
                    _geant4LateralSpread = GetGeant4LateralSpread();
                    CalculateGeant4ScatteringStats();
                    _geant4DataValid = true;
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[Agent #{AgentIndex}] Geant4 fetch failed: {e.Message}");
                _geant4DataValid = false;
            }
        }

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
                    g4Angles.Add(Vector3.Angle(dir1, dir2));
                }
            }

            if (g4Angles.Count > 0)
            {
                float sum = 0f;
                foreach (float a in g4Angles) sum += a;
                _geant4MeanScatterAngle = sum / g4Angles.Count;

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
                _geant4MeanScatterAngle = TargetMeanScattering;
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

        // ====================================================================
        // OBSERVATIONS (11 values - unchanged!)
        // ====================================================================

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
            sensor.AddObservation(_energy / _initialEnergy);

            // Step progress (1)
            sensor.AddObservation((float)_currentStep / MaxSteps);

            // Recent scattering statistics (2)
            float recentMean = 0f;
            float recentVariance = 0f;
            int lookback = Mathf.Min(10, _scatteringAngles.Count);

            if (lookback > 0)
            {
                for (int i = _scatteringAngles.Count - lookback; i < _scatteringAngles.Count; i++)
                {
                    recentMean += _scatteringAngles[i];
                }
                recentMean /= lookback;

                for (int i = _scatteringAngles.Count - lookback; i < _scatteringAngles.Count; i++)
                {
                    float diff = _scatteringAngles[i] - recentMean;
                    recentVariance += diff * diff;
                }
                recentVariance /= lookback;
            }

            sensor.AddObservation(recentMean / 30f);
            sensor.AddObservation(Mathf.Sqrt(recentVariance) / 15f);

            // Spiral indicator (1)
            float spiralMagnitude = _cumulativeAngularMomentum.magnitude / Mathf.Max(1f, _currentStep);
            sensor.AddObservation(Mathf.Clamp01(spiralMagnitude * 10f));
        }

        // ====================================================================
        // ACTIONS - V6 with Progressive Boundaries
        // ====================================================================

        public override void OnActionReceived(ActionBuffers actions)
        {
            if (ShouldEndEpisode())
            {
                ProcessEpisodeEnd();
                return;
            }

            var act = actions.ContinuousActions;

            Vector3 directionDelta = new Vector3(act[0], act[1], act[2]);
            float stepSizeFactor = (act[3] + 1f) / 2f;

            ApplyPureAgentAction(directionDelta, stepSizeFactor);

            // ================================================================
            // V6: PROGRESSIVE BOUNDARY PENALTIES (not death!)
            // ================================================================
            float boundaryPenalty = CalculateProgressiveBoundaryPenalty();
            if (boundaryPenalty > 0f)
            {
                AddReward(-boundaryPenalty);
            }

            // Only end episode if going BACKWARD out of phantom
            if (_position.x < ElectronPhysics.PHANTOM_ENTRY_X - 0.5f)
            {
                _exitedBoundary = true;
                ProcessEpisodeEnd();
                return;
            }

            float reward = CalculateStepReward();
            AddReward(reward);
            _episodeRewardSum += reward;

            if (ShowVisualization)
            {
                transform.localPosition = _position;
            }

            OnStepTaken?.Invoke(transform.position);

            if (VerboseLogging && _currentStep % LogInterval == 0)
            {
                LogStepInfo(reward);
            }

            _currentStep++;
        }

        /// <summary>
        /// V6: Progressive boundary penalty instead of hard cutoff.
        /// Penalty increases as agent approaches edge, but doesn't end episode.
        /// </summary>
        private float CalculateProgressiveBoundaryPenalty()
        {
            float penalty = 0f;
            float phantomSize = ElectronPhysics.PHANTOM_HALF_SIZE;
            float softBoundary = phantomSize * SoftBoundaryStart;

            // Check Y boundary
            float yAbs = Mathf.Abs(_position.y);
            if (yAbs > softBoundary)
            {
                float overshoot = (yAbs - softBoundary) / (phantomSize - softBoundary);
                penalty += MaxBoundaryPenalty * overshoot * overshoot;  // Quadratic increase
            }

            // Check Z boundary
            float zAbs = Mathf.Abs(_position.z);
            if (zAbs > softBoundary)
            {
                float overshoot = (zAbs - softBoundary) / (phantomSize - softBoundary);
                penalty += MaxBoundaryPenalty * overshoot * overshoot;
            }

            // Check X forward boundary (past phantom exit)
            float xMax = ElectronPhysics.PHANTOM_ENTRY_X + 2f * phantomSize;
            if (_position.x > xMax)
            {
                float overshoot = (_position.x - xMax) / phantomSize;
                penalty += MaxBoundaryPenalty * overshoot;
            }

            return penalty;
        }

        private void ApplyPureAgentAction(Vector3 directionDelta, float stepSizeFactor)
        {
            _previousPosition = _position;
            _previousDirection = _momentumDirection;
            _previousEnergy = _energy;

            float stepSize = Mathf.Lerp(MinStepSize, MaxStepSize, stepSizeFactor);

            Vector3 scaledDelta = directionDelta * MaxDirectionChange;
            Vector3 proposedDirection = (_momentumDirection + scaledDelta).normalized;

            if (proposedDirection.magnitude < 0.001f)
            {
                proposedDirection = _momentumDirection;
            }

            float proposedAngle = Vector3.Angle(_momentumDirection, proposedDirection);

            Vector3 newDirection;
            if (proposedAngle > MaxScatterAnglePerStep)
            {
                float clampedAngle = MaxScatterAnglePerStep;
                Vector3 rotationAxis = Vector3.Cross(_momentumDirection, proposedDirection).normalized;

                if (rotationAxis.magnitude < 0.001f)
                {
                    rotationAxis = Vector3.Cross(_momentumDirection, Vector3.up).normalized;
                    if (rotationAxis.magnitude < 0.001f)
                    {
                        rotationAxis = Vector3.Cross(_momentumDirection, Vector3.right).normalized;
                    }
                }

                Quaternion rotation = Quaternion.AngleAxis(clampedAngle, rotationAxis);
                newDirection = (rotation * _momentumDirection).normalized;
            }
            else
            {
                newDirection = proposedDirection;
            }

            float scatterAngle = Vector3.Angle(_momentumDirection, newDirection);
            _scatteringAngles.Add(scatterAngle);

            Vector3 scatterAxis = Vector3.Cross(_momentumDirection, newDirection);
            _scatteringAxes.Add(scatterAxis);
            _cumulativeAngularMomentum += scatterAxis;

            // V6: Track angular sector
            UpdateAngularSector(newDirection);

            _recentDirections.Add(newDirection);
            if (_recentDirections.Count > DIRECTION_HISTORY_SIZE)
            {
                _recentDirections.RemoveAt(0);
            }

            _momentumDirection = newDirection;

            Vector3 deltaPos = _momentumDirection * stepSize;
            _position += deltaPos;
            _cumulativePathLength += stepSize;

            float energyLoss = ElectronPhysics.CalculateEnergyLoss(_energy, stepSize);
            float fluctuation = Random.Range(0.85f, 1.15f);
            energyLoss *= fluctuation;

            _energy -= energyLoss;
            _energy = Mathf.Max(0f, _energy);
            _totalEnergyDeposited += energyLoss;

            _trajectoryPositions.Add(_position);
            _trajectoryEnergies.Add(_energy);
            _lateralPositionsY.Add(_position.y);
            _lateralPositionsZ.Add(_position.z);
        }

        /// <summary>
        /// V6: Track which angular sectors have been explored.
        /// Encourages full 360° coverage like Geant4 "dandelion".
        /// </summary>
        private void UpdateAngularSector(Vector3 direction)
        {
            // Calculate angle in Y-Z plane (perpendicular to beam axis X)
            float angle = Mathf.Atan2(direction.z, direction.y) * Mathf.Rad2Deg;
            if (angle < 0) angle += 360f;

            int sector = (int)(angle / (360f / AngularSectors)) % AngularSectors;
            _angularSectorHits[sector]++;

            // Update coverage metric
            int exploredSectors = 0;
            for (int i = 0; i < AngularSectors; i++)
            {
                if (_angularSectorHits[i] > 0) exploredSectors++;
            }
            _totalAngularCoverage = (float)exploredSectors / AngularSectors;
        }

        // ====================================================================
        // STEP REWARD - V6 with Angular Diversity
        // ====================================================================

        private float CalculateStepReward()
        {
            float reward = 0f;
            float lastScatterAngle = _scatteringAngles[_scatteringAngles.Count - 1];
            float energyFraction = _energy / _initialEnergy;

            float currentDepth = _position.x - ElectronPhysics.PHANTOM_ENTRY_X;
            float previousDepth = _previousPosition.x - ElectronPhysics.PHANTOM_ENTRY_X;
            float depthDelta = currentDepth - previousDepth;

            // ================================================================
            // PHASE 1: INITIAL PENETRATION (energy > 75%)
            // ================================================================
            if (energyFraction > InitialStraightEnergyThreshold)
            {
                if (depthDelta > 0)
                {
                    reward += W_InitialStraight * depthDelta * 35f;
                }
                else if (depthDelta < -0.001f)
                {
                    reward -= W_InitialStraight * Mathf.Abs(depthDelta) * 50f;
                }

                if (lastScatterAngle < 2f)
                {
                    reward += W_InitialStraight * 0.3f;
                }
                else if (lastScatterAngle < 4f)
                {
                    reward += W_InitialStraight * 0.15f;
                }
                else if (lastScatterAngle > 8f)
                {
                    float excess = (lastScatterAngle - 8f) / 12f;
                    reward -= W_InitialStraight * excess * 0.5f;
                }
            }
            // ================================================================
            // PHASE 2: TRANSITION (energy 40-75%)
            // ================================================================
            else if (energyFraction > 0.4f)
            {
                if (depthDelta > 0)
                {
                    reward += W_ForwardProgress * depthDelta * 8f;
                }
                else if (depthDelta < -0.002f)
                {
                    reward -= W_BackscatterPenalty * Mathf.Abs(depthDelta) * 4f;
                }

                float maxAllowedScatter = Mathf.Lerp(MaxScatterAnglePerStep, 8f, energyFraction);
                if (lastScatterAngle <= maxAllowedScatter)
                {
                    reward += W_ScatteringBounds * 0.03f;
                }

                // V6: Start rewarding angular diversity
                reward += CalculateAngularDiversityReward() * 0.3f;
            }
            // ================================================================
            // PHASE 3: DEEP SCATTERING (energy < 40%)
            // ================================================================
            else
            {
                if (depthDelta > 0)
                {
                    reward += W_ForwardProgress * depthDelta * 3f;
                }

                if (lastScatterAngle >= MinAnglePerStep && lastScatterAngle <= MaxScatterAnglePerStep)
                {
                    reward += W_ScatteringVariance * 0.03f;
                }

                // V6: Full angular diversity reward at low energy
                reward += CalculateAngularDiversityReward();

                // V6: Outward arc reward
                reward += CalculateOutwardArcReward();

                // Reward lateral spread at low energy
                float lateral = Mathf.Sqrt(_position.y * _position.y + _position.z * _position.z);
                float depthFraction = currentDepth / (2f * ElectronPhysics.PHANTOM_HALF_SIZE);
                float expectedLateral = TargetLateralSpread * depthFraction;

                if (lateral >= expectedLateral * 0.3f)
                {
                    reward += W_LateralSpread * 0.02f;
                }
            }

            // ================================================================
            // CONTINUOUS REWARDS
            // ================================================================
            if (currentDepth > 0)
            {
                float depthBonus = currentDepth / (2f * ElectronPhysics.PHANTOM_HALF_SIZE);
                reward += W_ForwardProgress * 0.005f * depthBonus;
            }

            // Anti-spiral
            if (_currentStep > 0 && _currentStep % 5 == 0 && _scatteringAxes.Count >= 5)
            {
                Vector3 recentAxisSum = Vector3.zero;
                int checkSteps = Mathf.Min(5, _scatteringAxes.Count);

                for (int i = _scatteringAxes.Count - checkSteps; i < _scatteringAxes.Count; i++)
                {
                    if (_scatteringAxes[i].magnitude > 0.001f)
                    {
                        recentAxisSum += _scatteringAxes[i].normalized;
                    }
                }

                float spiralIndicator = recentAxisSum.magnitude / checkSteps;
                if (spiralIndicator > 0.5f)
                {
                    reward -= W_AntiSpiral * (spiralIndicator - 0.5f) * 0.1f;
                }
            }

            reward += SurvivalBonus;

            return reward;
        }

        /// <summary>
        /// V6: Reward for exploring different angular sectors.
        /// Prevents mode collapse where all trajectories go same direction.
        /// </summary>
        private float CalculateAngularDiversityReward()
        {
            float reward = 0f;

            // Check if we just hit a new sector
            float angle = Mathf.Atan2(_momentumDirection.z, _momentumDirection.y) * Mathf.Rad2Deg;
            if (angle < 0) angle += 360f;
            int currentSector = (int)(angle / (360f / AngularSectors)) % AngularSectors;

            // Big bonus for first hit in any sector
            if (_angularSectorHits[currentSector] == 1)
            {
                reward += UnexploredSectorBonus;
            }

            // Continuous bonus for overall coverage
            reward += W_AngularDiversity * _totalAngularCoverage * 0.01f;

            // Penalty if too concentrated in one sector
            int maxHits = 0;
            int totalHits = 0;
            for (int i = 0; i < AngularSectors; i++)
            {
                maxHits = Mathf.Max(maxHits, _angularSectorHits[i]);
                totalHits += _angularSectorHits[i];
            }

            if (totalHits > 10)
            {
                float concentration = (float)maxHits / totalHits;
                if (concentration > 0.5f)  // More than 50% in one sector
                {
                    reward -= W_AngularDiversity * (concentration - 0.5f) * 0.1f;
                }
            }

            return reward;
        }

        /// <summary>
        /// V6: Reward for scattering OUTWARD from center axis.
        /// Creates the characteristic Geant4 "arc" shape.
        /// </summary>
        private float CalculateOutwardArcReward()
        {
            float reward = 0f;

            // Current lateral position (distance from X axis)
            float currentLateral = Mathf.Sqrt(_position.y * _position.y + _position.z * _position.z);
            float previousLateral = Mathf.Sqrt(_previousPosition.y * _previousPosition.y +
                                               _previousPosition.z * _previousPosition.z);

            float lateralDelta = currentLateral - previousLateral;

            // Reward moving OUTWARD at low energy
            float energyFraction = _energy / _initialEnergy;
            if (energyFraction < 0.5f && lateralDelta > 0)
            {
                reward += W_OutwardScatter * lateralDelta * 5f;
            }

            // Check if scattering direction is outward (away from center)
            Vector2 lateralPos = new Vector2(_position.y, _position.z);
            Vector2 lateralDir = new Vector2(_momentumDirection.y, _momentumDirection.z);

            if (lateralPos.magnitude > 0.1f && lateralDir.magnitude > 0.01f)
            {
                float dotProduct = Vector2.Dot(lateralPos.normalized, lateralDir.normalized);
                if (dotProduct > 0.3f)  // Moving outward
                {
                    reward += W_OutwardScatter * dotProduct * 0.05f;
                }
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
                    geant4Reward = CalculateGeant4ComparisonReward();
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

            float remainingEnergyFraction = _energy / _initialEnergy;
            if (remainingEnergyFraction < 0.05f)
            {
                reward += W_EnergyDepletion;
            }
            else if (remainingEnergyFraction < 0.2f)
            {
                reward += W_EnergyDepletion * 0.5f;
            }

            // X-axis penetration depth
            float finalDepthX = _position.x - ElectronPhysics.PHANTOM_ENTRY_X;
            float maxDepth = 2f * ElectronPhysics.PHANTOM_HALF_SIZE;
            float depthFraction = finalDepthX / maxDepth;

            if (depthFraction > 0.6f)
            {
                reward += W_ForwardProgress * depthFraction * 5f;
            }
            else if (depthFraction > 0.4f)
            {
                reward += W_ForwardProgress * depthFraction * 3f;
            }
            else if (depthFraction > 0.25f)
            {
                reward += W_ForwardProgress * depthFraction * 1.5f;
            }
            else if (depthFraction < 0.15f)
            {
                float shortness = (0.15f - depthFraction) / 0.15f;
                reward -= W_ForwardProgress * shortness * 3f;
            }

            float stepsUsedFraction = (float)_currentStep / MaxSteps;
            if (stepsUsedFraction < 0.3f && depthFraction < 0.3f)
            {
                reward -= W_ForwardProgress * 2f;
            }

            // Initial straight section quality
            if (_scatteringAngles.Count > 20)
            {
                int initialSteps = _scatteringAngles.Count / 4;
                float initialSum = 0f;
                for (int i = 0; i < initialSteps; i++)
                {
                    initialSum += _scatteringAngles[i];
                }
                float initialMeanAngle = initialSum / initialSteps;

                if (initialMeanAngle < 4f)
                {
                    reward += W_InitialStraight * 1.5f;
                }
                else if (initialMeanAngle < 6f)
                {
                    reward += W_InitialStraight * 0.8f;
                }
                else if (initialMeanAngle < 8f)
                {
                    reward += W_InitialStraight * 0.3f;
                }
                else
                {
                    float excess = (initialMeanAngle - 8f) / 10f;
                    reward -= W_InitialStraight * excess * 1.0f;
                }
            }

            // Final scattering quality
            if (_scatteringAngles.Count > 10)
            {
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

                if (stdDev >= MinScatteringStdDev)
                {
                    reward += W_ScatteringVariance * 0.5f;
                    _goodScatteringCount++;
                }
                else
                {
                    _straightLineCount++;
                }

                float spiralMagnitude = _cumulativeAngularMomentum.magnitude / _scatteringAngles.Count;
                if (spiralMagnitude < 0.3f)
                {
                    reward += W_AntiSpiral * 0.3f;
                }
                else
                {
                    reward -= W_AntiSpiral * spiralMagnitude;
                }
            }

            // Final lateral spread
            float finalLateral = Mathf.Sqrt(_position.y * _position.y + _position.z * _position.z);
            float lateralError = Mathf.Abs(finalLateral - TargetLateralSpread) / TargetLateralSpread;

            if (lateralError < 0.5f)
            {
                reward += W_LateralSpread * (1f - lateralError);
            }
            else if (finalLateral < TargetLateralSpread * 0.2f)
            {
                reward -= W_LateralSpread * 0.5f;
            }

            // ================================================================
            // V6: ANGULAR COVERAGE BONUS (critical for "dandelion" shape!)
            // ================================================================
            reward += W_AngularDiversity * _totalAngularCoverage * 2f;

            // Bonus for exploring many sectors
            int exploredSectors = 0;
            for (int i = 0; i < AngularSectors; i++)
            {
                if (_angularSectorHits[i] > 0) exploredSectors++;
            }

            if (exploredSectors >= AngularSectors * 0.75f)  // 75%+ coverage
            {
                reward += W_AngularDiversity * 1.5f;
            }
            else if (exploredSectors >= AngularSectors * 0.5f)  // 50%+ coverage
            {
                reward += W_AngularDiversity * 0.5f;
            }
            else if (exploredSectors <= AngularSectors * 0.25f)  // Poor coverage
            {
                reward -= W_AngularDiversity * 0.5f;
            }

            // Big bonus
            if (_scatteringAngles.Count > 20)
            {
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

                int initialSteps = _scatteringAngles.Count / 5;
                float initialSum = 0f;
                for (int i = 0; i < initialSteps; i++)
                {
                    initialSum += _scatteringAngles[i];
                }
                float initialMeanAngle = initialSum / initialSteps;

                bool goodVariance = stdDev >= MinScatteringStdDev;
                bool notSpiral = spiralMag < 0.3f;
                bool goodLateral = finalLateral >= TargetLateralSpread * 0.2f;
                bool goodDepth = depthFraction >= 0.3f;
                bool excellentDepth = depthFraction >= 0.5f;
                bool goodInitialStraight = initialMeanAngle < 5f;
                bool goodAngularCoverage = _totalAngularCoverage >= 0.5f;  // V6

                if (goodVariance && notSpiral && goodLateral && excellentDepth &&
                    goodInitialStraight && goodAngularCoverage)
                {
                    reward += 300f;  // V6: Even bigger bonus for full Geant4 match!
                }
                else if (goodDepth && goodInitialStraight && goodVariance && goodAngularCoverage)
                {
                    reward += 150f;
                }
                else if (goodDepth && goodInitialStraight && goodAngularCoverage)
                {
                    reward += 80f;
                }
                else if (goodDepth && goodInitialStraight)
                {
                    reward += 40f;
                }
                else if (goodDepth)
                {
                    reward += 20f;
                }
            }

            return reward;
        }

        private float CalculateGeant4ComparisonReward()
        {
            if (!_geant4DataValid) return 0f;

            float reward = 0f;
            float agentLateralSpread = Mathf.Sqrt(_position.y * _position.y + _position.z * _position.z);

            if (_geant4LateralSpread > 0.01f)
            {
                float lateralError = Mathf.Abs(agentLateralSpread - _geant4LateralSpread) / _geant4LateralSpread;

                if (lateralError < StatisticalTolerance)
                {
                    reward += W_Geant4Match * 0.4f * (1f - lateralError / StatisticalTolerance);
                }
                else if (agentLateralSpread < _geant4LateralSpread * 0.3f)
                {
                    reward -= W_Geant4Match * 0.3f;
                }
            }

            if (_scatteringAngles.Count > 10 && _geant4MeanScatterAngle > 0.1f)
            {
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

                float meanError = Mathf.Abs(agentMean - _geant4MeanScatterAngle) / _geant4MeanScatterAngle;
                if (meanError < StatisticalTolerance)
                {
                    reward += W_Geant4Match * 0.3f * (1f - meanError / StatisticalTolerance);
                }

                if (_geant4ScatterStdDev > 0.1f)
                {
                    float stdError = Mathf.Abs(agentStdDev - _geant4ScatterStdDev) / _geant4ScatterStdDev;
                    if (stdError < StatisticalTolerance * 1.5f)
                    {
                        reward += W_Geant4Match * 0.3f * (1f - stdError / (StatisticalTolerance * 1.5f));
                    }
                }
            }

            if (_geant4PathLength > 0.1f)
            {
                float pathError = Mathf.Abs(_cumulativePathLength - _geant4PathLength) / _geant4PathLength;
                if (pathError < StatisticalTolerance)
                {
                    reward += W_Geant4Match * 0.2f * (1f - pathError / StatisticalTolerance);
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
            float lastAngle = _scatteringAngles.Count > 0 ? _scatteringAngles[_scatteringAngles.Count - 1] : 0f;
            float lateral = Mathf.Sqrt(_position.y * _position.y + _position.z * _position.z);
            Debug.Log($"[Agent #{AgentIndex} Step {_currentStep}] " +
                     $"E={_energy:F2}MeV, Angle={lastAngle:F1}°, Lateral={lateral:F3}cm, " +
                     $"Coverage={_totalAngularCoverage:P0}, R={reward:F2}");
        }

        private void LogEpisodeSummary(float trajectoryReward, float geant4Reward)
        {
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
            float finalLateral = Mathf.Sqrt(_position.y * _position.y + _position.z * _position.z);

            int exploredSectors = 0;
            for (int i = 0; i < AngularSectors; i++)
            {
                if (_angularSectorHits[i] > 0) exploredSectors++;
            }

            Debug.Log($"[Agent #{AgentIndex}] Episode {CompletedEpisodes}:");
            Debug.Log($"  Scattering: mean={meanAngle:F2}°, std={stdDev:F2}°");
            Debug.Log($"  Lateral: {finalLateral:F3}cm, spiral={spiralMag:F3}");
            Debug.Log($"  Angular: {exploredSectors}/{AngularSectors} sectors ({_totalAngularCoverage:P0})");
            Debug.Log($"  Rewards: traj={trajectoryReward:F1}, g4={geant4Reward:F1}, total={_episodeRewardSum:F1}");
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
        public int GetGoodScatteringCount() => _goodScatteringCount;
        public float GetAngularCoverage() => _totalAngularCoverage;

        // ====================================================================
        // HEURISTIC
        // ====================================================================

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var actions = actionsOut.ContinuousActions;

            // Random exploration with bias toward forward motion
            actions[0] = Random.Range(-0.5f, 1f);  // Slight forward bias
            actions[1] = Random.Range(-1f, 1f);
            actions[2] = Random.Range(-1f, 1f);
            actions[3] = Random.Range(-0.5f, 0.5f);
        }

        /// <summary>
        /// Calculate the current angular sector (0 to AngularSectors-1) based on Y/Z position.
        /// </summary>
        private int GetCurrentAngularSector()
        {
            float angle = Mathf.Atan2(_position.z, _position.y) * Mathf.Rad2Deg;
            if (angle < 0) angle += 360f;
            return (int)(angle / (360f / AngularSectors)) % AngularSectors;
        }

        /// <summary>
        /// Get angular coverage as fraction (0 to 1).
        /// </summary>
        public float GetAngularCoverageFraction()
        {
            int explored = 0;
            for (int i = 0; i < AngularSectors; i++)
            {
                if (_angularSectorHits[i] > 0) explored++;
            }
            return (float)explored / AngularSectors;
        }
    }
}