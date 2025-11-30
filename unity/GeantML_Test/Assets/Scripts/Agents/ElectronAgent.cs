using Core;
using System;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using Random = UnityEngine.Random;

namespace Agents
{
    public class ElectronAgent : Agent
    {
        [Header("Simulation Settings")]
        public int MaxSteps = 500;
        public bool ShowVisualization = true;

        [Header("Inference Mode")]
        [Tooltip("If true, runs without Geant4 (for deployment)")]
        public bool IsInferenceMode = false;

        [Header("Training Settings")]
        [Tooltip("Enable Scheduled Sampling (recommended)")]
        public bool UseScheduledSampling = true;

        [Tooltip("Episodes to reach minimum teacher forcing")]
        public int ScheduledSamplingEpisodes = 10000;

        [Tooltip("Minimum teacher forcing probability (keep some ground truth)")]
        [Range(0f, 1f)]
        public float MinTeacherForcingProb = 0.1f;

        [Header("Physics Constraints")]
        [Tooltip("Maximum step size in cm (Geant4 typical: 0.01-0.05 cm)")]
        public float MaxStepSize = 0.05f;

        [Tooltip("Phantom bounds (half-size in cm)")]
        public float PhantomHalfSize = 5.0f;

        private float[] _trajectoryBuffer;
        private int _trajectoryLength = 0;
        private int _currentStep = 0;

        private Vector3 _agentPosition;
        private Vector3 _agentMomentumDirection;
        private float _agentEnergy;

        private Vector3 _previousPosition;
        private Vector3 _previousDirection;

        private float _cumulativePathLength = 0f;

        private const float MASS_E = 0.511f;
        private const float INITIAL_ENERGY = 10.0f;

        private const float W_POS = 50.0f;
        private const float W_MOM = 10.0f;
        private const float W_ENERGY = 15.0f;
        private const float W_PHYSICS = 20.0f;
        private const float W_DIR = 5.0f;
        private const float W_STEP_SIZE = 30.0f;
        private const float W_SMOOTHNESS = 10.0f;
        private const float W_BOUNDARY = 100.0f;
        private const float W_RANGE = 25.0f;

        public override void Initialize()
        {
            _trajectoryBuffer = new float[MaxSteps * 7];
            Debug.Log($"[ElectronAgent] Initialized");
            Debug.Log($"  - Inference Mode: {IsInferenceMode}");
            Debug.Log($"  - Scheduled Sampling: {UseScheduledSampling}");
            Debug.Log($"  - Max Step Size: {MaxStepSize} cm");
            Debug.Log($"  - Phantom Half-Size: {PhantomHalfSize} cm");
        }

        public override void OnEpisodeBegin()
        {
            _currentStep = 0;
            _cumulativePathLength = 0f;

            Vector3 standardInitialPos = new Vector3(-6f, 0f, 0f);
            Vector3 standardInitialDir = new Vector3(1f, 0f, 0f);
            float standardInitialEnergy = INITIAL_ENERGY;

            if (IsInferenceMode)
            {
                _agentPosition = standardInitialPos;
                _agentMomentumDirection = standardInitialDir;
                _agentEnergy = standardInitialEnergy;
                _trajectoryLength = MaxSteps;

                Debug.Log("[ElectronAgent] Episode started in INFERENCE MODE");
                Debug.Log($"  Initial: pos={_agentPosition}, E={_agentEnergy} MeV");
            }
            else
            {
                int attempts = 0;
                do
                {
                    _trajectoryLength = Geant4Interface.RunSimulationBatch(_trajectoryBuffer, MaxSteps);
                    attempts++;
                } while (_trajectoryLength < 2 && attempts < 10);

                if (_trajectoryLength < 2)
                {
                    Debug.LogWarning("[ElectronAgent] Geant4 returned empty data - using standard initial conditions");
                    _agentPosition = standardInitialPos;
                    _agentMomentumDirection = standardInitialDir;
                    _agentEnergy = standardInitialEnergy;
                    _trajectoryLength = 50;
                }
                else
                {
                    Vector3 geant4Pos = new Vector3(
                        _trajectoryBuffer[0],
                        _trajectoryBuffer[1],
                        _trajectoryBuffer[2]
                    );
                    _agentPosition = ConvertGeant4ToUnity(geant4Pos);

                    Vector3 geant4Mom = new Vector3(
                        _trajectoryBuffer[3],
                        _trajectoryBuffer[4],
                        _trajectoryBuffer[5]
                    );
                    Vector3 unityMom = ConvertGeant4ToUnity(geant4Mom);
                    _agentMomentumDirection = (unityMom.magnitude > 0.001f) ? unityMom.normalized : new Vector3(1f, 0f, 0f);

                    _agentEnergy = _trajectoryBuffer[6];

                    ConvertBufferToUnityCoordinates();

                    Debug.Log($"[ElectronAgent] Episode started with {_trajectoryLength} steps from Geant4");
                    Debug.Log($"  Geant4 initial: ({_trajectoryBuffer[0]:F3}, {_trajectoryBuffer[1]:F3}, {_trajectoryBuffer[2]:F3})");
                    Debug.Log($"  Unity initial: ({_agentPosition.x:F3}, {_agentPosition.y:F3}, {_agentPosition.z:F3})");
                }
            }

            _previousPosition = _agentPosition;
            _previousDirection = _agentMomentumDirection;

            if (ShowVisualization)
            {
                transform.localPosition = _agentPosition;
            }
        }

        private Vector3 ConvertGeant4ToUnity(Vector3 geant4Vector)
        {
            return geant4Vector;
        }

        private void ConvertBufferToUnityCoordinates()
        {
            for (int i = 0; i < _trajectoryLength; i++)
            {
                int idx = i * 7;

                Vector3 geant4Pos = new Vector3(
                    _trajectoryBuffer[idx],
                    _trajectoryBuffer[idx + 1],
                    _trajectoryBuffer[idx + 2]
                );
                Vector3 unityPos = ConvertGeant4ToUnity(geant4Pos);
                _trajectoryBuffer[idx] = unityPos.x;
                _trajectoryBuffer[idx + 1] = unityPos.y;
                _trajectoryBuffer[idx + 2] = unityPos.z;

                Vector3 geant4Mom = new Vector3(
                    _trajectoryBuffer[idx + 3],
                    _trajectoryBuffer[idx + 4],
                    _trajectoryBuffer[idx + 5]
                );
                Vector3 unityMom = ConvertGeant4ToUnity(geant4Mom);
                _trajectoryBuffer[idx + 3] = unityMom.x;
                _trajectoryBuffer[idx + 4] = unityMom.y;
                _trajectoryBuffer[idx + 5] = unityMom.z;
            }
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            bool useGroundTruth = false;

            if (!IsInferenceMode && UseScheduledSampling)
            {
                float progress = Mathf.Clamp01((float)CompletedEpisodes / ScheduledSamplingEpisodes);
                float teacherProb = Mathf.Lerp(1.0f, MinTeacherForcingProb, progress);
                useGroundTruth = Random.value < teacherProb;

                if (CompletedEpisodes % 1000 == 0 && _currentStep == 0)
                {
                    Debug.Log($"[Episode {CompletedEpisodes}] Teacher Forcing Prob: {teacherProb:F3}");
                }
            }
            else if (!IsInferenceMode)
            {
                useGroundTruth = true;
            }

            if (useGroundTruth && _currentStep < _trajectoryLength)
            {
                int baseIdx = _currentStep * 7;
                sensor.AddObservation(_trajectoryBuffer[baseIdx]);
                sensor.AddObservation(_trajectoryBuffer[baseIdx + 1]);
                sensor.AddObservation(_trajectoryBuffer[baseIdx + 2]);

                Vector3 momentum = new Vector3(
                    _trajectoryBuffer[baseIdx + 3],
                    _trajectoryBuffer[baseIdx + 4],
                    _trajectoryBuffer[baseIdx + 5]
                );
                Vector3 direction = (momentum.magnitude > 0.001f) ? momentum.normalized : new Vector3(1f, 0f, 0f);

                sensor.AddObservation(direction.x);
                sensor.AddObservation(direction.y);
                sensor.AddObservation(direction.z);
                sensor.AddObservation(_trajectoryBuffer[baseIdx + 6]);

                _agentPosition = new Vector3(
                    _trajectoryBuffer[baseIdx],
                    _trajectoryBuffer[baseIdx + 1],
                    _trajectoryBuffer[baseIdx + 2]
                );
                _agentMomentumDirection = direction;
                _agentEnergy = _trajectoryBuffer[baseIdx + 6];
            }
            else
            {
                sensor.AddObservation(_agentPosition.x);
                sensor.AddObservation(_agentPosition.y);
                sensor.AddObservation(_agentPosition.z);
                sensor.AddObservation(_agentMomentumDirection.x);
                sensor.AddObservation(_agentMomentumDirection.y);
                sensor.AddObservation(_agentMomentumDirection.z);
                sensor.AddObservation(_agentEnergy);
            }

            // float remainingStepsNormalized = 1.0f - ((float)_currentStep / _trajectoryLength);
            // sensor.AddObservation(remainingStepsNormalized);
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            if (_currentStep >= _trajectoryLength - 1)
            {
                if (!IsInferenceMode)
                {
                    AddReward(10.0f);
                }
                EndEpisode();
                return;
            }

            var act = actions.ContinuousActions;

            Vector3 deltaPred_Pos_Normalized = new Vector3(act[0], act[1], act[2]);
            Vector3 deltaPred_Mom_Normalized = new Vector3(act[3], act[4], act[5]);
            float deltaPred_Energy_Normalized = act[6];

            Vector3 deltaPred_Pos = deltaPred_Pos_Normalized * MaxStepSize;
            float deltaPred_Energy = (deltaPred_Energy_Normalized - 1.0f) * 0.025f;

            _previousPosition = _agentPosition;
            _previousDirection = _agentMomentumDirection;

            _agentPosition += deltaPred_Pos;
            _agentEnergy += deltaPred_Energy;
            _agentEnergy = Mathf.Max(_agentEnergy, 0f);

            if (_agentEnergy <= 0.0f)
            {
                if (!IsInferenceMode)
                {
                    AddReward(5.0f);
                }
                Debug.Log($"[Step {_currentStep}] Energy depleted - episode complete");
                EndEpisode();
                return;
            }

            Vector3 directionChange = new Vector3(act[3], act[4], act[5]) * 0.03f;
            _agentMomentumDirection = (_agentMomentumDirection + directionChange).normalized;

            float stepLength = deltaPred_Pos.magnitude;
            _cumulativePathLength += stepLength;

            float reward = 0.0f;

            if (!IsInferenceMode)
            {
                reward = CalculatePhysicsInformedReward(
                    deltaPred_Pos,
                    deltaPred_Energy
                );

                SetReward(reward);

                if (_currentStep % 50 == 0)
                {
                    Debug.Log($"[Step {_currentStep}] Reward: {reward:F2}, Pos: {_agentPosition}, E: {_agentEnergy:F3} MeV");
                }
            }

            if (ShowVisualization)
            {
                transform.localPosition = _agentPosition;
            }

            _currentStep++;
        }

        private float CalculatePhysicsInformedReward(
            Vector3 deltaPred_Pos,
            float deltaPred_Energy)
        {
            float reward = 0.0f;

            int currIdx = _currentStep * 7;
            int nextIdx = (_currentStep + 1) * 7;

            Vector3 posGT_Curr = new Vector3(
                _trajectoryBuffer[currIdx],
                _trajectoryBuffer[currIdx + 1],
                _trajectoryBuffer[currIdx + 2]
            );
            Vector3 posGT_Next = new Vector3(
                _trajectoryBuffer[nextIdx],
                _trajectoryBuffer[nextIdx + 1],
                _trajectoryBuffer[nextIdx + 2]
            );
            Vector3 momGT_Next = new Vector3(
                _trajectoryBuffer[nextIdx + 3],
                _trajectoryBuffer[nextIdx + 4],
                _trajectoryBuffer[nextIdx + 5]
            );
            float energyGT_Next = _trajectoryBuffer[nextIdx + 6];

            Vector3 deltaTrue_Pos = posGT_Next - posGT_Curr;

            float posError = Vector3.Distance(deltaPred_Pos, deltaTrue_Pos);
            reward -= posError * W_POS;

            float E_total = _agentEnergy + MASS_E;
            float p_magnitude = Mathf.Sqrt(E_total * E_total - MASS_E * MASS_E);
            Vector3 agentMomentum = _agentMomentumDirection * p_magnitude;

            float momError = Vector3.Distance(agentMomentum, momGT_Next);
            reward -= momError * W_MOM;

            float energyError = Mathf.Abs(_agentEnergy - energyGT_Next);
            reward -= energyError * W_ENERGY;

            float p_sq = agentMomentum.sqrMagnitude;
            float e_physical = Mathf.Sqrt(p_sq + MASS_E * MASS_E) - MASS_E;
            float physError = Mathf.Abs(_agentEnergy - e_physical);
            reward -= physError * W_PHYSICS;

            if (deltaPred_Pos.x > 0)
                reward += W_DIR;
            else
                reward -= W_DIR * 2;

            float stepSize = deltaPred_Pos.magnitude;
            if (stepSize > MaxStepSize)
            {
                float excess = stepSize - MaxStepSize;
                reward -= excess * W_STEP_SIZE;
            }

            if (_currentStep > 0)
            {
                Vector3 prevDelta = _agentPosition - _previousPosition;
                float angleChange = Vector3.Angle(prevDelta, deltaPred_Pos);
                if (angleChange > 30f)
                {
                    reward -= (angleChange / 30f) * W_SMOOTHNESS;
                }
            }

            if (Mathf.Abs(_agentPosition.x) > PhantomHalfSize ||
                Mathf.Abs(_agentPosition.y) > PhantomHalfSize ||
                Mathf.Abs(_agentPosition.z) > PhantomHalfSize)
            {
                reward -= W_BOUNDARY;

                if (Mathf.Abs(_agentPosition.x) > PhantomHalfSize + 2f)
                {
                    Debug.LogWarning($"[Step {_currentStep}] Agent left phantom! Terminating episode.");
                    EndEpisode();
                }
            }

            if (posError < 0.01f) reward += 10.0f;
            else if (posError < 0.05f) reward += 5.0f;
            else if (posError < 0.1f) reward += 2.0f;

            if (!IsInferenceMode && _currentStep > 0)
            {
                float geant4PathLength = CalculateGeant4PathLength(_currentStep);
                float pathError = Mathf.Abs(_cumulativePathLength - geant4PathLength);

                if (pathError > geant4PathLength * 0.1f)
                {
                    reward -= pathError * W_RANGE;
                }
                else if (pathError < geant4PathLength * 0.05f)
                {
                    reward += 5.0f;
                }
            }

            return reward;
        }

        private float CalculateGeant4PathLength(int upToStep)
        {
            float totalLength = 0f;

            for (int i = 0; i < upToStep; i++)
            {
                int idx = i * 7;
                int nextIdx = (i + 1) * 7;

                Vector3 pos1 = new Vector3(
                    _trajectoryBuffer[idx],
                    _trajectoryBuffer[idx + 1],
                    _trajectoryBuffer[idx + 2]
                );
                Vector3 pos2 = new Vector3(
                    _trajectoryBuffer[nextIdx],
                    _trajectoryBuffer[nextIdx + 1],
                    _trajectoryBuffer[nextIdx + 2]
                );

                totalLength += Vector3.Distance(pos1, pos2);
            }

            return totalLength;
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var continuousActionsOut = actionsOut.ContinuousActions;

            continuousActionsOut[0] = 0.5f;
            continuousActionsOut[1] = 0.0f;
            continuousActionsOut[2] = 0.0f;
            continuousActionsOut[3] = 0.0f;
            continuousActionsOut[4] = 0.0f;
            continuousActionsOut[5] = 0.0f;
            continuousActionsOut[6] = -0.2f;
        }
    }
}