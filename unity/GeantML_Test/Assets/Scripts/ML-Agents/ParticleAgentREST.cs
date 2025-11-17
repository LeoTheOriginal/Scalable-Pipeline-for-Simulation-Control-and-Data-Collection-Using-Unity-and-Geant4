using System;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

/// <summary>
/// ML-Agents Particle Agent with REST API integration
/// NOW WITH TRAJECTORY RECORDING for batch processing
/// </summary>
[RequireComponent(typeof(TrailRenderer))]
public class ParticleAgentREST : Agent
{
    [SerializeField] private bool serverInitialized = false;
    [SerializeField] private bool serverInitializing = false;

    [Header("Agent Configuration")]
    [Tooltip("Unique agent ID")]
    public int agentId = 0;

    [Tooltip("Maximum steps per episode")]
    public int maxEpisodeSteps = 1000;

    [Header("Particle Physics")]
    [Tooltip("Initial particle energy range (MeV)")]
    public Vector2 energyRange = new Vector2(1f, 20f);

    [SerializeField] private float initialEnergy;
    [SerializeField] private float currentEnergy;
    [SerializeField] private Vector3 particleDirection;

    [Tooltip("Energy loss per cm traveled")]
    public float energyLossPerCm = 0.05f;

    [Header("Movement")]
    [Tooltip("Movement speed (cm per step)")]
    public float moveSpeed = 0.1f;

    [Header("Environment")]
    [Tooltip("Water phantom size (cm)")]
    public Vector3 phantomSize = new Vector3(5f, 5f, 5f);

    [Tooltip("Water phantom center")]
    public Vector3 phantomCenter = Vector3.zero;

    [Header("References")]
    [Tooltip("REST client for server communication")]
    public RestClient restClient;

    [Header("Training Mode")]
    [Tooltip("Per-step mode (real-time) or Episode mode (batch)")]
    public bool usePerStepMode = false;

    [Header("Visualization")]
    public Color trailColor = Color.cyan;
    public bool showDebugInfo = true;

    // Private variables
    private Vector3 startPosition = new Vector3(-6f, 0f, 0f);
    private int episodeStepCount = 0;
    private float totalEpisodeReward = 0f;
    private bool waitingForServerResponse = false;
    private TrailRenderer trail;

    // Trajectory recording (for batch mode)
    private List<StepData> recordedSteps = new List<StepData>();
    private InitialConditions episodeInitialConditions;

    // Events
    public event Action<TrajectoryData> OnTrajectoryCompleted;

    // Geant4 ground truth (for visualization)
    private Vector3 geant4Position;
    private float geant4Energy;

    public override void Initialize()
    {
        base.Initialize();

        // Find REST client if not assigned
        if (restClient == null)
        {
            // najpierw spróbuj z tego samego obiektu
            restClient = GetComponent<RestClient>();

            // jak nie ma – szukaj w scenie
            if (restClient == null)
            {
                restClient = FindObjectOfType<RestClient>();
            }

            if (restClient == null)
            {
                Debug.LogError($"[ParticleAgent {agentId}] ❌ RestClient not found in scene!");
                return;
            }
        }


        // Setup trail renderer
        trail = GetComponent<TrailRenderer>();
        if (trail != null)
        {
            trail.time = 10f;
            trail.startWidth = 0.05f;
            trail.endWidth = 0.01f;
            trail.material = new Material(Shader.Find("Sprites/Default"));
            trail.startColor = trailColor;
            trail.endColor = trailColor * 0.5f;
        }

        Debug.Log($"[ParticleAgent {agentId}] ✅ Initialized (Mode: {(usePerStepMode ? "Per-Step" : "Episode Batch")})");
    }

    public override void OnEpisodeBegin()
    {
        // Reset lokalny
        transform.position = startPosition;

        initialEnergy = UnityEngine.Random.Range(energyRange.x, energyRange.y);
        currentEnergy = initialEnergy;
        particleDirection = UnityEngine.Random.onUnitSphere;

        episodeStepCount = 0;
        totalEpisodeReward = 0f;
        waitingForServerResponse = false;

        recordedSteps.Clear();

        episodeInitialConditions = new InitialConditions
        {
            particleType = "e-",
            initialEnergy = initialEnergy,
            initialPosition = new float[] { startPosition.x, startPosition.y, startPosition.z },
            initialDirection = new float[] { particleDirection.x, particleDirection.y, particleDirection.z }
        };

        if (trail != null)
            trail.Clear();

        // 🔑 flagi serwera
        serverInitialized = false;
        serverInitializing = false;

        if (usePerStepMode)
        {
            serverInitializing = true;
            StartCoroutine(restClient.InitializeAgent(
                agentId,
                "e-",
                initialEnergy,
                startPosition,
                particleDirection,
                success =>
                {
                    serverInitializing = false;
                    serverInitialized = success;

                    if (!success)
                    {
                        Debug.LogError($"[ParticleAgent {agentId}] Failed to initialize on server!");
                    }
                }
            ));
        }

        if (showDebugInfo)
        {
            Debug.Log($"[ParticleAgent {agentId}] 🔄 Episode start: " +
                      $"Energy={initialEnergy:F2} MeV, Dir={particleDirection}");
        }
    }


    public override void CollectObservations(VectorSensor sensor)
    {
        // Position relative to start (3 values)
        Vector3 relativePos = transform.position - startPosition;
        sensor.AddObservation(relativePos.x);
        sensor.AddObservation(relativePos.y);
        sensor.AddObservation(relativePos.z);

        // Velocity = direction * energy_fraction (3 values)
        float energyFraction = currentEnergy / Mathf.Max(initialEnergy, 0.01f);
        Vector3 velocity = particleDirection * energyFraction;
        sensor.AddObservation(velocity.x);
        sensor.AddObservation(velocity.y);
        sensor.AddObservation(velocity.z);

        // Current energy normalized (1 value)
        sensor.AddObservation(energyFraction);

        // Direction (3 values)
        sensor.AddObservation(particleDirection.x);
        sensor.AddObservation(particleDirection.y);
        sensor.AddObservation(particleDirection.z);

        // Total: 10 observations
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Don't act if waiting for server response (per-step mode only)
        if (usePerStepMode)
        {
            if (serverInitializing || !serverInitialized)
                return;

            if (waitingForServerResponse)
                return;
        }

        // Get continuous actions [moveX, moveY, moveZ]
        float moveX = actions.ContinuousActions[0];
        float moveY = actions.ContinuousActions[1];
        float moveZ = actions.ContinuousActions[2];

        // Create movement vector
        Vector3 movement = new Vector3(moveX, moveY, moveZ);
        movement = Vector3.ClampMagnitude(movement, 1f) * moveSpeed;

        // Apply movement
        Vector3 previousPosition = transform.position;
        transform.position += movement;

        // Update direction (if moving)
        if (movement.magnitude > 0.001f)
        {
            particleDirection = movement.normalized;
        }

        // Energy loss proportional to distance traveled
        float distanceTraveled = movement.magnitude;
        float energyLoss = energyLossPerCm * distanceTraveled;
        currentEnergy -= energyLoss;
        currentEnergy = Mathf.Max(currentEnergy, 0f);

        // Record step (for batch mode)
        RecordStep(previousPosition, transform.position, energyLoss);

        // Increment step counter
        episodeStepCount++;

        // === MODE SELECTION ===
        if (usePerStepMode)
        {
            // PER-STEP MODE: cała logika reward/termination jest po stronie serwera
            SendStepToServer(energyLoss);
        }
        else
        {
            // EPISODE BATCH MODE: lokalny living penalty + lokalne termination
            AddReward(-0.001f);  // Small living penalty
            CheckLocalTermination();
        }
    }


    private void RecordStep(Vector3 prevPos, Vector3 newPos, float energyLoss)
    {
        StepData step = new StepData
        {
            stepNumber = episodeStepCount,
            position = new float[] { newPos.x, newPos.y, newPos.z },
            direction = new float[] { particleDirection.x, particleDirection.y, particleDirection.z },
            energy = currentEnergy,
            energyDeposited = energyLoss
        };

        recordedSteps.Add(step);
    }

    private void SendStepToServer(float energyLoss)
    {
        waitingForServerResponse = true;
        StartCoroutine(restClient.SendStep(
            agentId,
            transform.position,
            particleDirection,
            currentEnergy,
            energyLoss,
            OnServerResponse
        ));

        //AddReward(-0.001f);  // Small living penalty
        CheckLocalTermination();
    }

    private void OnServerResponse(StepResponse response)
    {
        waitingForServerResponse = false;

        if (!response.success)
        {
            Debug.LogError($"[ParticleAgent {agentId}] Server error, ending episode");
            EndEpisode();
            return;
        }

        // Apply step reward
        AddReward(response.reward);
        totalEpisodeReward += response.reward;

        // Store Geant4 ground truth for visualization
        if (response.geant4_state != null && response.geant4_state.position != null)
        {
            geant4Position = new Vector3(
                response.geant4_state.position[0],
                response.geant4_state.position[1],
                response.geant4_state.position[2]
            );
            geant4Energy = response.geant4_state.energy;
        }

        // Debug logging
        if (showDebugInfo && episodeStepCount % 20 == 0)
        {
            Debug.Log($"[ParticleAgent {agentId}] Step {episodeStepCount}: " +
                     $"Reward={response.reward:F4}, " +
                     $"PosErr={response.metrics.position_error:F3}cm, " +
                     $"Latency={response.processing_time_ms:F1}ms");
        }

        // Check if Geant4 says episode is done
        if (response.episode_done)
        {
            if (showDebugInfo)
            {
                Debug.Log($"[ParticleAgent {agentId}] Episode done by Geant4: " +
                         $"{response.termination_reason}, TotalReward={totalEpisodeReward:F2}");
            }

            if (response.termination_reason == "particle_stopped")
            {
                AddReward(5.0f);
            }

            EndEpisode();
        }
    }

    private void CheckLocalTermination()
    {
        // Energy depleted
        if (currentEnergy < 0.01f)
        {
            if (showDebugInfo)
            {
                Debug.Log($"[ParticleAgent {agentId}] Termination: energy depleted");
            }
            AddReward(1.0f);

            // In batch mode, submit trajectory
            if (!usePerStepMode)
            {
                SubmitTrajectoryToBatch();
            }

            EndEpisode();
            return;
        }

        // Exited phantom
        Vector3 relativePos = transform.position - phantomCenter;
        if (Mathf.Abs(relativePos.x) > phantomSize.x / 2f ||
            Mathf.Abs(relativePos.y) > phantomSize.y / 2f ||
            Mathf.Abs(relativePos.z) > phantomSize.z / 2f)
        {
            if (showDebugInfo)
            {
                Debug.Log($"[ParticleAgent {agentId}] Termination: exited phantom");
            }
            AddReward(-5.0f);

            if (!usePerStepMode)
            {
                SubmitTrajectoryToBatch();
            }

            EndEpisode();
            return;
        }

        // Max steps
        if (episodeStepCount >= maxEpisodeSteps)
        {
            if (showDebugInfo)
            {
                Debug.Log($"[ParticleAgent {agentId}] Termination: max steps");
            }

            if (!usePerStepMode)
            {
                SubmitTrajectoryToBatch();
            }

            EndEpisode();
        }
    }

    private void SubmitTrajectoryToBatch()
    {
        TrajectoryData trajectory = new TrajectoryData
        {
            agentId = agentId,
            initialConditions = episodeInitialConditions,
            steps = recordedSteps
        };

        // Trigger event for TrainingCoordinator
        OnTrajectoryCompleted?.Invoke(trajectory);

        if (showDebugInfo)
        {
            Debug.Log($"[ParticleAgent {agentId}] 📦 Trajectory submitted: {recordedSteps.Count} steps");
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;

        continuousActions[0] = Input.GetAxis("Horizontal");
        continuousActions[2] = Input.GetAxis("Vertical");

        if (Input.GetKey(KeyCode.Space))
            continuousActions[1] = 1f;
        else if (Input.GetKey(KeyCode.LeftShift))
            continuousActions[1] = -1f;
        else
            continuousActions[1] = 0f;
    }

    private void OnDrawGizmos()
    {
        Gizmos.color = Color.blue;
        Gizmos.DrawWireCube(phantomCenter, phantomSize);

        Gizmos.color = Color.yellow;
        Gizmos.DrawRay(transform.position, particleDirection * 0.5f);

        if (geant4Position != Vector3.zero)
        {
            Gizmos.color = Color.red;
            Gizmos.DrawWireSphere(geant4Position, 0.1f);
            Gizmos.DrawLine(transform.position, geant4Position);
        }
    }
}