using System;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

/// <summary>
/// ML-Agents Particle Agent - BATCH MODE ONLY
/// Fixed initial conditions, 6 observations (position + momentum)
/// </summary>
[RequireComponent(typeof(TrailRenderer))]
public class ParticleAgentREST : Agent
{
    [Header("Agent Configuration")]
    [Tooltip("Unique agent ID")]
    public int agentId = 0;

    [Tooltip("Maximum steps per episode")]
    public int maxEpisodeSteps = 1000;

    [Header("Particle Physics - FIXED")]
    [Tooltip("Initial particle energy (MeV) - FIXED")]
    public float initialEnergy = 10.0f;

    [SerializeField] private float currentEnergy;
    [SerializeField] private Vector3 particleDirection;
    [SerializeField] private Vector3 particleMomentum;

    [Tooltip("Energy loss per cm traveled")]
    public float energyLossPerCm = 0.05f;

    [Header("Movement")]
    [Tooltip("Movement speed (cm per step)")]
    public float moveSpeed = 0.1f;

    [Header("Environment - FIXED 10cm")]
    [Tooltip("Water phantom size (cm) - MUST BE 10x10x10")]
    public Vector3 phantomSize = new Vector3(10f, 10f, 10f);

    [Tooltip("Water phantom center")]
    public Vector3 phantomCenter = Vector3.zero;

    [Header("Visualization")]
    public Color trailColor = Color.cyan;
    public bool showDebugInfo = true;

    // Private variables
    private Vector3 startPosition = new Vector3(-6f, 0f, 0f);  // FIXED
    private Vector3 initialDirection = new Vector3(1f, 0f, 0f);  // FIXED
    private int episodeStepCount = 0;
    private float totalEpisodeReward = 0f;
    private TrailRenderer trail;

    // Trajectory recording
    private List<StepData> recordedSteps = new List<StepData>();
    private InitialConditions episodeInitialConditions;

    // Events
    public event Action<TrajectoryData> OnTrajectoryCompleted;

    public override void Initialize()
    {
        base.Initialize();

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

        Debug.Log($"[ParticleAgent {agentId}] ✅ Initialized (BATCH MODE)");
        Debug.Log($"[ParticleAgent {agentId}]    Phantom: {phantomSize} cm");
        Debug.Log($"[ParticleAgent {agentId}]    Fixed Energy: {initialEnergy} MeV");
    }

    public override void OnEpisodeBegin()
    {
        // ====================================================================
        // FIXED INITIAL CONDITIONS (synchronized with Geant4)
        // ====================================================================
        transform.position = startPosition;  // (-6, 0, 0)
        currentEnergy = initialEnergy;  // 10.0 MeV
        particleDirection = initialDirection;  // (1, 0, 0)
        particleMomentum = particleDirection * currentEnergy;

        episodeStepCount = 0;
        totalEpisodeReward = 0f;

        recordedSteps.Clear();

        // Store initial conditions
        episodeInitialConditions = new InitialConditions
        {
            particleType = "e-",
            initialEnergy = initialEnergy,
            initialPosition = new float[] { startPosition.x, startPosition.y, startPosition.z },
            initialDirection = new float[] { initialDirection.x, initialDirection.y, initialDirection.z }
        };

        if (trail != null)
            trail.Clear();

        if (showDebugInfo)
        {
            Debug.Log($"[ParticleAgent {agentId}] 🔄 Episode start");
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // ====================================================================
        // SIMPLIFIED OBSERVATIONS: 6 values (position + momentum)
        // ====================================================================

        // Position relative to start (3 values)
        Vector3 relativePos = transform.position - startPosition;
        sensor.AddObservation(relativePos.x);
        sensor.AddObservation(relativePos.y);
        sensor.AddObservation(relativePos.z);

        // Momentum (3 values)
        sensor.AddObservation(particleMomentum.x);
        sensor.AddObservation(particleMomentum.y);
        sensor.AddObservation(particleMomentum.z);

        // Total: 6 observations
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
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

        // Update momentum
        particleMomentum = particleDirection * currentEnergy;

        // Record step
        RecordStep(previousPosition, transform.position);

        // Increment step counter
        episodeStepCount++;

        // Small living penalty
        AddReward(-0.001f);

        // Check termination
        CheckLocalTermination();
    }

    private void RecordStep(Vector3 prevPos, Vector3 newPos)
    {
        StepData step = new StepData
        {
            stepNumber = episodeStepCount,
            position = new float[] { newPos.x, newPos.y, newPos.z },
            momentum = new float[] { particleMomentum.x, particleMomentum.y, particleMomentum.z },
            energy = currentEnergy
        };

        recordedSteps.Add(step);
    }

    private void CheckLocalTermination()
    {
        // Energy depleted
        if (currentEnergy < 0.01f)
        {
            if (showDebugInfo)
            {
                Debug.Log($"[ParticleAgent {agentId}] Termination: energy depleted ({episodeStepCount} steps)");
            }
            AddReward(1.0f);
            SubmitTrajectoryToBatch();
            EndEpisode();
            return;
        }

        // POPRAWKA: Definiujemy granice "Labu", a nie Fantomu.
        // Skoro start jest na -6, a fantom kończy się na +5, dajmy mu np. od -7 do +6 na X.
        // Na Y i Z też dajmy lekki margines, żeby nie ginął od razu jak muśnie krawędź.

        Vector3 relativePos = transform.position - phantomCenter;

        // Ustalmy bezpieczne granice świata (World Bounds)
        // X: Startuje z -6, leci do +5. Dajmy mu zakres +/- 7.0f
        // Y/Z: Fantom ma +/- 5.0f. Dajmy mu +/- 6.0f (żeby mógł lekko chybić i dostać karę później lub lecieć po krawędzi)

        float xLimit = 7.0f;
        float yzLimit = 6.0f; // Trochę szerzej niż fantom (który ma 5.0f pół-wymiaru)

        bool isOutsideWorld =
            Mathf.Abs(relativePos.x) > xLimit ||
            Mathf.Abs(relativePos.y) > yzLimit ||
            Mathf.Abs(relativePos.z) > yzLimit;

        if (isOutsideWorld)
        {
            if (showDebugInfo)
            {
                // Tutaj logujemy, że wyleciał poza obszar roboczy
                Debug.Log($"[ParticleAgent {agentId}] Termination: exited world bounds (Pos: {relativePos})");
            }

            // Kara za wylecenie w kosmos
            AddReward(-5.0f);
            SubmitTrajectoryToBatch();
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
            SubmitTrajectoryToBatch();
            EndEpisode();
        }
    }

    private void SubmitTrajectoryToBatch()
    {
        TrajectoryData trajectory = new TrajectoryData
        {
            agentId = agentId,
            trajectoryId = -1,  // Will be assigned by coordinator
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
    }
}

// ============================================================================
// Data structures
// ============================================================================

[System.Serializable]
public class TrajectoryData
{
    public int agentId;
    public int trajectoryId;
    public InitialConditions initialConditions;
    public List<StepData> steps;
}

[System.Serializable]
public class InitialConditions
{
    public string particleType;
    public float initialEnergy;
    public float[] initialPosition;
    public float[] initialDirection;
}

[System.Serializable]
public class StepData
{
    public int stepNumber;
    public float[] position;
    public float[] momentum;
    public float energy;
}