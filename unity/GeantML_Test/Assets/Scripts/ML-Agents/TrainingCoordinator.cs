using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using UnityEngine;

/// <summary>
/// Coordinates training of multiple ParticleAgents
/// Implements trajectory buffering (>1000) and batch processing
/// 
/// Meeting requirements:
/// - "buforujemy wiele historii cząstki (>1000)"
/// - "odpalamy na kilku agentach równolegle"
/// </summary>
public class TrainingCoordinator : MonoBehaviour
{
    [Header("Training Configuration")]
    [Tooltip("Number of active agents")]
    public int numAgents = 16;

    [Tooltip("Trajectory buffer size (>1000)")]
    public int trajectoryBufferSize = 1000;

    [Tooltip("Auto-send to Python when buffer full")]
    public bool autoSendWhenFull = true;

    [Header("References")]
    public RestClient restClient;
    public GameObject agentPrefab;
    public Transform agentsParent;

    [Header("Status (Read-only)")]
    [SerializeField] private int activeAgents = 0;
    [SerializeField] private int completedTrajectories = 0;
    [SerializeField] private int bufferCount = 0;

    // Internal
    private List<ParticleAgentREST> agents = new List<ParticleAgentREST>();
    private Queue<TrajectoryData> trajectoryBuffer = new Queue<TrajectoryData>();
    private bool isProcessingBatch = false;

    void Start()
    {
        if (restClient == null)
        {
            restClient = FindObjectOfType<RestClient>();
        }

        StartCoroutine(InitializeTraining());
    }

    IEnumerator InitializeTraining()
    {
        Debug.Log($"[TrainingCoordinator] Initializing with {numAgents} agents");

        // Wait for REST connection
        while (!restClient.IsConnected())
        {
            yield return new WaitForSeconds(0.5f);
        }

        // Spawn agents
        SpawnAgents();

        Debug.Log($"[TrainingCoordinator] Training initialized");
    }

    void SpawnAgents()
    {
        if (agentPrefab == null)
        {
            Debug.LogError("[TrainingCoordinator] Agent prefab not assigned!");
            return;
        }

        for (int i = 0; i < numAgents; i++)
        {
            Vector3 spawnPos = new Vector3(-6f, i * 0.5f, 0f);  // Staggered start

            GameObject agentObj = Instantiate(agentPrefab, spawnPos, Quaternion.identity, agentsParent);
            agentObj.name = $"ParticleAgent_{i}";

            ParticleAgentREST agent = agentObj.GetComponent<ParticleAgentREST>();
            if (agent != null)
            {
                agent.agentId = i;
                agent.restClient = restClient;

                // Subscribe to agent events
                agent.OnTrajectoryCompleted += HandleTrajectoryCompleted;

                agents.Add(agent);
                activeAgents++;
            }
        }

        Debug.Log($"[TrainingCoordinator] Spawned {agents.Count} agents");
    }

    /// <summary>
    /// Called when an agent completes a trajectory
    /// </summary>
    void HandleTrajectoryCompleted(TrajectoryData trajectory)
    {
        // Add to buffer
        trajectoryBuffer.Enqueue(trajectory);
        bufferCount = trajectoryBuffer.Count;
        completedTrajectories++;

        Debug.Log($"[TrainingCoordinator] Trajectory received from agent {trajectory.agentId}, " +
                 $"Buffer: {bufferCount}/{trajectoryBufferSize}");

        // Check if buffer is full
        if (autoSendWhenFull && bufferCount >= trajectoryBufferSize && !isProcessingBatch)
        {
            Debug.Log($"[TrainingCoordinator] Buffer full! Sending batch to Python...");
            StartCoroutine(ProcessTrajectoryBatch());
        }
    }

    /// <summary>
    /// Send buffered trajectories to Python for Geant4 processing
    /// </summary>
    IEnumerator ProcessTrajectoryBatch()
    {
        isProcessingBatch = true;

        // Extract trajectories from buffer
        List<TrajectoryData> batch = new List<TrajectoryData>();
        while (trajectoryBuffer.Count > 0 && batch.Count < trajectoryBufferSize)
        {
            batch.Add(trajectoryBuffer.Dequeue());
        }

        bufferCount = trajectoryBuffer.Count;

        Debug.Log($"[TrainingCoordinator] Processing batch: {batch.Count} trajectories");

        // TODO: Send batch to Python REST endpoint
        // This will be implemented as a new endpoint in rest_server.py

        // For now, just log
        yield return new WaitForSeconds(0.1f);

        Debug.Log($"[TrainingCoordinator] Batch processed");

        isProcessingBatch = false;
    }

    void OnGUI()
    {
        GUILayout.BeginArea(new Rect(10, 10, 300, 200));

        GUILayout.Label($"<b>Training Coordinator</b>");
        GUILayout.Label($"Active Agents: {activeAgents}/{numAgents}");
        GUILayout.Label($"Completed Trajectories: {completedTrajectories}");
        GUILayout.Label($"Buffer: {bufferCount}/{trajectoryBufferSize} ({(float)bufferCount / trajectoryBufferSize * 100:F1}%)");
        GUILayout.Label($"Processing: {(isProcessingBatch ? "YES" : "NO")}");

        if (!isProcessingBatch && bufferCount > 0)
        {
            if (GUILayout.Button("Process Batch Now"))
            {
                StartCoroutine(ProcessTrajectoryBatch());
            }
        }

        GUILayout.EndArea();
    }
}

[System.Serializable]
public class TrajectoryData
{
    public int agentId;
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
    public float[] direction;
    public float energy;
    public float energyDeposited;
}