using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

/// <summary>
/// Coordinates training of multiple ParticleAgents
/// Implements trajectory buffering (>1000) and batch processing
/// 
/// Requirements:
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
    [SerializeField] private int batchesSent = 0;
    [SerializeField] private float lastBatchProcessingTime = 0f;

    // Internal
    private List<ParticleAgentREST> agents = new List<ParticleAgentREST>();
    private Queue<TrajectoryData> trajectoryBuffer = new Queue<TrajectoryData>();
    private bool isProcessingBatch = false;
    private int nextTrajectoryId = 0;

    // Statistics
    private float totalReward = 0f;
    private int successfulTrajectories = 0;
    private List<float> rewardHistory = new List<float>();

    void Start()
    {
        if (restClient == null)
        {
            restClient = FindObjectOfType<RestClient>();
            if (restClient == null)
            {
                Debug.LogError("[TrainingCoordinator] RestClient not found!");
                return;
            }
        }

        StartCoroutine(InitializeTraining());
    }

    IEnumerator InitializeTraining()
    {
        Debug.Log($"[TrainingCoordinator] Initializing with {numAgents} agents");
        Debug.Log($"[TrainingCoordinator] Buffer size: {trajectoryBufferSize}");

        // Wait for REST connection
        Debug.Log("[TrainingCoordinator] Waiting for server connection...");
        while (!restClient.IsConnected())
        {
            yield return new WaitForSeconds(0.5f);
        }

        Debug.Log("[TrainingCoordinator] ✅ Server connected!");

        // Spawn agents
        SpawnAgents();

        Debug.Log($"[TrainingCoordinator] ✅ Training initialized");
        Debug.Log($"[TrainingCoordinator]    Agents: {agents.Count}");
        Debug.Log($"[TrainingCoordinator]    Buffer: {trajectoryBufferSize}");
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
            // Staggered vertical positioning to avoid overlap
            Vector3 spawnPos = new Vector3(-6f, i * 0.3f, 0f);

            GameObject agentObj = Instantiate(agentPrefab, spawnPos, Quaternion.identity, agentsParent);
            agentObj.name = $"ParticleAgent_{i}";

            ParticleAgentREST agent = agentObj.GetComponent<ParticleAgentREST>();
            if (agent != null)
            {
                agent.agentId = i;

                // Subscribe to agent events
                agent.OnTrajectoryCompleted += HandleTrajectoryCompleted;

                agents.Add(agent);
                activeAgents++;
            }
            else
            {
                Debug.LogError($"[TrainingCoordinator] Agent prefab missing ParticleAgentREST component!");
                Destroy(agentObj);
            }
        }

        Debug.Log($"[TrainingCoordinator] Spawned {agents.Count} agents");
    }

    /// <summary>
    /// Called when an agent completes a trajectory
    /// </summary>
    void HandleTrajectoryCompleted(TrajectoryData trajectory)
    {
        // Assign unique trajectory ID
        trajectory.trajectoryId = nextTrajectoryId++;

        // Add to buffer
        trajectoryBuffer.Enqueue(trajectory);
        bufferCount = trajectoryBuffer.Count;
        completedTrajectories++;

        Debug.Log($"[TrainingCoordinator] Trajectory {trajectory.trajectoryId} from agent {trajectory.agentId} " +
                  $"({trajectory.steps.Count} steps), Buffer: {bufferCount}/{trajectoryBufferSize}");

        // Check if buffer is full
        if (autoSendWhenFull && bufferCount >= trajectoryBufferSize && !isProcessingBatch)
        {
            Debug.Log($"[TrainingCoordinator] 📦 Buffer full! Sending batch to Python...");
            StartCoroutine(ProcessTrajectoryBatch());
        }
    }

    /// <summary>
    /// Send buffered trajectories to Python for Geant4 processing
    /// </summary>
    IEnumerator ProcessTrajectoryBatch()
    {
        if (isProcessingBatch)
        {
            Debug.LogWarning("[TrainingCoordinator] Batch already being processed!");
            yield break;
        }

        if (trajectoryBuffer.Count == 0)
        {
            Debug.LogWarning("[TrainingCoordinator] Buffer is empty!");
            yield break;
        }

        isProcessingBatch = true;

        // Extract trajectories from buffer
        List<TrajectoryData> batch = new List<TrajectoryData>();
        int batchSize = Mathf.Min(trajectoryBufferSize, trajectoryBuffer.Count);

        for (int i = 0; i < batchSize; i++)
        {
            if (trajectoryBuffer.Count > 0)
            {
                batch.Add(trajectoryBuffer.Dequeue());
            }
        }

        bufferCount = trajectoryBuffer.Count;

        Debug.Log($"[TrainingCoordinator] 🚀 Processing batch: {batch.Count} trajectories");
        Debug.Log($"[TrainingCoordinator]    Remaining in buffer: {bufferCount}");

        float startTime = Time.realtimeSinceStartup;

        // Send batch to server
        bool responseReceived = false;
        BatchResponse batchResponse = null;

        yield return restClient.SendTrajectoryBatch(batch, (response) =>
        {
            batchResponse = response;
            responseReceived = true;
        });

        // Wait for response
        float timeout = 300f; // 5 minutes
        float elapsed = 0f;
        while (!responseReceived && elapsed < timeout)
        {
            elapsed += Time.deltaTime;
            yield return null;
        }

        if (!responseReceived || batchResponse == null)
        {
            Debug.LogError("[TrainingCoordinator] ❌ Batch processing failed or timed out!");
            isProcessingBatch = false;
            yield break;
        }

        // Process response
        lastBatchProcessingTime = Time.realtimeSinceStartup - startTime;
        batchesSent++;

        Debug.Log($"[TrainingCoordinator] ✅ Batch processed!");
        Debug.Log($"[TrainingCoordinator]    Trajectories: {batchResponse.trajectories_processed}");
        Debug.Log($"[TrainingCoordinator]    Time: {lastBatchProcessingTime:F2}s");
        Debug.Log($"[TrainingCoordinator]    Results: {batchResponse.results.Length}");

        // Process results and update statistics
        ProcessBatchResults(batchResponse);

        isProcessingBatch = false;

        // If buffer still has data, process next batch
        if (autoSendWhenFull && bufferCount >= trajectoryBufferSize)
        {
            Debug.Log("[TrainingCoordinator] Buffer full again, processing next batch...");
            yield return new WaitForSeconds(0.5f);
            StartCoroutine(ProcessTrajectoryBatch());
        }
    }

    /// <summary>
    /// Process batch results and update statistics
    /// </summary>
    void ProcessBatchResults(BatchResponse response)
    {
        if (response.results == null || response.results.Length == 0)
        {
            Debug.LogWarning("[TrainingCoordinator] No results in response!");
            return;
        }

        foreach (TrajectoryResult result in response.results)
        {
            EpisodeSummary summary = result.episode_summary;

            // Update statistics
            totalReward += summary.total_reward;
            rewardHistory.Add(summary.total_reward);

            if (summary.success)
            {
                successfulTrajectories++;
            }

            // Keep history limited
            if (rewardHistory.Count > 1000)
            {
                rewardHistory.RemoveAt(0);
            }
        }

        // Log statistics
        float avgReward = rewardHistory.Count > 0 ? rewardHistory.Average() : 0f;
        float successRate = completedTrajectories > 0
            ? (float)successfulTrajectories / completedTrajectories * 100f
            : 0f;

        Debug.Log($"[TrainingCoordinator] 📊 Statistics:");
        Debug.Log($"[TrainingCoordinator]    Avg Reward (last {rewardHistory.Count}): {avgReward:F2}");
        Debug.Log($"[TrainingCoordinator]    Success Rate: {successRate:F1}%");
        Debug.Log($"[TrainingCoordinator]    Total Completed: {completedTrajectories}");
    }

    /// <summary>
    /// Manual batch processing trigger
    /// </summary>
    public void TriggerBatchProcessing()
    {
        if (!isProcessingBatch && bufferCount > 0)
        {
            Debug.Log("[TrainingCoordinator] Manual batch trigger");
            StartCoroutine(ProcessTrajectoryBatch());
        }
        else
        {
            Debug.LogWarning("[TrainingCoordinator] Cannot trigger: " +
                           $"isProcessing={isProcessingBatch}, bufferCount={bufferCount}");
        }
    }

    void OnGUI()
    {
        GUILayout.BeginArea(new Rect(10, 10, 400, 300));
        GUILayout.BeginVertical("box");

        GUILayout.Label($"<b><size=16>Training Coordinator</size></b>");
        GUILayout.Space(5);

        GUILayout.Label($"<b>Agents:</b> {activeAgents}/{numAgents}");
        GUILayout.Label($"<b>Completed Trajectories:</b> {completedTrajectories}");
        GUILayout.Label($"<b>Buffer:</b> {bufferCount}/{trajectoryBufferSize} ({(float)bufferCount / trajectoryBufferSize * 100:F1}%)");
        GUILayout.Label($"<b>Batches Sent:</b> {batchesSent}");
        GUILayout.Label($"<b>Processing:</b> {(isProcessingBatch ? "<color=yellow>YES</color>" : "<color=green>NO</color>")}");

        GUILayout.Space(5);

        if (lastBatchProcessingTime > 0)
        {
            GUILayout.Label($"<b>Last Batch Time:</b> {lastBatchProcessingTime:F2}s");
        }

        if (rewardHistory.Count > 0)
        {
            float avgReward = rewardHistory.Average();
            float successRate = completedTrajectories > 0
                ? (float)successfulTrajectories / completedTrajectories * 100f
                : 0f;

            GUILayout.Space(5);
            GUILayout.Label($"<b>Avg Reward:</b> {avgReward:F2}");
            GUILayout.Label($"<b>Success Rate:</b> {successRate:F1}%");
        }

        GUILayout.Space(10);

        if (!isProcessingBatch && bufferCount > 0)
        {
            if (GUILayout.Button($"Process Batch Now ({bufferCount} trajectories)"))
            {
                TriggerBatchProcessing();
            }
        }

        if (isProcessingBatch)
        {
            GUILayout.Label("<color=yellow>Processing batch, please wait...</color>");
        }

        GUILayout.EndVertical();
        GUILayout.EndArea();
    }
}