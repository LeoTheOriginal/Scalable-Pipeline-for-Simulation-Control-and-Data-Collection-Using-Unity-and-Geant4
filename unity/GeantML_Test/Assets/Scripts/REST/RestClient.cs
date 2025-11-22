using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// REST API Client for Unity-Python communication
/// BATCH MODE ONLY - sends completed trajectories in batches
/// </summary>
public class RestClient : MonoBehaviour
{
    [Header("Connection Settings")]
    [Tooltip("URL of Python REST server")]
    public string serverUrl = "http://localhost:5000";

    [Tooltip("Connection timeout in seconds")]
    public float connectionTimeout = 30f;

    [Tooltip("Batch request timeout in seconds")]
    public float batchTimeout = 300f;

    [Header("Status (Read-only)")]
    [SerializeField] private bool isConnected = false;
    [SerializeField] private string serverStatus = "Not connected";
    [SerializeField] private int totalTrajectoriesProcessed = 0;
    [SerializeField] private int totalBatchesProcessed = 0;

    void Start()
    {
        StartCoroutine(ConnectToServer());
    }

    /// <summary>
    /// Test connection to server
    /// </summary>
    public IEnumerator ConnectToServer()
    {
        Debug.Log($"[RestClient] Connecting to {serverUrl}...");

        using (UnityWebRequest request = UnityWebRequest.Get($"{serverUrl}/health"))
        {
            request.timeout = (int)connectionTimeout;

            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                string json = request.downloadHandler.text;
                HealthResponse health = JsonUtility.FromJson<HealthResponse>(json);

                Debug.Log($"[RestClient] ✅ Connected successfully!");
                Debug.Log($"[RestClient]    Status: {health.status}");
                Debug.Log($"[RestClient]    Server: {health.server}");
                Debug.Log($"[RestClient]    Geant4: {health.geant4_version}");
                Debug.Log($"[RestClient]    Workers: {health.config.num_workers}");

                isConnected = true;
                serverStatus = $"Connected - {health.server}";
                totalTrajectoriesProcessed = health.statistics.total_trajectories_processed;
                totalBatchesProcessed = health.statistics.total_batches_processed;
            }
            else
            {
                Debug.LogError($"[RestClient] ❌ Connection failed!");
                Debug.LogError($"[RestClient]    Error: {request.error}");
                Debug.LogError($"[RestClient]    Make sure Python server is running!");

                isConnected = false;
                serverStatus = $"Failed: {request.error}";
            }
        }
    }

    /// <summary>
    /// Send batch of trajectories to Python server for Geant4 processing
    /// </summary>
    public IEnumerator SendTrajectoryBatch(
        List<TrajectoryData> trajectories,
        Action<BatchResponse> callback)
    {
        if (!isConnected)
        {
            Debug.LogError("[RestClient] Not connected to server!");
            callback?.Invoke(null);
            yield break;
        }

        if (trajectories == null || trajectories.Count == 0)
        {
            Debug.LogWarning("[RestClient] Empty trajectory batch!");
            callback?.Invoke(null);
            yield break;
        }

        Debug.Log($"[RestClient] Sending batch: {trajectories.Count} trajectories");

        float startTime = Time.realtimeSinceStartup;

        // Prepare batch request
        BatchRequest batchRequest = new BatchRequest
        {
            trajectories = trajectories.ToArray()
        };

        string json = JsonUtility.ToJson(batchRequest);
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);

        Debug.Log($"[RestClient] Request size: {bodyRaw.Length / 1024f:F1} KB");

        using (UnityWebRequest request = new UnityWebRequest($"{serverUrl}/trajectory/process_batch", "POST"))
        {
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            request.timeout = (int)batchTimeout;

            yield return request.SendWebRequest();

            float requestTime = Time.realtimeSinceStartup - startTime;

            if (request.result == UnityWebRequest.Result.Success)
            {
                string responseJson = request.downloadHandler.text;
                BatchResponse response = JsonUtility.FromJson<BatchResponse>(responseJson);

                Debug.Log($"[RestClient] ✅ Batch processed successfully!");
                Debug.Log($"[RestClient]    Trajectories: {response.trajectories_processed}");
                Debug.Log($"[RestClient]    Server time: {response.processing_time_seconds:F2}s");
                Debug.Log($"[RestClient]    Total time: {requestTime:F2}s");
                Debug.Log($"[RestClient]    Results: {response.results.Length}");

                // Update statistics
                totalTrajectoriesProcessed += response.trajectories_processed;
                totalBatchesProcessed++;

                callback?.Invoke(response);
            }
            else
            {
                Debug.LogError($"[RestClient] ❌ Batch processing failed!");
                Debug.LogError($"[RestClient]    Error: {request.error}");
                Debug.LogError($"[RestClient]    Response code: {request.responseCode}");

                if (request.downloadHandler != null && !string.IsNullOrEmpty(request.downloadHandler.text))
                {
                    Debug.LogError($"[RestClient]    Server response: {request.downloadHandler.text}");
                }

                callback?.Invoke(null);
            }
        }
    }

    /// <summary>
    /// Get server configuration
    /// </summary>
    public IEnumerator GetServerConfig(Action<ServerConfig> callback)
    {
        using (UnityWebRequest request = UnityWebRequest.Get($"{serverUrl}/config"))
        {
            request.timeout = (int)connectionTimeout;

            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                string json = request.downloadHandler.text;
                ServerConfig config = JsonUtility.FromJson<ServerConfig>(json);

                Debug.Log($"[RestClient] Server config retrieved");
                callback?.Invoke(config);
            }
            else
            {
                Debug.LogError($"[RestClient] Failed to get config: {request.error}");
                callback?.Invoke(null);
            }
        }
    }

    public bool IsConnected() => isConnected;
    public int GetTotalTrajectoriesProcessed() => totalTrajectoriesProcessed;
    public int GetTotalBatchesProcessed() => totalBatchesProcessed;
}

// ============================================================================
// Data structures for JSON serialization
// ============================================================================

[System.Serializable]
public class HealthResponse
{
    public string status;
    public string server;
    public string geant4_version;
    public float server_time;
    public ServerConfigInfo config;
    public ServerStatistics statistics;
}

[System.Serializable]
public class ServerConfigInfo
{
    public int buffer_size;
    public int num_workers;
    public string geant4_executable;
}

[System.Serializable]
public class ServerStatistics
{
    public int total_trajectories_processed;
    public int total_batches_processed;
}

[System.Serializable]
public class BatchRequest
{
    public TrajectoryData[] trajectories;
}

[System.Serializable]
public class BatchResponse
{
    public bool success;
    public int trajectories_processed;
    public float processing_time_seconds;
    public TrajectoryResult[] results;
}

[System.Serializable]
public class TrajectoryResult
{
    public int trajectory_id;
    public int agent_id;
    public EpisodeSummary episode_summary;
}

[System.Serializable]
public class EpisodeSummary
{
    public float total_reward;
    public float mean_position_error;
    public float mean_momentum_error;
    public int num_steps;
    public bool success;
}

[System.Serializable]
public class ServerConfig
{
    public InitialConditionsConfig initial_conditions;
    public TrainingConfig training;
    public RewardConfig reward;
}

[System.Serializable]
public class InitialConditionsConfig
{
    public string particle_type;
    public float particle_energy;
    public float[] particle_position;
    public float[] particle_direction;
}

[System.Serializable]
public class TrainingConfig
{
    public int buffer_size;
    public int num_workers;
    public float[] phantom_size;
    public float[] phantom_center;
}

[System.Serializable]
public class RewardConfig
{
    public float position_weight;
    public float momentum_weight;
    public float completion_bonus;
    public float exit_penalty;
}