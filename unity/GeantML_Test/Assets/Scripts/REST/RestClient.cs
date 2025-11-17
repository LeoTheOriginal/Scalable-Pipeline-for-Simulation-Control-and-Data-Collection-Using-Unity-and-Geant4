using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// REST API Client for Unity-Python communication
/// Simple, reliable alternative to gRPC
/// </summary>
public class RestClient : MonoBehaviour
{
    [Header("Connection Settings")]
    [Tooltip("URL of Python REST server")]
    public string serverUrl = "http://localhost:5000";

    [Tooltip("Connection timeout in seconds")]
    public float connectionTimeout = 5f;

    [Header("Status (Read-only)")]
    [SerializeField] private bool isConnected = false;
    [SerializeField] private int activeAgents = 0;
    [SerializeField] private float averageLatency = 0f;

    // Statistics
    private List<float> latencyHistory = new List<float>();
    private const int MAX_LATENCY_HISTORY = 100;

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

                Debug.Log($"[RestClient]   Connected successfully!");
                Debug.Log($"[RestClient]   Status: {health.status}");
                Debug.Log($"[RestClient]   Geant4: {health.geant4_version}");
                Debug.Log($"[RestClient]   Active agents: {health.active_agents}");

                isConnected = true;
            }
            else
            {
                Debug.LogError($"[RestClient]   Connection failed!");
                Debug.LogError($"[RestClient]   Error: {request.error}");
                Debug.LogError($"[RestClient]   Make sure Python server is running!");

                isConnected = false;
            }
        }
    }

    /// <summary>
    /// Initialize agent with initial particle conditions
    /// </summary>
    public IEnumerator InitializeAgent(
        int agentId,
        string particleType,
        float initialEnergy,
        Vector3 initialPosition,
        Vector3 initialDirection,
        Action<bool> callback = null)
    {
        InitializeRequest initData = new InitializeRequest
        {
            agent_id = agentId,
            particle_type = particleType,
            initial_energy = initialEnergy,
            initial_position = new float[] { initialPosition.x, initialPosition.y, initialPosition.z },
            initial_direction = new float[] { initialDirection.x, initialDirection.y, initialDirection.z }
        };

        string json = JsonUtility.ToJson(initData);
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);

        using (UnityWebRequest request = new UnityWebRequest($"{serverUrl}/initialize", "POST"))
        {
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            request.timeout = (int)connectionTimeout;

            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                Debug.Log($"[RestClient] Agent {agentId} initialized successfully");
                activeAgents++;
                callback?.Invoke(true);
            }
            else
            {
                Debug.LogError($"[RestClient] Initialize failed for agent {agentId}: {request.error}");
                callback?.Invoke(false);
            }
        }
    }

    /// <summary>
    /// Send step to server and receive reward
    /// </summary>
    public IEnumerator SendStep(
        int agentId,
        Vector3 unityPosition,
        Vector3 unityDirection,
        float unityEnergy,
        float energyDeposited,
        Action<StepResponse> callback)
    {
        float startTime = Time.realtimeSinceStartup;

        StepRequest stepData = new StepRequest
        {
            agent_id = agentId,
            unity_position = new float[] { unityPosition.x, unityPosition.y, unityPosition.z },
            unity_direction = new float[] { unityDirection.x, unityDirection.y, unityDirection.z },
            unity_energy = unityEnergy,
            energy_deposited = energyDeposited
        };

        string json = JsonUtility.ToJson(stepData);
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);

        using (UnityWebRequest request = new UnityWebRequest($"{serverUrl}/step", "POST"))
        {
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            request.timeout = (int)connectionTimeout;

            yield return request.SendWebRequest();

            float latency = (Time.realtimeSinceStartup - startTime) * 1000f;
            UpdateLatency(latency);

            if (request.result == UnityWebRequest.Result.Success)
            {
                string responseJson = request.downloadHandler.text;
                StepResponse response = JsonUtility.FromJson<StepResponse>(responseJson);

                callback?.Invoke(response);

                if (response.episode_done)
                {
                    activeAgents = Mathf.Max(0, activeAgents - 1);
                }
            }
            else
            {
                Debug.LogError($"[RestClient] Step failed for agent {agentId}: {request.error}");

                // Return error response
                StepResponse errorResponse = new StepResponse
                {
                    success = false,
                    agent_id = agentId,
                    reward = -10f,
                    episode_done = true,
                    termination_reason = $"error: {request.error}"
                };

                callback?.Invoke(errorResponse);
            }
        }
    }

    /// <summary>
    /// Reset agent on server
    /// </summary>
    public IEnumerator ResetAgent(int agentId, Action<bool> callback = null)
    {
        ResetRequest resetData = new ResetRequest { agent_id = agentId };
        string json = JsonUtility.ToJson(resetData);
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);

        using (UnityWebRequest request = new UnityWebRequest($"{serverUrl}/reset", "POST"))
        {
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                Debug.Log($"[RestClient] Agent {agentId} reset");
                callback?.Invoke(true);
            }
            else
            {
                Debug.LogError($"[RestClient] Reset failed: {request.error}");
                callback?.Invoke(false);
            }
        }
    }

    private void UpdateLatency(float latency)
    {
        latencyHistory.Add(latency);
        if (latencyHistory.Count > MAX_LATENCY_HISTORY)
        {
            latencyHistory.RemoveAt(0);
        }

        float sum = 0f;
        foreach (float l in latencyHistory)
        {
            sum += l;
        }
        averageLatency = sum / latencyHistory.Count;
    }

    public bool IsConnected() => isConnected;
    public float GetAverageLatency() => averageLatency;
    public int GetActiveAgents() => activeAgents;
}

// ============================================================================
// Data structures for JSON serialization
// ============================================================================

[System.Serializable]
public class HealthResponse
{
    public string status;
    public int active_agents;
    public string geant4_version;
}

[System.Serializable]
public class InitializeRequest
{
    public int agent_id;
    public string particle_type;
    public float initial_energy;
    public float[] initial_position;
    public float[] initial_direction;
}

[System.Serializable]
public class StepRequest
{
    public int agent_id;
    public float[] unity_position;
    public float[] unity_direction;
    public float unity_energy;
    public float energy_deposited;
}

[System.Serializable]
public class StepResponse
{
    public bool success;
    public int agent_id;
    public float reward;
    public Geant4State geant4_state;
    public StepMetrics metrics;
    public bool episode_done;
    public string termination_reason;
    public float processing_time_ms;
}

[System.Serializable]
public class Geant4State
{
    public float[] position;
    public float[] direction;
    public float energy;
    public float energy_deposited;
    public float step_length;
    public string process_name;
}

[System.Serializable]
public class StepMetrics
{
    public float position_error;
    public float energy_error;
    public float direction_error;
}

[System.Serializable]
public class ResetRequest
{
    public int agent_id;
}