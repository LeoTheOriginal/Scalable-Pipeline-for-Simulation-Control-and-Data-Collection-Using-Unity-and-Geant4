using UnityEngine;
using Unity.MLAgents.Policies;
using Unity.Barracuda; // For NNModel
using Agents;
using System.Collections.Generic;

namespace Management
{
    /// <summary>
    /// Global manager for multi-agent training farm.
    /// 
    /// Features:
    /// - Centralized configuration for all agents
    /// - Bulk behavior type switching (Training/Inference)
    /// - Model assignment for all agents
    /// - Statistics collection across farm
    /// - Override options for individual agents
    /// 
    /// Usage:
    /// 1. Add this to scene root
    /// 2. Create agent prefabs under this GameObject
    /// 3. Configure global settings
    /// 4. Individual agents can override if needed
    /// </summary>
    public class AgentFarmManager : MonoBehaviour
    {
        // ====================================================================
        // INSPECTOR SETTINGS
        // ====================================================================

        [Header("Farm Configuration")]
        [Tooltip("Total number of agents in farm")]
        public int AgentCount = 24;

        [Tooltip("Apply global settings to all agents on Start")]
        public bool ApplyGlobalSettingsOnStart = true;

        [Header("Global Behavior Settings")]
        [Tooltip("Default behavior type for all agents")]
        public BehaviorType GlobalBehaviorType = BehaviorType.Default;

        [Tooltip("ONNX model for inference mode (leave empty for training)")]
        public NNModel GlobalInferenceModel = null;

        [Tooltip("Allow individual agents to override global settings")]
        public bool AllowIndividualOverrides = true;

        [Header("Global Training Settings")]
        [Tooltip("Training mode for all agents")]
        public TrainingMode GlobalTrainingMode = TrainingMode.Geant4Statistical;

        [Tooltip("Max steps per episode")]
        public int GlobalMaxSteps = 500;

        [Header("Global Reward Weights")]
        [Tooltip("Apply custom reward weights to all agents")]
        public bool UseCustomRewardWeights = false;

        [Range(0f, 50f)]
        public float W_ScatteringBounds = 10f;

        [Range(0f, 50f)]
        public float W_ScatteringVariance = 25f;

        [Range(0f, 50f)]
        public float W_AntiSpiral = 30f;

        [Range(0f, 50f)]
        public float W_MeanScattering = 20f;

        [Header("Visualization")]
        [Tooltip("Enable trajectory visualization for all agents")]
        public bool GlobalEnableVisualization = true;

        [Tooltip("Cumulative mode for all trajectories")]
        public bool GlobalCumulativeMode = true;

        [Header("Statistics")]
        [Tooltip("Collect statistics every N episodes")]
        public int StatisticsInterval = 100;

        [Tooltip("Log farm-wide statistics")]
        public bool LogFarmStatistics = true;

        [Header("Performance")]
        [Tooltip("Use Burst compilation (faster physics)")]
        public bool UseBurstCompilation = true;

        [Tooltip("Target FPS during training")]
        public int TargetFPS = -1; // -1 = unlimited

        // ====================================================================
        // RUNTIME STATE
        // ====================================================================

        private List<ElectronAgentPhysics> _agents;
        private List<BehaviorParameters> _behaviorParameters;
        private int _totalEpisodes;
        private int _totalSteps;
        private float _averageReward;
        private System.Diagnostics.Stopwatch _farmTimer;

        // ====================================================================
        // LIFECYCLE
        // ====================================================================

        void Awake()
        {
            _agents = new List<ElectronAgentPhysics>();
            _behaviorParameters = new List<BehaviorParameters>();
            _farmTimer = new System.Diagnostics.Stopwatch();

            // Set target FPS
            if (TargetFPS > 0)
            {
                Application.targetFrameRate = TargetFPS;
            }
            else
            {
                Application.targetFrameRate = -1; // Unlimited
            }

            // Find all agents in farm
            DiscoverAgents();
        }

        void Start()
        {
            if (ApplyGlobalSettingsOnStart)
            {
                ApplyGlobalSettings();
                AutoAssignAgentIndices();
            }

            _farmTimer.Start();

            Debug.Log($"[AgentFarmManager] Initialized with {_agents.Count} agents");
            Debug.Log($"  Behavior Type: {GlobalBehaviorType}");
            Debug.Log($"  Training Mode: {GlobalTrainingMode}");
            Debug.Log($"  Burst: {UseBurstCompilation}");
        }

        void Update()
        {
            // Statistics collection
            if (LogFarmStatistics && Time.frameCount % (StatisticsInterval * 100) == 0)
            {
                LogFarmStatistics_Internal();
            }
        }

        // ====================================================================
        // AGENT DISCOVERY
        // ====================================================================

        private void DiscoverAgents()
        {
            _agents.Clear(); // Upewnij się, że lista jest czysta
            _behaviorParameters.Clear();

            // 1. Najpierw szukamy standardowo w dzieciach (jeśli skrypt jest na samej górze)
            // Dodajemy 'true', aby znaleźć też nieaktywnych agentów (częsty błąd)
            ElectronAgentPhysics[] foundAgents = GetComponentsInChildren<ElectronAgentPhysics>(true);

            // 2. Jeśli nic nie znaleźliśmy, a mamy rodzica (przypadek z Twojego screena),
            // szukamy w całym rodzicu (TrainingFarm_v2)
            if (foundAgents.Length == 0 && transform.parent != null)
            {
                Debug.Log("[AgentFarmManager] Nie znaleziono agentów w dzieciach. Szukam w obiekcie nadrzędnym...");
                foundAgents = transform.parent.GetComponentsInChildren<ElectronAgentPhysics>(true);
            }

            // 3. Ostateczność: Znajdź wszystkich agentów tego typu na scenie (Global search)
            if (foundAgents.Length == 0)
            {
                Debug.LogWarning("[AgentFarmManager] Nadal brak agentów. Przeszukuję całą scenę (FindObjectsOfType)...");
                foundAgents = FindObjectsOfType<ElectronAgentPhysics>(true);
            }

            // Przypisanie znalezionych agentów do list
            foreach (var agent in foundAgents)
            {
                _agents.Add(agent);

                BehaviorParameters bp = agent.GetComponent<BehaviorParameters>();
                if (bp != null)
                {
                    _behaviorParameters.Add(bp);
                }
            }

            // Sortowanie po nazwie, aby indeksy (0, 1, 2...) odpowiadały kolejności środowisk (Environment_00, 01...)
            // To ważne dla spójności wizualizacji i logów!
            _agents.Sort((a, b) => string.Compare(a.transform.parent.parent.name, b.transform.parent.parent.name));

            Debug.Log($"[AgentFarmManager] Discovered {_agents.Count} agents");
        }


        // ====================================================================
        // AGENT INDEX MANAGEMENT
        // ====================================================================

        /// <summary>
        /// Automatically assign unique AgentIndex to all agents.
        /// This helps with debugging, visualization colors, and statistics.
        /// </summary>
        [ContextMenu("Auto-Assign Agent Indices")]
        public void AutoAssignAgentIndices()
        {
            if (_agents == null || _agents.Count == 0)
            {
                Debug.LogWarning("[AgentFarmManager] No agents found! Make sure agents are children of this GameObject.");
                return;
            }

            for (int i = 0; i < _agents.Count; i++)
            {
                _agents[i].AgentIndex = i;
            }

            Debug.Log($"[AgentFarmManager] Auto-assigned indices to {_agents.Count} agents");
            Debug.Log($"  Agent 0: {_agents[0].gameObject.name}");
            Debug.Log($"  Agent {_agents.Count - 1}: {_agents[_agents.Count - 1].gameObject.name}");

            #if UNITY_EDITOR
                // Mark scene as dirty so changes are saved
                UnityEditor.EditorUtility.SetDirty(gameObject);
                foreach (var agent in _agents)
                {
                    UnityEditor.EditorUtility.SetDirty(agent);
                }
            #endif
        }

        /// <summary>
        /// Verify that all agents have unique indices.
        /// </summary>
        [ContextMenu("Verify Agent Indices")]
        public void VerifyAgentIndices()
        {
            if (_agents == null || _agents.Count == 0)
            {
                Debug.LogWarning("[AgentFarmManager] No agents to verify!");
                return;
            }

            HashSet<int> usedIndices = new HashSet<int>();
            List<int> duplicates = new List<int>();

            foreach (var agent in _agents)
            {
                if (usedIndices.Contains(agent.AgentIndex))
                {
                    duplicates.Add(agent.AgentIndex);
                }
                usedIndices.Add(agent.AgentIndex);
            }

            if (duplicates.Count > 0)
            {
                Debug.LogWarning($"[AgentFarmManager] Found {duplicates.Count} duplicate indices!");
                Debug.LogWarning($"  Duplicates: {string.Join(", ", duplicates)}");
                Debug.LogWarning("  Use 'Auto-Assign Agent Indices' to fix.");
            }
            else
            {
                Debug.Log($"[AgentFarmManager] ✅ All {_agents.Count} agents have unique indices!");
                Debug.Log($"  Range: {_agents[0].AgentIndex} to {_agents[_agents.Count - 1].AgentIndex}");
            }
        }

        // ====================================================================
        // GLOBAL CONFIGURATION
        // ====================================================================

        /// <summary>
        /// Apply global settings to all agents.
        /// </summary>
        public void ApplyGlobalSettings()
        {
            int appliedCount = 0;

            for (int i = 0; i < _agents.Count; i++)
            {
                ElectronAgentPhysics agent = _agents[i];
                BehaviorParameters bp = _behaviorParameters[i];

                // Check if agent allows override
                bool canOverride = AllowIndividualOverrides && ShouldSkipAgent(agent);
                if (canOverride)
                {
                    Debug.Log($"[AgentFarmManager] Skipping Agent #{agent.AgentIndex} (individual override)");
                    continue;
                }

                // Apply behavior type
                if (bp != null)
                {
                    // 1. Najpierw przypisz model (nawet jeśli null)
                    bp.Model = GlobalInferenceModel;

                    // 2. Dopiero potem ustaw typ zachowania
                    // (Dzięki temu Unity widzi, że model już jest, zanim przełączy tryb)
                    bp.BehaviorType = GlobalBehaviorType;

                    // Zabezpieczenie: Jeśli użytkownik wybrał Inference, ale zapomniał modelu -> wymuś Default
                    if (GlobalBehaviorType == BehaviorType.InferenceOnly && GlobalInferenceModel == null)
                    {
                        Debug.LogWarning($"[AgentFarmManager] Agent #{agent.AgentIndex}: Wybrano InferenceOnly, ale brak modelu! Przełączam na Default.");
                        bp.BehaviorType = BehaviorType.Default;
                    }
                }

                // Apply training settings
                agent.Mode = GlobalTrainingMode;
                agent.MaxSteps = GlobalMaxSteps;

                // Apply reward weights
                if (UseCustomRewardWeights)
                {
                    agent.W_ScatteringBounds = W_ScatteringBounds;
                    agent.W_ScatteringVariance = W_ScatteringVariance;
                    agent.W_AntiSpiral = W_AntiSpiral;
                    agent.W_MeanScattering = W_MeanScattering;
                }

                // Apply visualization settings
                var visualizer = agent.GetComponent<Visualization.TrajectoryVisualizer>();
                if (visualizer != null)
                {
                    visualizer.EnableVisualization = GlobalEnableVisualization;
                    visualizer.CumulativeMode = GlobalCumulativeMode;
                }

                appliedCount++;
            }

            Debug.Log($"[AgentFarmManager] Applied global settings to {appliedCount}/{_agents.Count} agents");
        }

        private bool ShouldSkipAgent(ElectronAgentPhysics agent)
        {
            // Check if agent has custom tag indicating override
            return agent.CompareTag("OverrideAgent");
        }

        // ====================================================================
        // BULK OPERATIONS
        // ====================================================================

        /// <summary>
        /// Switch all agents to training mode.
        /// </summary>
        [ContextMenu("Switch All to Training")]
        public void SwitchAllToTraining()
        {
            GlobalBehaviorType = BehaviorType.Default;
            GlobalTrainingMode = TrainingMode.Geant4Statistical;
            ApplyGlobalSettings();
        }

        /// <summary>
        /// Switch all agents to inference mode.
        /// </summary>
        [ContextMenu("Switch All to Inference")]
        public void SwitchAllToInference()
        {
            if (GlobalInferenceModel == null)
            {
                Debug.LogError("[AgentFarmManager] No inference model assigned!");
                return;
            }

            GlobalBehaviorType = BehaviorType.InferenceOnly;
            GlobalTrainingMode = TrainingMode.Inference;
            ApplyGlobalSettings();

            Debug.Log("[AgentFarmManager] Switched all agents to inference mode");
        }

        /// <summary>
        /// Enable visualization for all agents.
        /// </summary>
        [ContextMenu("Enable All Visualization")]
        public void EnableAllVisualization()
        {
            GlobalEnableVisualization = true;
            foreach (var agent in _agents)
            {
                var visualizer = agent.GetComponent<Visualization.TrajectoryVisualizer>();
                if (visualizer != null)
                {
                    visualizer.SetVisualizationEnabled(true);
                }
            }
        }

        /// <summary>
        /// Disable visualization for all agents (faster training).
        /// </summary>
        [ContextMenu("Disable All Visualization")]
        public void DisableAllVisualization()
        {
            GlobalEnableVisualization = false;
            foreach (var agent in _agents)
            {
                var visualizer = agent.GetComponent<Visualization.TrajectoryVisualizer>();
                if (visualizer != null)
                {
                    visualizer.SetVisualizationEnabled(false);
                }
            }
        }

        /// <summary>
        /// Clear all trajectory visualizations.
        /// </summary>
        [ContextMenu("Clear All Trajectories")]
        public void ClearAllTrajectories()
        {
            foreach (var agent in _agents)
            {
                var visualizer = agent.GetComponent<Visualization.TrajectoryVisualizer>();
                if (visualizer != null)
                {
                    visualizer.ClearTrajectory();
                }
            }

            Debug.Log("[AgentFarmManager] Cleared all trajectories");
        }

        // ====================================================================
        // STATISTICS
        // ====================================================================

        private void LogFarmStatistics_Internal()
        {
            int activeBehaviors = 0;
            int completedEpisodes = 0;
            int boundaryExits = 0;
            int straightLines = 0;

            foreach (var agent in _agents)
            {
                // Count episodes
                if (agent.CompletedEpisodes > 0)
                {
                    activeBehaviors++;
                    completedEpisodes += agent.CompletedEpisodes;
                }

                // Count failures
                boundaryExits += agent.DidExitBoundary() ? 1 : 0;
                straightLines += agent.GetStraightLineCount();
            }

            float episodesPerAgent = (float)completedEpisodes / Mathf.Max(1, activeBehaviors);
            float failureRate = (float)(boundaryExits + straightLines) / Mathf.Max(1, completedEpisodes);

            Debug.Log($"[AgentFarmManager] Farm Statistics (after {_farmTimer.Elapsed.TotalMinutes:F1} min):");
            Debug.Log($"  Active Agents: {activeBehaviors}/{_agents.Count}");
            Debug.Log($"  Total Episodes: {completedEpisodes}");
            Debug.Log($"  Episodes/Agent: {episodesPerAgent:F1}");
            Debug.Log($"  Failure Rate: {failureRate:P1}");
            Debug.Log($"  Boundary Exits: {boundaryExits}");
            Debug.Log($"  Straight Lines: {straightLines}");
        }

        /// <summary>
        /// Get farm-wide statistics.
        /// </summary>
        public FarmStatistics GetStatistics()
        {
            FarmStatistics stats = new FarmStatistics();
            stats.AgentCount = _agents.Count;
            stats.ElapsedTime = (float)_farmTimer.Elapsed.TotalSeconds;

            foreach (var agent in _agents)
            {
                stats.TotalEpisodes += agent.CompletedEpisodes;
                stats.TotalBoundaryExits += agent.DidExitBoundary() ? 1 : 0;
                stats.TotalStraightLines += agent.GetStraightLineCount();
            }

            stats.EpisodesPerAgent = (float)stats.TotalEpisodes / Mathf.Max(1, _agents.Count);
            stats.FailureRate = (float)(stats.TotalBoundaryExits + stats.TotalStraightLines) /
                              Mathf.Max(1, stats.TotalEpisodes);

            return stats;
        }

        // ====================================================================
        // PUBLIC ACCESSORS
        // ====================================================================

        public List<ElectronAgentPhysics> GetAllAgents() => _agents;
        public int GetActiveAgentCount() => _agents.FindAll(a => a.CompletedEpisodes > 0).Count;
    }

    /// <summary>
    /// Farm-wide statistics container.
    /// </summary>
    [System.Serializable]
    public struct FarmStatistics
    {
        public int AgentCount;
        public int TotalEpisodes;
        public int TotalBoundaryExits;
        public int TotalStraightLines;
        public float EpisodesPerAgent;
        public float FailureRate;
        public float ElapsedTime;

        public override string ToString()
        {
            return $"Farm Stats: {AgentCount} agents, {TotalEpisodes} episodes, " +
                   $"{EpisodesPerAgent:F1} eps/agent, {FailureRate:P1} failure rate, " +
                   $"{ElapsedTime:F1}s elapsed";
        }
    }
}