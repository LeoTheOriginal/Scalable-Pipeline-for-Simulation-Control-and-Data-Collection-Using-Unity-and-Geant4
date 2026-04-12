using UnityEngine;
using Unity.Barracuda;
using System.Collections;
using System.Collections.Generic;

namespace Visualization
{
    /// <summary>
    /// Manager for coordinating multiple TrainedAgentBatchVisualizer instances.
    /// 
    /// Use case: A prefab containing multiple visualization modes (PointCloud, LineSegments, Density)
    /// that should all use the same trained ONNX model.
    /// 
    /// Features:
    /// - Single point of control for model assignment
    /// - Sequential or parallel simulation execution
    /// - Unified statistics export and comparison
    /// - Editor buttons for quick operations
    /// </summary>
    public class TrainedAgentVisualizationManager : MonoBehaviour
    {
        // ====================================================================
        // INSPECTOR SETTINGS
        // ====================================================================

        [Header("=== Model Configuration ===")]
        [Tooltip("Trained ONNX model to use for all visualizers")]
        public NNModel SharedModel;

        [Tooltip("Automatically apply model to all visualizers when changed")]
        public bool AutoApplyOnModelChange = true;

        [Header("=== Visualizer Discovery ===")]
        [Tooltip("Automatically find all visualizers in children on Start")]
        public bool AutoDiscoverVisualizers = true;

        [Tooltip("Include inactive GameObjects when searching")]
        public bool IncludeInactiveVisualizers = false;

        [Header("=== Execution Settings ===")]
        [Tooltip("Run all simulations automatically on Start")]
        public bool RunOnStart = false;

        [Tooltip("Delay between sequential simulations (seconds)")]
        [Range(0f, 2f)]
        public float SimulationDelay = 0.5f;

        [Tooltip("Run simulations sequentially (true) or wait for each to complete (false)")]
        public bool SequentialExecution = true;

        [Header("=== Statistics Export ===")]
        [Tooltip("Export combined statistics after all simulations complete")]
        public bool AutoExportStatistics = false;

        [Tooltip("Path for combined statistics file")]
        public string CombinedStatisticsPath = "combined_visualization_statistics.txt";

        [Header("=== Visualizers (Auto-populated or Manual) ===")]
        [SerializeField]
        private List<TrainedAgentBatchVisualizer> _visualizers = new List<TrainedAgentBatchVisualizer>();

        [Header("=== Runtime Status ===")]
        [SerializeField, ReadOnlyInspector]
        private bool _isRunning = false;

        [SerializeField, ReadOnlyInspector]
        private int _completedSimulations = 0;

        [SerializeField, ReadOnlyInspector]
        private string _currentStatus = "Idle";

        // ====================================================================
        // PRIVATE STATE
        // ====================================================================

        private NNModel _previousModel;
        private Coroutine _runningCoroutine;

        // ====================================================================
        // PUBLIC PROPERTIES
        // ====================================================================

        /// <summary>Number of registered visualizers.</summary>
        public int VisualizerCount => _visualizers.Count;

        /// <summary>Whether any simulation is currently running.</summary>
        public bool IsRunning => _isRunning;

        /// <summary>Current status message.</summary>
        public string Status => _currentStatus;

        /// <summary>Read-only access to visualizers list.</summary>
        public IReadOnlyList<TrainedAgentBatchVisualizer> Visualizers => _visualizers;

        // ====================================================================
        // LIFECYCLE
        // ====================================================================

        private void Awake()
        {
            _previousModel = SharedModel;
        }

        private void Start()
        {
            if (AutoDiscoverVisualizers)
            {
                DiscoverVisualizers();
            }

            if (SharedModel != null && _visualizers.Count > 0)
            {
                ApplyModelToAll();
            }

            if (RunOnStart && SharedModel != null)
            {
                RunAllSimulations();
            }
        }

        private void OnValidate()
        {
            // Detect model change in Inspector
            if (AutoApplyOnModelChange && SharedModel != _previousModel)
            {
                _previousModel = SharedModel;

                // Only apply if in Play mode (OnValidate is called in Edit mode too)
                if (Application.isPlaying && _visualizers.Count > 0)
                {
                    ApplyModelToAll();
                    Debug.Log($"[VisualizationManager] Model changed to: {(SharedModel != null ? SharedModel.name : "NULL")}");
                }
            }
        }

        private void OnDestroy()
        {
            if (_runningCoroutine != null)
            {
                StopCoroutine(_runningCoroutine);
            }
        }

        // ====================================================================
        // PUBLIC API - VISUALIZER MANAGEMENT
        // ====================================================================

        /// <summary>
        /// Discover all TrainedAgentBatchVisualizer components in children.
        /// </summary>
        public void DiscoverVisualizers()
        {
            _visualizers.Clear();

            TrainedAgentBatchVisualizer[] found = GetComponentsInChildren<TrainedAgentBatchVisualizer>(IncludeInactiveVisualizers);

            foreach (var visualizer in found)
            {
                // Don't include the manager's own GameObject if it has a visualizer
                if (visualizer.gameObject != gameObject)
                {
                    _visualizers.Add(visualizer);
                }
            }

            Debug.Log($"[VisualizationManager] Discovered {_visualizers.Count} visualizers");

            foreach (var v in _visualizers)
            {
                Debug.Log($"  - {v.gameObject.name} (Mode: {v.Mode})");
            }
        }

        /// <summary>
        /// Manually add a visualizer to the managed list.
        /// </summary>
        public void AddVisualizer(TrainedAgentBatchVisualizer visualizer)
        {
            if (visualizer != null && !_visualizers.Contains(visualizer))
            {
                _visualizers.Add(visualizer);

                // Apply current model if set
                if (SharedModel != null)
                {
                    visualizer.TrainedModel = SharedModel;
                }

                Debug.Log($"[VisualizationManager] Added visualizer: {visualizer.gameObject.name}");
            }
        }

        /// <summary>
        /// Remove a visualizer from the managed list.
        /// </summary>
        public void RemoveVisualizer(TrainedAgentBatchVisualizer visualizer)
        {
            if (_visualizers.Remove(visualizer))
            {
                Debug.Log($"[VisualizationManager] Removed visualizer: {visualizer.gameObject.name}");
            }
        }

        /// <summary>
        /// Clear all visualizers from the managed list.
        /// </summary>
        public void ClearVisualizers()
        {
            _visualizers.Clear();
            Debug.Log("[VisualizationManager] Cleared all visualizers");
        }

        // ====================================================================
        // PUBLIC API - MODEL MANAGEMENT
        // ====================================================================

        /// <summary>
        /// Apply the shared model to all registered visualizers.
        /// </summary>
        public void ApplyModelToAll()
        {
            if (SharedModel == null)
            {
                Debug.LogWarning("[VisualizationManager] Cannot apply NULL model!");
                return;
            }

            int applied = 0;
            foreach (var visualizer in _visualizers)
            {
                if (visualizer != null)
                {
                    visualizer.TrainedModel = SharedModel;
                    applied++;
                }
            }

            Debug.Log($"[VisualizationManager] Applied model '{SharedModel.name}' to {applied} visualizers");
        }

        /// <summary>
        /// Set a new model and apply it to all visualizers.
        /// </summary>
        public void SetModel(NNModel model)
        {
            SharedModel = model;
            _previousModel = model;

            if (model != null)
            {
                ApplyModelToAll();
            }
        }

        // ====================================================================
        // PUBLIC API - SIMULATION EXECUTION
        // ====================================================================

        /// <summary>
        /// Run simulations on all visualizers.
        /// </summary>
        public void RunAllSimulations()
        {
            if (_isRunning)
            {
                Debug.LogWarning("[VisualizationManager] Simulations already running!");
                return;
            }

            if (_visualizers.Count == 0)
            {
                Debug.LogWarning("[VisualizationManager] No visualizers to run!");
                return;
            }

            if (SharedModel == null)
            {
                Debug.LogError("[VisualizationManager] No model assigned!");
                return;
            }

            // Ensure model is applied
            ApplyModelToAll();

            _runningCoroutine = StartCoroutine(RunSimulationsCoroutine());
        }

        /// <summary>
        /// Stop all running simulations.
        /// </summary>
        public void StopAllSimulations()
        {
            if (_runningCoroutine != null)
            {
                StopCoroutine(_runningCoroutine);
                _runningCoroutine = null;
            }

            _isRunning = false;
            _currentStatus = "Stopped";
            Debug.Log("[VisualizationManager] Simulations stopped");
        }

        /// <summary>
        /// Clear visualizations from all visualizers.
        /// </summary>
        public void ClearAllVisualizations()
        {
            foreach (var visualizer in _visualizers)
            {
                if (visualizer != null)
                {
                    visualizer.ClearVisualization();
                }
            }

            Debug.Log("[VisualizationManager] Cleared all visualizations");
        }

        // ====================================================================
        // PUBLIC API - STATISTICS
        // ====================================================================

        /// <summary>
        /// Get statistics from all visualizers that have completed simulation.
        /// </summary>
        public List<TrainedAgentStatistics> GetAllStatistics()
        {
            List<TrainedAgentStatistics> allStats = new List<TrainedAgentStatistics>();

            foreach (var visualizer in _visualizers)
            {
                if (visualizer != null)
                {
                    var stats = visualizer.GetStatistics();
                    if (stats != null)
                    {
                        allStats.Add(stats);
                    }
                }
            }

            return allStats;
        }

        /// <summary>
        /// Export statistics from all visualizers to individual files.
        /// </summary>
        public void ExportAllStatistics()
        {
            foreach (var visualizer in _visualizers)
            {
                if (visualizer != null)
                {
                    visualizer.ExportStatistics();
                }
            }

            Debug.Log($"[VisualizationManager] Exported statistics from {_visualizers.Count} visualizers");
        }

        /// <summary>
        /// Export combined statistics comparison to a single file.
        /// </summary>
        public void ExportCombinedStatistics()
        {
            var allStats = GetAllStatistics();

            if (allStats.Count == 0)
            {
                Debug.LogWarning("[VisualizationManager] No statistics to export!");
                return;
            }

            string report = GenerateCombinedStatisticsReport(allStats);
            string fullPath = System.IO.Path.Combine(Application.dataPath, CombinedStatisticsPath);

            System.IO.File.WriteAllText(fullPath, report);
            Debug.Log($"[VisualizationManager] Combined statistics exported to: {fullPath}");
        }

        /// <summary>
        /// Log a comparison of statistics from all visualizers to the console.
        /// </summary>
        public void LogStatisticsComparison()
        {
            var allStats = GetAllStatistics();

            if (allStats.Count == 0)
            {
                Debug.Log("[VisualizationManager] No statistics available for comparison");
                return;
            }

            Debug.Log("=======================================================");
            Debug.Log("[VisualizationManager] STATISTICS COMPARISON");
            Debug.Log("=======================================================");

            for (int i = 0; i < _visualizers.Count && i < allStats.Count; i++)
            {
                var v = _visualizers[i];
                var s = allStats[i];

                Debug.Log($"\n--- {v.gameObject.name} ({v.Mode}) ---");
                Debug.Log(s.ToString());
            }

            // Calculate and show averages if multiple visualizers
            if (allStats.Count > 1)
            {
                Debug.Log("\n--- AVERAGES ACROSS ALL VISUALIZERS ---");
                LogAverageStatistics(allStats);
            }
        }

        // ====================================================================
        // COROUTINES
        // ====================================================================

        private IEnumerator RunSimulationsCoroutine()
        {
            _isRunning = true;
            _completedSimulations = 0;
            _currentStatus = "Starting simulations...";

            Debug.Log("=======================================================");
            Debug.Log("[VisualizationManager] STARTING ALL SIMULATIONS");
            Debug.Log($"  Model: {SharedModel.name}");
            Debug.Log($"  Visualizers: {_visualizers.Count}");
            Debug.Log($"  Mode: {(SequentialExecution ? "Sequential" : "Parallel")}");
            Debug.Log("=======================================================");

            System.Diagnostics.Stopwatch totalTimer = System.Diagnostics.Stopwatch.StartNew();

            if (SequentialExecution)
            {
                // Run one at a time
                for (int i = 0; i < _visualizers.Count; i++)
                {
                    var visualizer = _visualizers[i];
                    if (visualizer == null) continue;

                    _currentStatus = $"Running {visualizer.gameObject.name} ({i + 1}/{_visualizers.Count})";
                    Debug.Log($"[VisualizationManager] {_currentStatus}");

                    visualizer.RunBatchSimulation();

                    // Wait for this visualizer to complete
                    // We detect completion by checking if it's still simulating
                    // Note: This requires TrainedAgentBatchVisualizer to have a public IsSimulating property
                    yield return StartCoroutine(WaitForVisualizerCompletion(visualizer));

                    _completedSimulations++;

                    if (SimulationDelay > 0 && i < _visualizers.Count - 1)
                    {
                        yield return new WaitForSeconds(SimulationDelay);
                    }
                }
            }
            else
            {
                // Start all at once
                foreach (var visualizer in _visualizers)
                {
                    if (visualizer != null)
                    {
                        visualizer.RunBatchSimulation();
                    }
                }

                _currentStatus = $"Running all {_visualizers.Count} visualizers in parallel";

                // Wait for all to complete
                yield return StartCoroutine(WaitForAllVisualizersCompletion());
            }

            totalTimer.Stop();

            _currentStatus = "All simulations complete";
            _isRunning = false;

            Debug.Log("=======================================================");
            Debug.Log("[VisualizationManager] ALL SIMULATIONS COMPLETE");
            Debug.Log($"  Total time: {totalTimer.ElapsedMilliseconds / 1000f:F2} seconds");
            Debug.Log("=======================================================");

            // Auto-export if enabled
            if (AutoExportStatistics)
            {
                ExportCombinedStatistics();
            }

            // Log comparison
            LogStatisticsComparison();
        }

        private IEnumerator WaitForVisualizerCompletion(TrainedAgentBatchVisualizer visualizer)
        {
            // Wait a frame for simulation to start
            yield return null;

            // Poll until the visualizer has data (indicating completion)
            // We use GetStatistics() != null as completion indicator
            float timeout = 300f; // 5 minute timeout
            float elapsed = 0f;

            while (elapsed < timeout)
            {
                var stats = visualizer.GetStatistics();
                if (stats != null && stats.NumParticles > 0)
                {
                    yield break; // Completed
                }

                yield return new WaitForSeconds(0.5f);
                elapsed += 0.5f;
            }

            Debug.LogWarning($"[VisualizationManager] Timeout waiting for {visualizer.gameObject.name}");
        }

        private IEnumerator WaitForAllVisualizersCompletion()
        {
            float timeout = 300f;
            float elapsed = 0f;

            while (elapsed < timeout)
            {
                int completed = 0;

                foreach (var visualizer in _visualizers)
                {
                    if (visualizer == null) continue;

                    var stats = visualizer.GetStatistics();
                    if (stats != null && stats.NumParticles > 0)
                    {
                        completed++;
                    }
                }

                _completedSimulations = completed;
                _currentStatus = $"Completed: {completed}/{_visualizers.Count}";

                if (completed >= _visualizers.Count)
                {
                    yield break;
                }

                yield return new WaitForSeconds(0.5f);
                elapsed += 0.5f;
            }

            Debug.LogWarning("[VisualizationManager] Timeout waiting for all visualizers");
        }

        // ====================================================================
        // STATISTICS HELPERS
        // ====================================================================

        private string GenerateCombinedStatisticsReport(List<TrainedAgentStatistics> allStats)
        {
            System.Text.StringBuilder sb = new System.Text.StringBuilder();

            sb.AppendLine("Combined Visualization Statistics Report");
            sb.AppendLine("========================================");
            sb.AppendLine($"Generated: {System.DateTime.Now}");
            sb.AppendLine($"Model: {(SharedModel != null ? SharedModel.name : "Unknown")}");
            sb.AppendLine($"Visualizers: {_visualizers.Count}");
            sb.AppendLine();

            // Individual visualizer statistics
            for (int i = 0; i < _visualizers.Count && i < allStats.Count; i++)
            {
                var v = _visualizers[i];
                var s = allStats[i];

                sb.AppendLine($"--- {v.gameObject.name} ({v.Mode}) ---");
                sb.AppendLine(s.ToDetailedString());
                sb.AppendLine();
            }

            // Averages
            if (allStats.Count > 1)
            {
                sb.AppendLine("--- AVERAGE STATISTICS ---");
                sb.AppendLine(CalculateAverageStatisticsString(allStats));
            }

            return sb.ToString();
        }

        private void LogAverageStatistics(List<TrainedAgentStatistics> allStats)
        {
            if (allStats.Count == 0) return;

            float avgPathLength = 0f;
            float avgPenetration = 0f;
            float avgLateralSpread = 0f;
            float avgScatterAngle = 0f;

            foreach (var s in allStats)
            {
                avgPathLength += s.MeanPathLength;
                avgPenetration += s.MeanPenetrationDepth;
                avgLateralSpread += s.MeanLateralSpread;
                avgScatterAngle += s.MeanScatterAngle;
            }

            int n = allStats.Count;
            Debug.Log($"  Avg Path Length: {avgPathLength / n:F3} cm");
            Debug.Log($"  Avg Penetration: {avgPenetration / n:F3} cm");
            Debug.Log($"  Avg Lateral Spread: {avgLateralSpread / n:F3} cm");
            Debug.Log($"  Avg Scatter Angle: {avgScatterAngle / n:F2}°");
        }

        private string CalculateAverageStatisticsString(List<TrainedAgentStatistics> allStats)
        {
            if (allStats.Count == 0) return "No statistics available";

            float avgPathLength = 0f;
            float avgPenetration = 0f;
            float avgLateralSpread = 0f;
            float avgScatterAngle = 0f;

            foreach (var s in allStats)
            {
                avgPathLength += s.MeanPathLength;
                avgPenetration += s.MeanPenetrationDepth;
                avgLateralSpread += s.MeanLateralSpread;
                avgScatterAngle += s.MeanScatterAngle;
            }

            int n = allStats.Count;
            return $"Path Length: {avgPathLength / n:F4} cm\n" +
                   $"Penetration Depth: {avgPenetration / n:F4} cm\n" +
                   $"Lateral Spread: {avgLateralSpread / n:F4} cm\n" +
                   $"Scatter Angle: {avgScatterAngle / n:F2}°";
        }

        // ====================================================================
        // EDITOR CONTEXT MENU
        // ====================================================================

#if UNITY_EDITOR
        [ContextMenu("Discover Visualizers")]
        private void EditorDiscoverVisualizers()
        {
            DiscoverVisualizers();
        }

        [ContextMenu("Apply Model to All")]
        private void EditorApplyModelToAll()
        {
            ApplyModelToAll();
        }

        [ContextMenu("Run All Simulations")]
        private void EditorRunAllSimulations()
        {
            RunAllSimulations();
        }

        [ContextMenu("Stop All Simulations")]
        private void EditorStopAllSimulations()
        {
            StopAllSimulations();
        }

        [ContextMenu("Clear All Visualizations")]
        private void EditorClearAllVisualizations()
        {
            ClearAllVisualizations();
        }

        [ContextMenu("Export All Statistics")]
        private void EditorExportAllStatistics()
        {
            ExportAllStatistics();
        }

        [ContextMenu("Export Combined Statistics")]
        private void EditorExportCombinedStatistics()
        {
            ExportCombinedStatistics();
        }

        [ContextMenu("Log Statistics Comparison")]
        private void EditorLogStatisticsComparison()
        {
            LogStatisticsComparison();
        }
#endif
    }

    // ====================================================================
    // HELPER ATTRIBUTE FOR READ-ONLY INSPECTOR FIELDS
    // ====================================================================

    /// <summary>
    /// Attribute to display a field as read-only in the Inspector.
    /// </summary>
    public class ReadOnlyInspectorAttribute : PropertyAttribute { }

#if UNITY_EDITOR
    [UnityEditor.CustomPropertyDrawer(typeof(ReadOnlyInspectorAttribute))]
    public class ReadOnlyInspectorDrawer : UnityEditor.PropertyDrawer
    {
        public override void OnGUI(Rect position, UnityEditor.SerializedProperty property, GUIContent label)
        {
            GUI.enabled = false;
            UnityEditor.EditorGUI.PropertyField(position, property, label, true);
            GUI.enabled = true;
        }
    }
#endif
}