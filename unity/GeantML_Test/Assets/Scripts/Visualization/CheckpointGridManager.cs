using UnityEngine;
using Unity.Barracuda;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using TMPro;

namespace Visualization
{
    /// <summary>
    /// Manager for generating a grid of density texture visualizations from multiple training checkpoints.
    /// 
    /// Purpose: Visualize training progress by comparing density textures at different training steps.
    /// Creates a grid of prefabs along the Z-axis, each showing trajectories from a different checkpoint.
    /// 
    /// Output:
    /// - PNG images of each density texture
    /// - CSV file with statistics from all checkpoints
    /// - Comparison data for convergence analysis
    /// 
    /// Usage:
    /// 1. Assign the DensityTexture prefab
    /// 2. Set the checkpoint folder path (relative to Assets/)
    /// 3. Set the output folder path
    /// 4. Click "Scan and Generate Grid" in context menu
    /// </summary>
    public class CheckpointGridManager : MonoBehaviour
    {
        // ====================================================================
        // INSPECTOR SETTINGS
        // ====================================================================

        [Header("=== Prefab Configuration ===")]
        [Tooltip("Prefab with TrainedAgentBatchVisualizer configured for DensityTexture mode")]
        public GameObject DensityTexturePrefab;

        [Header("=== Checkpoint Source ===")]
        [Tooltip("Folder containing checkpoints relative to Assets/ (e.g., 'Models/results/ppo_base_v1/ElectronPhysics')")]
        public string CheckpointFolderPath = "Models/results/ppo_base_v1/ElectronPhysics";

        [Tooltip("Pattern to extract step count from filename (default matches 'ElectronPhysics-{steps}.onnx')")]
        public string FilenamePattern = @"ElectronPhysics-(\d+)\.onnx";

        [Header("=== Output Configuration ===")]
        [Tooltip("Base output folder for images and statistics")]
        public string OutputBasePath = @"C:\Thesis\figures";

        [Tooltip("Subfolder for density textures (will be created under OutputBasePath)")]
        public string DensityTextureSubfolder = "density_texture";

        [Tooltip("Algorithm name for subfolder (extracted from path or manual)")]
        public string AlgorithmName = "ppo_base_v1";

        [Header("=== Grid Layout ===")]
        [Tooltip("Spacing between prefabs along Z-axis (cm)")]
        public float GridSpacingZ = 15f;

        [Tooltip("Starting Z position for first prefab")]
        public float StartPositionZ = 0f;

        [Tooltip("Maximum number of checkpoints to process (0 = all)")]
        public int MaxCheckpoints = 0;

        [Tooltip("Skip every N checkpoints (1 = use all, 2 = every other, etc.)")]
        public int CheckpointSkipInterval = 1;

        [Header("=== Simulation Settings ===")]
        [Tooltip("Number of particles per checkpoint (lower = faster, higher = better quality)")]
        public int ParticlesPerCheckpoint = 10000;

        [Tooltip("Delay between simulations (seconds). Set to 0 for fastest processing.")]
        public float SimulationDelay = 0.5f;

        [Tooltip("LSTM models: override memory size if auto-detection fails (0 = auto)")]
        public int LSTMMemorySizeOverride = 0;

        [Tooltip("Force disable LSTM in visualizers - use MLP inference instead (may give incorrect results)")]
        public bool ForceDisableLSTMInVisualizers = false;

        [Header("=== Performance Optimization ===")]
        [Tooltip("Skip visual rendering in scene - only generate PNG files (MUCH faster)")]
        public bool FastModeSkipRendering = false;

        [Tooltip("Skip LSTM models - they have Barracuda compatibility issues")]
        public bool SkipLSTMModels = false;

        [Tooltip("Reduce yield frequency for faster processing")]
        public bool ReduceYieldFrequency = false;

        [Tooltip("Use legacy mode for maximum compatibility (slower but works with LSTM)")]
        public bool LegacyModeForLSTM = true;

        [Header("=== Label Settings ===")]
        [Tooltip("Format string for labels (use {0} for step count, {1} for formatted steps like '100k')")]
        public string LabelFormat = "{1} steps";

        [Tooltip("Label font size")]
        public float LabelFontSize = 0.5f;

        [Header("=== Export Settings ===")]
        [Tooltip("Export PNG images of density textures")]
        public bool ExportPNGImages = true;

        [Tooltip("Export CSV with statistics")]
        public bool ExportStatisticsCSV = true;

        [Tooltip("PNG image resolution")]
        public int ImageResolution = 512;

        [Header("=== Runtime Status ===")]
        [SerializeField]
        private int _checkpointsFound = 0;

        [SerializeField]
        private int _checkpointsProcessed = 0;

        [SerializeField]
        private string _currentStatus = "Idle";

        [SerializeField]
        private bool _isProcessing = false;

        // ====================================================================
        // PRIVATE STATE
        // ====================================================================

        private List<CheckpointInfo> _checkpoints = new List<CheckpointInfo>();
        private List<GameObject> _spawnedPrefabs = new List<GameObject>();
        private List<CheckpointStatistics> _allStatistics = new List<CheckpointStatistics>();
        private Coroutine _processingCoroutine;

        // ====================================================================
        // DATA STRUCTURES
        // ====================================================================

        [System.Serializable]
        public class CheckpointInfo
        {
            public string FilePath;
            public string FileName;
            public int StepCount;
            public NNModel Model;

            public string FormattedSteps
            {
                get
                {
                    if (StepCount >= 1000000)
                        return $"{StepCount / 1000000f:F1}M";
                    else if (StepCount >= 1000)
                        return $"{StepCount / 1000}k";
                    else
                        return StepCount.ToString();
                }
            }
        }

        [System.Serializable]
        public class CheckpointStatistics
        {
            public int StepCount;
            public string CheckpointName;
            public float MeanPathLength;
            public float StdPathLength;
            public float MeanPenetrationDepth;
            public float StdPenetrationDepth;
            public float MeanLateralSpread;
            public float StdLateralSpread;
            public float MeanScatterAngle;
            public float StdScatterAngle;
            public int NumParticles;
            public int BoundaryExits;
            public float BoundaryExitRate;
        }

        // ====================================================================
        // PUBLIC PROPERTIES
        // ====================================================================

        public bool IsProcessing => _isProcessing;
        public int CheckpointsFound => _checkpointsFound;
        public int CheckpointsProcessed => _checkpointsProcessed;
        public string Status => _currentStatus;
        public IReadOnlyList<CheckpointInfo> Checkpoints => _checkpoints;

        // ====================================================================
        // PUBLIC API
        // ====================================================================

        /// <summary>
        /// Scan checkpoint folder and generate the visualization grid.
        /// </summary>
        public void ScanAndGenerateGrid()
        {
            if (_isProcessing)
            {
                Debug.LogWarning("[CheckpointGridManager] Already processing!");
                return;
            }

            if (DensityTexturePrefab == null)
            {
                Debug.LogError("[CheckpointGridManager] DensityTexturePrefab is not assigned!");
                return;
            }

            _processingCoroutine = StartCoroutine(ProcessCheckpointsCoroutine());
        }

        /// <summary>
        /// Stop processing and clear generated prefabs.
        /// </summary>
        public void StopAndClear()
        {
            if (_processingCoroutine != null)
            {
                StopCoroutine(_processingCoroutine);
                _processingCoroutine = null;
            }

            ClearSpawnedPrefabs();
            _isProcessing = false;
            _currentStatus = "Stopped";
        }

        /// <summary>
        /// Clear all spawned prefabs.
        /// </summary>
        public void ClearSpawnedPrefabs()
        {
            foreach (var prefab in _spawnedPrefabs)
            {
                if (prefab != null)
                {
                    if (Application.isPlaying)
                        Destroy(prefab);
                    else
                        DestroyImmediate(prefab);
                }
            }
            _spawnedPrefabs.Clear();
            _checkpointsProcessed = 0;
        }

        /// <summary>
        /// Export all collected statistics to CSV.
        /// </summary>
        public void ExportStatistics()
        {
            if (_allStatistics.Count == 0)
            {
                Debug.LogWarning("[CheckpointGridManager] No statistics to export!");
                return;
            }

            string outputPath = GetOutputFolderPath();
            EnsureDirectoryExists(outputPath);

            string csvPath = Path.Combine(outputPath, $"{AlgorithmName}_statistics.csv");
            ExportStatisticsToCSV(csvPath);

            Debug.Log($"[CheckpointGridManager] Statistics exported to: {csvPath}");
        }

        /// <summary>
        /// Only scan checkpoints without generating (for preview).
        /// </summary>
        public void ScanCheckpointsOnly()
        {
            ScanCheckpointFolder();
            Debug.Log($"[CheckpointGridManager] Found {_checkpoints.Count} checkpoints:");
            foreach (var cp in _checkpoints)
            {
                Debug.Log($"  - {cp.FileName}: {cp.FormattedSteps}");
            }
        }

        // ====================================================================
        // MAIN PROCESSING COROUTINE
        // ====================================================================

        private IEnumerator ProcessCheckpointsCoroutine()
        {
            _isProcessing = true;
            _currentStatus = "Scanning checkpoints...";
            _allStatistics.Clear();

            Debug.Log("=======================================================");
            Debug.Log("[CheckpointGridManager] STARTING CHECKPOINT GRID GENERATION");
            Debug.Log("=======================================================");

            // Step 1: Scan checkpoint folder
            yield return null;
            ScanCheckpointFolder();

            if (_checkpoints.Count == 0)
            {
                Debug.LogError("[CheckpointGridManager] No checkpoints found!");
                _isProcessing = false;
                _currentStatus = "No checkpoints found";
                yield break;
            }

            Debug.Log($"[CheckpointGridManager] Found {_checkpoints.Count} checkpoints");

            // Step 2: Prepare output folder
            string outputPath = GetOutputFolderPath();
            EnsureDirectoryExists(outputPath);
            Debug.Log($"[CheckpointGridManager] Output folder: {outputPath}");

            // Step 3: Clear previous prefabs
            ClearSpawnedPrefabs();
            yield return null;

            // Step 4: Generate grid
            _currentStatus = "Generating grid...";
            System.Diagnostics.Stopwatch totalTimer = System.Diagnostics.Stopwatch.StartNew();

            // Check if this is LSTM folder and should be skipped
            bool isLSTMFolder = CheckpointFolderPath.ToLower().Contains("lstm");
            if (SkipLSTMModels && isLSTMFolder)
            {
                Debug.LogWarning("=======================================================");
                Debug.LogWarning("[CheckpointGridManager] LSTM FOLDER DETECTED - SKIPPING");
                Debug.LogWarning($"[CheckpointGridManager] Path: {CheckpointFolderPath}");
                Debug.LogWarning("[CheckpointGridManager] LSTM models have Barracuda compatibility issues.");
                Debug.LogWarning("[CheckpointGridManager] Set SkipLSTMModels=false to attempt anyway (will likely fail).");
                Debug.LogWarning("=======================================================");

                _isProcessing = false;
                _currentStatus = "Skipped (LSTM folder)";
                yield break;
            }

            for (int i = 0; i < _checkpoints.Count; i++)
            {
                var checkpoint = _checkpoints[i];
                _currentStatus = $"Processing {checkpoint.FormattedSteps} ({i + 1}/{_checkpoints.Count})";

                System.Diagnostics.Stopwatch checkpointTimer = System.Diagnostics.Stopwatch.StartNew();

                Debug.Log($"[CheckpointGridManager] Processing checkpoint: {checkpoint.FileName}");

                // Spawn prefab
                GameObject prefabInstance = SpawnPrefabForCheckpoint(checkpoint, i);
                if (prefabInstance == null)
                {
                    Debug.LogError($"[CheckpointGridManager] Failed to spawn prefab for {checkpoint.FileName}");
                    continue;
                }

                _spawnedPrefabs.Add(prefabInstance);

                // Get visualizer and configure
                var visualizer = prefabInstance.GetComponentInChildren<TrainedAgentBatchVisualizer>();
                if (visualizer == null)
                {
                    Debug.LogError($"[CheckpointGridManager] No TrainedAgentBatchVisualizer found in prefab!");
                    continue;
                }

                // In LegacyMode, use simpler ClearVisualization instead of FullReset
                // FullReset can cause issues with LSTM state management
                if (LegacyModeForLSTM)
                {
                    visualizer.ClearVisualization();
                }
                else
                {
                    // CRITICAL: Full reset before each checkpoint to clear LSTM state and GPU memory
                    visualizer.FullReset();

                    // Small delay to let GPU memory clear
                    yield return new WaitForSeconds(0.2f);
                }
                ;

                // Configure visualizer
                visualizer.TrainedModel = checkpoint.Model;
                visualizer.NumParticles = ParticlesPerCheckpoint;
                visualizer.Mode = TrainedAgentBatchVisualizer.VisualizationMode.DensityTexture;

                // Performance settings - disable in LegacyMode for LSTM compatibility
                if (LegacyModeForLSTM)
                {
                    visualizer.FastModeSkipRendering = false;
                    visualizer.ReduceYieldFrequency = false;
                }
                else
                {
                    visualizer.FastModeSkipRendering = FastModeSkipRendering;
                    visualizer.ReduceYieldFrequency = ReduceYieldFrequency;
                }

                // Handle LSTM settings
                if (ForceDisableLSTMInVisualizers)
                {
                    visualizer.ForceDisableLSTM = true;
                }
                if (LSTMMemorySizeOverride > 0)
                {
                    visualizer.LSTMMemorySizeOverride = LSTMMemorySizeOverride;
                }

                // Update label
                UpdatePrefabLabel(prefabInstance, checkpoint);

                // Run simulation
                visualizer.RunBatchSimulation();

                // Wait for completion with error handling
                bool simulationSuccess = true;
                System.Exception simulationError = null;

                yield return StartCoroutine(WaitForVisualizerCompletionWithErrorCheck(visualizer,
                    (success, error) => { simulationSuccess = success; simulationError = error; }));

                if (!simulationSuccess)
                {
                    Debug.LogWarning($"[CheckpointGridManager] Checkpoint {checkpoint.FileName} failed: {simulationError?.Message ?? "Unknown error"}");
                    Debug.LogWarning($"[CheckpointGridManager] Skipping this checkpoint and continuing...");

                    // Destroy failed prefab to clean up
                    if (prefabInstance != null)
                    {
                        _spawnedPrefabs.Remove(prefabInstance);
                        Destroy(prefabInstance);
                    }

                    _checkpointsProcessed = i + 1;

                    // Small delay before next to let Unity clean up
                    yield return new WaitForSeconds(0.5f);
                    continue;
                }

                // Collect statistics
                var stats = visualizer.GetStatistics();
                if (stats != null)
                {
                    var cpStats = ConvertToCheckpointStatistics(stats, checkpoint);
                    _allStatistics.Add(cpStats);
                }

                // Export PNG
                if (ExportPNGImages)
                {
                    yield return StartCoroutine(ExportDensityTexturePNG(visualizer, checkpoint, outputPath));
                }

                if (ExportStatisticsCSV)  // Używamy tej samej flagi co dla agregowanych statystyk
                {
                    string trajectoriesFileName = $"{AlgorithmName}_{checkpoint.StepCount:D7}_trajectories.csv";
                    string trajectoriesFilePath = System.IO.Path.Combine(outputPath, trajectoriesFileName);

                    visualizer.ExportTrajectoriesToCSV(trajectoriesFilePath);

                    Debug.Log($"[CheckpointGridManager] ✓ Exported per-trajectory CSV: {trajectoriesFileName}");
                }

                checkpointTimer.Stop();
                _checkpointsProcessed = i + 1;

                float checkpointSec = checkpointTimer.ElapsedMilliseconds / 1000f;
                float particlesPerSec = ParticlesPerCheckpoint / Mathf.Max(0.001f, checkpointSec);
                Debug.Log($"[CheckpointGridManager] Checkpoint {i + 1}/{_checkpoints.Count} done in {checkpointSec:F1}s ({particlesPerSec:F0} particles/sec)");

                // CRITICAL: Force cleanup after each checkpoint to prevent GPU memory accumulation
                // This is especially important for LSTM models which can leak memory
                System.GC.Collect();
                Resources.UnloadUnusedAssets();
                yield return null; // Let Unity process the cleanup

                // Delay before next
                if (SimulationDelay > 0 && i < _checkpoints.Count - 1)
                {
                    yield return new WaitForSeconds(SimulationDelay);
                }
            }

            totalTimer.Stop();

            // Step 5: Export statistics CSV
            if (ExportStatisticsCSV && _allStatistics.Count > 0)
            {
                string csvPath = Path.Combine(outputPath, $"{AlgorithmName}_statistics.csv");
                ExportStatisticsToCSV(csvPath);
            }

            _isProcessing = false;
            _currentStatus = $"Complete! {_checkpoints.Count} checkpoints processed";

            Debug.Log("=======================================================");
            Debug.Log("[CheckpointGridManager] GRID GENERATION COMPLETE");
            Debug.Log($"  Checkpoints: {_checkpoints.Count}");
            Debug.Log($"  Total time: {totalTimer.ElapsedMilliseconds / 1000f:F1} seconds");
            Debug.Log($"  Output: {outputPath}");
            Debug.Log("=======================================================");
        }

        // ====================================================================
        // CHECKPOINT SCANNING
        // ====================================================================

        private void ScanCheckpointFolder()
        {
            _checkpoints.Clear();

            string fullPath = Path.Combine(Application.dataPath, CheckpointFolderPath);

            if (!Directory.Exists(fullPath))
            {
                Debug.LogError($"[CheckpointGridManager] Checkpoint folder not found: {fullPath}");
                return;
            }

            // Find all .onnx files
            string[] onnxFiles = Directory.GetFiles(fullPath, "*.onnx");
            Regex stepPattern = new Regex(FilenamePattern);

            foreach (string filePath in onnxFiles)
            {
                string fileName = Path.GetFileName(filePath);
                Match match = stepPattern.Match(fileName);

                if (match.Success && match.Groups.Count > 1)
                {
                    if (int.TryParse(match.Groups[1].Value, out int stepCount))
                    {
#if UNITY_EDITOR
                        // Load the model asset (Editor only)
                        string assetPath = "Assets/" + CheckpointFolderPath + "/" + fileName;
                        NNModel model = UnityEditor.AssetDatabase.LoadAssetAtPath<NNModel>(assetPath);

                        if (model != null)
                        {
                            _checkpoints.Add(new CheckpointInfo
                            {
                                FilePath = filePath,
                                FileName = fileName,
                                StepCount = stepCount,
                                Model = model
                            });
                        }
                        else
                        {
                            Debug.LogWarning($"[CheckpointGridManager] Could not load model: {assetPath}");
                        }
#else
                        Debug.LogWarning("[CheckpointGridManager] Model loading only available in Editor!");
#endif
                    }
                }
            }

            // Sort by step count
            _checkpoints = _checkpoints.OrderBy(c => c.StepCount).ToList();

            // Apply skip interval
            if (CheckpointSkipInterval > 1)
            {
                _checkpoints = _checkpoints.Where((c, i) => i % CheckpointSkipInterval == 0).ToList();
            }

            // Apply max limit
            if (MaxCheckpoints > 0 && _checkpoints.Count > MaxCheckpoints)
            {
                _checkpoints = _checkpoints.Take(MaxCheckpoints).ToList();
            }

            _checkpointsFound = _checkpoints.Count;
        }

        // ====================================================================
        // PREFAB SPAWNING
        // ====================================================================

        private GameObject SpawnPrefabForCheckpoint(CheckpointInfo checkpoint, int index)
        {
            Vector3 position = new Vector3(0f, 0f, StartPositionZ + index * GridSpacingZ);

            GameObject instance = Instantiate(DensityTexturePrefab, position, Quaternion.identity, transform);
            instance.name = $"Checkpoint_{checkpoint.FormattedSteps}_{checkpoint.StepCount}";

            return instance;
        }

        private void UpdatePrefabLabel(GameObject prefabInstance, CheckpointInfo checkpoint)
        {
            // Find TextMeshPro label in children
            var labels = prefabInstance.GetComponentsInChildren<TextMeshPro>(true);

            foreach (var label in labels)
            {
                string formattedLabel = string.Format(LabelFormat, checkpoint.StepCount, checkpoint.FormattedSteps);
                label.text = formattedLabel;

                if (LabelFontSize > 0)
                {
                    label.fontSize = LabelFontSize;
                }
            }

            // Also try TextMeshProUGUI (for UI canvases)
            var uiLabels = prefabInstance.GetComponentsInChildren<TextMeshProUGUI>(true);
            foreach (var label in uiLabels)
            {
                string formattedLabel = string.Format(LabelFormat, checkpoint.StepCount, checkpoint.FormattedSteps);
                label.text = formattedLabel;
            }
        }

        // ====================================================================
        // WAITING FOR COMPLETION
        // ====================================================================

        private IEnumerator WaitForVisualizerCompletion(TrainedAgentBatchVisualizer visualizer)
        {
            yield return null; // Wait one frame for simulation to start

            float timeout = 300f; // 5 minutes
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

            Debug.LogWarning($"[CheckpointGridManager] Timeout waiting for visualizer");
        }

        private IEnumerator WaitForVisualizerCompletionWithErrorCheck(TrainedAgentBatchVisualizer visualizer,
            System.Action<bool, System.Exception> onComplete)
        {
            yield return null; // Wait one frame for simulation to start

            float timeout = 300f; // 5 minutes
            float elapsed = 0f;

            while (elapsed < timeout)
            {
                // Check if visualizer finished (either success or stopped)
                if (!visualizer.IsSimulating)
                {
                    var stats = visualizer.GetStatistics();
                    if (stats != null && stats.NumParticles > 0)
                    {
                        onComplete(true, null);
                        yield break;
                    }
                    else
                    {
                        // Simulation stopped but no stats - likely an error
                        onComplete(false, new System.Exception("Simulation stopped without producing statistics"));
                        yield break;
                    }
                }

                yield return new WaitForSeconds(0.5f);
                elapsed += 0.5f;
            }

            onComplete(false, new System.Exception("Timeout waiting for simulation"));
        }

        // ====================================================================
        // PNG EXPORT
        // ====================================================================

        private IEnumerator ExportDensityTexturePNG(TrainedAgentBatchVisualizer visualizer, CheckpointInfo checkpoint, string outputPath)
        {
            // Try direct texture access first (faster, no rendering needed)
            Texture2D directTexture = visualizer.GetDensityTexture();

            if (directTexture != null)
            {
                // Direct export - much faster!
                string fileName = $"{AlgorithmName}_{checkpoint.StepCount:D7}_steps.png";
                string filePath = Path.Combine(outputPath, fileName);

                byte[] pngData = directTexture.EncodeToPNG();
                File.WriteAllBytes(filePath, pngData);
                Debug.Log($"[CheckpointGridManager] Exported (direct): {fileName}");
                yield break;
            }

            // Fallback: Find texture through renderer (slower)
            yield return new WaitForEndOfFrame();

            var renderers = visualizer.GetComponentsInChildren<MeshRenderer>();

            foreach (var renderer in renderers)
            {
                if (renderer.material != null && renderer.material.mainTexture != null)
                {
                    Texture2D texture = renderer.material.mainTexture as Texture2D;
                    if (texture != null)
                    {
                        // Create a readable copy
                        Texture2D readableTexture = CreateReadableTexture(texture);

                        if (readableTexture != null)
                        {
                            byte[] pngData = readableTexture.EncodeToPNG();
                            string fileName = $"{AlgorithmName}_{checkpoint.StepCount:D7}_steps.png";
                            string filePath = Path.Combine(outputPath, fileName);

                            File.WriteAllBytes(filePath, pngData);
                            Debug.Log($"[CheckpointGridManager] Exported (fallback): {fileName}");

                            // Cleanup
                            if (Application.isPlaying)
                                Destroy(readableTexture);
                            else
                                DestroyImmediate(readableTexture);
                        }

                        break; // Only export first texture found
                    }
                }
            }
        }

        private Texture2D CreateReadableTexture(Texture2D source)
        {
            // Create a temporary RenderTexture
            RenderTexture rt = RenderTexture.GetTemporary(source.width, source.height, 0, RenderTextureFormat.ARGB32);
            Graphics.Blit(source, rt);

            // Read pixels from RenderTexture
            RenderTexture previous = RenderTexture.active;
            RenderTexture.active = rt;

            Texture2D readable = new Texture2D(source.width, source.height, TextureFormat.RGBA32, false);
            readable.ReadPixels(new Rect(0, 0, source.width, source.height), 0, 0);
            readable.Apply();

            RenderTexture.active = previous;
            RenderTexture.ReleaseTemporary(rt);

            return readable;
        }

        // ====================================================================
        // STATISTICS EXPORT
        // ====================================================================

        private CheckpointStatistics ConvertToCheckpointStatistics(TrainedAgentStatistics stats, CheckpointInfo checkpoint)
        {
            return new CheckpointStatistics
            {
                StepCount = checkpoint.StepCount,
                CheckpointName = checkpoint.FileName,
                MeanPathLength = stats.MeanPathLength,
                StdPathLength = stats.StdPathLength,
                MeanPenetrationDepth = stats.MeanPenetrationDepth,
                StdPenetrationDepth = stats.StdPenetrationDepth,
                MeanLateralSpread = stats.MeanLateralSpread,
                StdLateralSpread = stats.StdLateralSpread,
                MeanScatterAngle = stats.MeanScatterAngle,
                StdScatterAngle = stats.StdScatterAngle,
                NumParticles = stats.NumParticles,
                BoundaryExits = stats.BoundaryExits,
                BoundaryExitRate = (float)stats.BoundaryExits / stats.NumParticles * 100f
            };
        }

        private void ExportStatisticsToCSV(string filePath)
        {
            using (StreamWriter writer = new StreamWriter(filePath))
            {
                // Header
                writer.WriteLine("StepCount,CheckpointName,MeanPathLength,StdPathLength,MeanPenetrationDepth,StdPenetrationDepth,MeanLateralSpread,StdLateralSpread,MeanScatterAngle,StdScatterAngle,NumParticles,BoundaryExits,BoundaryExitRate");

                // Data rows
                foreach (var stats in _allStatistics.OrderBy(s => s.StepCount))
                {
                    writer.WriteLine($"{stats.StepCount},{stats.CheckpointName},{stats.MeanPathLength:F4},{stats.StdPathLength:F4},{stats.MeanPenetrationDepth:F4},{stats.StdPenetrationDepth:F4},{stats.MeanLateralSpread:F4},{stats.StdLateralSpread:F4},{stats.MeanScatterAngle:F4},{stats.StdScatterAngle:F4},{stats.NumParticles},{stats.BoundaryExits},{stats.BoundaryExitRate:F2}");
                }
            }

            Debug.Log($"[CheckpointGridManager] CSV exported: {filePath}");
        }

        // ====================================================================
        // HELPER METHODS
        // ====================================================================

        private string GetOutputFolderPath()
        {
            return Path.Combine(OutputBasePath, DensityTextureSubfolder, AlgorithmName);
        }

        private void EnsureDirectoryExists(string path)
        {
            if (!Directory.Exists(path))
            {
                Directory.CreateDirectory(path);
                Debug.Log($"[CheckpointGridManager] Created directory: {path}");
            }
        }

        // ====================================================================
        // EDITOR CONTEXT MENU
        // ====================================================================

#if UNITY_EDITOR
        [ContextMenu("Scan Checkpoints Only (Preview)")]
        private void EditorScanCheckpointsOnly()
        {
            ScanCheckpointsOnly();
        }

        [ContextMenu("Scan and Generate Grid")]
        private void EditorScanAndGenerateGrid()
        {
            ScanAndGenerateGrid();
        }

        [ContextMenu("Stop and Clear")]
        private void EditorStopAndClear()
        {
            StopAndClear();
        }

        [ContextMenu("Export Statistics CSV")]
        private void EditorExportStatistics()
        {
            ExportStatistics();
        }

        [ContextMenu("Open Output Folder")]
        private void EditorOpenOutputFolder()
        {
            string path = GetOutputFolderPath();
            if (Directory.Exists(path))
            {
                System.Diagnostics.Process.Start("explorer.exe", path.Replace("/", "\\"));
            }
            else
            {
                Debug.LogWarning($"[CheckpointGridManager] Output folder does not exist: {path}");
            }
        }

        [ContextMenu("Auto-detect Algorithm Name from Path")]
        private void EditorAutoDetectAlgorithmName()
        {
            // Extract algorithm name from path (e.g., "Models/results/ppo_base_v1/ElectronPhysics" -> "ppo_base_v1")
            string[] parts = CheckpointFolderPath.Split('/');
            if (parts.Length >= 2)
            {
                // Assume format: .../results/{algorithm_name}/...
                for (int i = 0; i < parts.Length - 1; i++)
                {
                    if (parts[i] == "results" && i + 1 < parts.Length)
                    {
                        AlgorithmName = parts[i + 1];
                        Debug.Log($"[CheckpointGridManager] Auto-detected algorithm name: {AlgorithmName}");
                        return;
                    }
                }
            }
            Debug.LogWarning("[CheckpointGridManager] Could not auto-detect algorithm name from path");
        }
#endif
    }
}