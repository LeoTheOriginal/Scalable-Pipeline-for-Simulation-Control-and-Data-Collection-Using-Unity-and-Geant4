using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using Unity.Barracuda;
using Physics;

namespace Visualization
{
    /// <summary>
    /// Batch visualizer for trained RL agent trajectories.
    /// Supports both standard MLP models and LSTM recurrent models.
    /// 
    /// Runs the trained ONNX model in inference mode to generate
    /// thousands of trajectories rapidly, then visualizes them
    /// using various rendering methods.
    /// 
    /// Energy gradient: Green (high energy) → Red (low energy)
    /// 
    /// LSTM Support:
    /// - Automatically detects LSTM models by checking for 'recurrent_in' input
    /// - Maintains hidden state across steps within each particle trajectory
    /// - Resets hidden state at the start of each new particle
    /// </summary>
    public class TrainedAgentBatchVisualizer : MonoBehaviour
    {
        // ====================================================================
        // INSPECTOR SETTINGS
        // ====================================================================

        [Header("Model Settings")]
        [Tooltip("Trained ONNX model for inference")]
        public NNModel TrainedModel;

        [Header("Simulation Settings")]
        [Tooltip("Number of particles (episodes) to simulate")]
        public int NumParticles = 100000;

        [Tooltip("Maximum steps per particle")]
        public int MaxStepsPerParticle = 500;

        [Tooltip("Run simulation on Start")]
        public bool RunOnStart = false;

        [Tooltip("Maximum total steps to visualize (memory limit)")]
        public int MaxVisualizationSteps = 10000000;

        [Header("Physics Settings")]
        [Tooltip("Maximum step size in cm")]
        public float MaxStepSize = 0.03f;

        [Tooltip("Minimum step size in cm")]
        public float MinStepSize = 0.005f;

        [Tooltip("Maximum direction change per step")]
        public float MaxDirectionChange = 0.5f;

        [Tooltip("Maximum scatter angle per step (degrees)")]
        public float MaxScatterAnglePerStep = 20f;

        [Header("Stochastic Policy Settings")]
        [Tooltip("Standard deviation for action noise (mimics ML-Agents stochastic policy). Higher = more spread.")]
        [Range(0f, 1f)]
        public float ActionNoiseStdDev = 0.3f;

        [Header("LSTM Settings")]
        [Tooltip("Override LSTM memory size (0 = auto-detect from model). Use if auto-detection fails.")]
        public int LSTMMemorySizeOverride = 0;

        [Tooltip("Force disable LSTM - treat LSTM models as regular MLP (may give incorrect results but won't crash)")]
        public bool ForceDisableLSTM = false;

        [Tooltip("Recreate LSTM worker every N particles to prevent memory issues (0 = never)")]
        public int LSTMWorkerRecreateInterval = 0;

        [Tooltip("Disable auto-adjustment of LSTM size on errors (use for debugging)")]
        public bool DisableLSTMAutoAdjust = true;

        [Header("Performance Optimization")]
        [Tooltip("Skip visual rendering in scene - only generate texture data (MUCH faster for PNG export)")]
        public bool FastModeSkipRendering = false;

        [Tooltip("Reduce yield frequency for faster processing (less responsive UI but faster)")]
        public bool ReduceYieldFrequency = false;

        [Tooltip("Yield every N particles (when ReduceYieldFrequency is true)")]
        public int YieldEveryNParticles = 500;

        [Header("Visualization Mode")]
        [Tooltip("Use point cloud (faster) or line rendering")]
        public VisualizationMode Mode = VisualizationMode.PointCloud;

        [Tooltip("Point/line size")]
        public float PointSize = 0.01f;

        [Header("Color Settings")]
        [Tooltip("High energy color (10 MeV) - Blue (physics standard)")]
        public Color HighEnergyColor = new Color(0.2f, 0.4f, 1.0f, 0.9f);

        [Tooltip("Mid energy color (5 MeV) - Red")]
        public Color MidEnergyColor = new Color(1.0f, 0.3f, 0.3f, 0.85f);

        [Tooltip("Low energy color (approaching 0 MeV) - White")]
        public Color LowEnergyColor = new Color(1.0f, 1.0f, 1.0f, 0.95f);

        [Tooltip("Background/phantom color (dark navy)")]
        public Color PhantomColor = new Color(0.05f, 0.05f, 0.15f, 1.0f);

        [Header("Density Settings")]
        [Tooltip("Use logarithmic scaling for density")]
        public bool UseLogScale = true;

        [Tooltip("Minimum alpha for sparse regions")]
        [Range(0.01f, 0.5f)]
        public float MinAlpha = 0.05f;

        [Tooltip("Maximum alpha for dense regions")]
        [Range(0.5f, 1.0f)]
        public float MaxAlpha = 0.95f;

        [Tooltip("Texture resolution (square)")]
        public int TextureResolution = 512;

        [Header("Statistics")]
        [Tooltip("Path for statistics file")]
        public string StatisticsFilePath = "trained_agent_statistics.txt";

        [Header("Debug")]
        public bool ShowProgress = true;
        public bool ShowCoordinateAxes = false;

        [Tooltip("Log every N particles during simulation")]
        public int ProgressLogInterval = 10000;

        public enum VisualizationMode
        {
            PointCloud,
            LineSegments,
            DensityTexture
        }

        // ====================================================================
        // RUNTIME STATE
        // ====================================================================

        private bool _isSimulating = false;
        private bool _hasData = false;
        private TrainedAgentStatistics _statistics;

        // Neural network inference
        private IWorker _worker;
        private Model _model;
        private string _inputName;
        private string _outputName;

        // LSTM support
        private bool _isLSTM = false;
        private string _recurrentInName;
        private string _recurrentOutName;
        private int _lstmHiddenSize = 128;  // Default, will be detected from model

        // Trajectory data storage
        private List<float> _allPositionsX;
        private List<float> _allPositionsY;
        private List<float> _allPositionsZ;
        private List<float> _allEnergies;
        private List<int> _particleBoundaries;

        // Visualization objects
        private Mesh _pointMesh;
        private Material _pointMaterial;
        private GameObject _visualizationObject;
        private Texture2D _densityTexture;
        private float[,] _densityMap;
        private float[,] _energyMap;

        // ====================================================================
        // PUBLIC PROPERTIES
        // ====================================================================

        /// <summary>Whether simulation is currently running.</summary>
        public bool IsSimulating => _isSimulating;

        /// <summary>Whether visualization data is available.</summary>
        public bool HasData => _hasData;

        /// <summary>Whether the loaded model is LSTM.</summary>
        public bool IsLSTMModel => _isLSTM;

        // ====================================================================
        // PUBLIC API
        // ====================================================================

        /// <summary>
        /// Start batch simulation asynchronously.
        /// </summary>
        public void RunBatchSimulation()
        {
            Debug.Log("[TrainedAgentVisualizer] RunBatchSimulation() called");

            if (_isSimulating)
            {
                Debug.LogWarning("[TrainedAgentVisualizer] Simulation already running!");
                return;
            }

            if (TrainedModel == null)
            {
                Debug.LogError("[TrainedAgentVisualizer] No trained model assigned! Drag an ONNX file to the 'Trained Model' field.");
                return;
            }

            Debug.Log($"[TrainedAgentVisualizer] Starting coroutine with model: {TrainedModel.name}");
            StartCoroutine(RunSimulationCoroutine());
        }

        /// <summary>
        /// Get computed statistics.
        /// </summary>
        public TrainedAgentStatistics GetStatistics()
        {
            return _statistics;
        }

        /// <summary>
        /// Export statistics to file.
        /// </summary>
        public void ExportStatistics()
        {
            if (!_hasData || _statistics == null)
            {
                Debug.LogWarning("[TrainedAgentVisualizer] No data to export!");
                return;
            }

            string fullPath = System.IO.Path.Combine(Application.dataPath, StatisticsFilePath);
            System.IO.File.WriteAllText(fullPath, _statistics.ToDetailedString());
            Debug.Log($"[TrainedAgentVisualizer] Statistics exported to: {fullPath}");
        }

        /// <summary>
        /// Clear visualization.
        /// </summary>
        public void ClearVisualization()
        {
            if (_visualizationObject != null)
            {
                Destroy(_visualizationObject);
                _visualizationObject = null;
            }

            _hasData = false;
        }

        /// <summary>
        /// Full reset - clears visualization AND disposes neural network.
        /// Call this before switching to a new model.
        /// </summary>
        public void FullReset()
        {
            Debug.Log("[TrainedAgentVisualizer] Performing full reset...");

            // Stop any running simulation
            StopAllCoroutines();
            _isSimulating = false;

            // Clear visualization
            ClearVisualization();

            // Dispose neural network worker
            DisposeWorker();

            // Clear all data lists
            if (_allPositionsX != null) _allPositionsX.Clear();
            if (_allPositionsY != null) _allPositionsY.Clear();
            if (_allPositionsZ != null) _allPositionsZ.Clear();
            if (_allEnergies != null) _allEnergies.Clear();
            if (_particleBoundaries != null) _particleBoundaries.Clear();

            // Reset statistics
            _statistics = null;

            // Reset LSTM state
            _isLSTM = false;
            _recurrentInName = null;
            _recurrentOutName = null;
            _lstmHiddenSize = 128;
            _lstmSizeMismatchDetected = false;
            _triedMLPFallback = false;

            // Force garbage collection to free GPU memory
            System.GC.Collect();
            Resources.UnloadUnusedAssets();

            Debug.Log("[TrainedAgentVisualizer] Full reset complete");
        }

        // ====================================================================
        // LIFECYCLE
        // ====================================================================

        void Awake()
        {
            Debug.Log($"[TrainedAgentVisualizer] Awake() on {gameObject.name}, enabled={enabled}");
        }

        void OnEnable()
        {
            Debug.Log($"[TrainedAgentVisualizer] OnEnable() on {gameObject.name}");
        }

        void Start()
        {
            Debug.Log($"[TrainedAgentVisualizer] Start() called on {gameObject.name}");
            Debug.Log($"[TrainedAgentVisualizer] RunOnStart={RunOnStart}, TrainedModel={(TrainedModel != null ? TrainedModel.name : "NULL")}");

            _allPositionsX = new List<float>();
            _allPositionsY = new List<float>();
            _allPositionsZ = new List<float>();
            _allEnergies = new List<float>();
            _particleBoundaries = new List<int>();

            if (RunOnStart)
            {
                if (TrainedModel != null)
                {
                    Debug.Log("[TrainedAgentVisualizer] Starting batch simulation...");
                    RunBatchSimulation();
                }
                else
                {
                    // Not an error when used with CheckpointGridManager - it assigns model later
                    Debug.Log("[TrainedAgentVisualizer] RunOnStart is true but TrainedModel not yet assigned. " +
                             "If using CheckpointGridManager, this is expected - simulation will start when model is assigned.");
                }
            }
            else
            {
                Debug.Log("[TrainedAgentVisualizer] RunOnStart is false. Use context menu 'Run Simulation' to start manually.");
            }

            if (ShowCoordinateAxes)
            {
                DrawCoordinateAxes();
            }
        }

        void OnDestroy()
        {
            ClearVisualization();
            DisposeWorker();

            if (_pointMaterial != null) Destroy(_pointMaterial);
            if (_pointMesh != null) Destroy(_pointMesh);
            if (_densityTexture != null) Destroy(_densityTexture);
        }

        private void DisposeWorker()
        {
            if (_worker != null)
            {
                _worker.Dispose();
                _worker = null;
            }
        }

        // ====================================================================
        // SIMULATION COROUTINE
        // ====================================================================

        private IEnumerator RunSimulationCoroutine()
        {
            _isSimulating = true;
            _hasData = false;

            Debug.Log("=======================================================");
            Debug.Log("[TrainedAgentVisualizer] STARTING BATCH SIMULATION");
            Debug.Log("=======================================================");
            Debug.Log($"[TrainedAgentVisualizer] NumParticles: {NumParticles}");
            Debug.Log($"[TrainedAgentVisualizer] MaxStepsPerParticle: {MaxStepsPerParticle}");
            Debug.Log($"[TrainedAgentVisualizer] ActionNoiseStdDev: {ActionNoiseStdDev}");
            Debug.Log($"[TrainedAgentVisualizer] Mode: {Mode}");

            // Initialize neural network
            bool nnSuccess = false;
            try
            {
                nnSuccess = InitializeNeuralNetwork();
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[TrainedAgentVisualizer] Exception in InitializeNeuralNetwork: {e.Message}\n{e.StackTrace}");
                _isSimulating = false;
                yield break;
            }

            if (!nnSuccess)
            {
                Debug.LogError("[TrainedAgentVisualizer] Failed to initialize neural network - ABORTING!");
                _isSimulating = false;
                yield break;
            }

            yield return null;

            // Clear previous data
            _allPositionsX.Clear();
            _allPositionsY.Clear();
            _allPositionsZ.Clear();
            _allEnergies.Clear();
            _particleBoundaries.Clear();

            // Statistics accumulators
            List<float> allPathLengths = new List<float>();
            List<float> allFinalDepths = new List<float>();
            List<float> allLateralSpreads = new List<float>();
            List<float> allMeanScatterAngles = new List<float>();
            List<int> allStepCounts = new List<int>();
            int boundaryExits = 0;

            System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
            int totalSteps = 0;

            Debug.Log("-------------------------------------------------------");
            Debug.Log($"[TrainedAgentVisualizer] Starting particle simulation loop... (LSTM: {_isLSTM})");
            Debug.Log("-------------------------------------------------------");

            // Run simulations
            int failedParticles = 0;
            int maxFailedParticles = Mathf.Max(10, NumParticles / 100); // Allow up to 1% failures
            _lstmSizeMismatchDetected = false;
            _triedMLPFallback = false;

            for (int p = 0; p < NumParticles; p++)
            {
                if (totalSteps >= MaxVisualizationSteps)
                {
                    Debug.Log($"[TrainedAgentVisualizer] Reached max visualization steps ({MaxVisualizationSteps})");
                    break;
                }

                // Check if we need to auto-adjust LSTM size and restart
                // DISABLED by default as it can cause more problems than it solves
                if (!DisableLSTMAutoAdjust && _lstmSizeMismatchDetected && _isLSTM)
                {
                    Debug.LogWarning($"[TrainedAgentVisualizer] LSTM size mismatch detected! Restarting with adjusted size...");

                    // Try with half the size (hidden_units instead of memory_size)
                    int newSize = _lstmHiddenSize / 2;
                    Debug.Log($"[TrainedAgentVisualizer] Adjusting LSTM size: {_lstmHiddenSize} -> {newSize}");
                    _lstmHiddenSize = newSize;
                    _lstmSizeMismatchDetected = false;

                    // Restart simulation from beginning
                    _allPositionsX.Clear();
                    _allPositionsY.Clear();
                    _allPositionsZ.Clear();
                    _allEnergies.Clear();
                    _particleBoundaries.Clear();
                    allPathLengths.Clear();
                    allFinalDepths.Clear();
                    allLateralSpreads.Clear();
                    allMeanScatterAngles.Clear();
                    allStepCounts.Clear();
                    boundaryExits = 0;
                    totalSteps = 0;
                    failedParticles = 0;
                    p = -1; // Will become 0 after p++

                    yield return null;
                    continue;
                }

                // Mark particle boundary
                _particleBoundaries.Add(_allPositionsX.Count);

                // Simulate single particle with error handling
                ParticleResult result;
                bool needsLSTMFallback = false;

                try
                {
                    result = SimulateSingleParticle(p < 3); // Pass verbose flag for first 3 particles
                }
                catch (System.Exception e)
                {
                    failedParticles++;

                    // Check if this is LSTM batch size incompatibility (e.g., "Expected: 64 == 4096")
                    // Only try fallback once - if we already tried MLP and it still fails, give up
                    if (_isLSTM && !_triedMLPFallback &&
                        (e.Message.Contains("4096") || e.Message.Contains("1024") ||
                         (e.Message.Contains("Expected") && e.Message.Contains("=="))))
                    {
                        Debug.LogError("[TrainedAgentVisualizer] ================================================");
                        Debug.LogError("[TrainedAgentVisualizer] LSTM BATCH SIZE INCOMPATIBILITY DETECTED!");
                        Debug.LogError($"[TrainedAgentVisualizer] Error: {e.Message}");
                        Debug.LogError("[TrainedAgentVisualizer] ================================================");
                        Debug.LogWarning("[TrainedAgentVisualizer] AUTO-FALLBACK: Switching to MLP mode...");

                        needsLSTMFallback = true;
                        _triedMLPFallback = true; // Mark that we tried fallback
                    }
                    else if (_triedMLPFallback && e.Message.Contains("Expected"))
                    {
                        // MLP fallback also failed - this model is simply incompatible
                        Debug.LogError("[TrainedAgentVisualizer] ================================================");
                        Debug.LogError("[TrainedAgentVisualizer] LSTM MODEL INCOMPATIBLE WITH BARRACUDA!");
                        Debug.LogError("[TrainedAgentVisualizer] Both LSTM and MLP inference failed.");
                        Debug.LogError("[TrainedAgentVisualizer] This checkpoint cannot be visualized.");
                        Debug.LogError("[TrainedAgentVisualizer] ================================================");
                        _isSimulating = false;
                        _hasData = false;
                        yield break; // Abort this checkpoint entirely
                    }
                    else
                    {
                        // Check for LSTM size mismatch
                        if (_lstmSizeMismatchDetected)
                        {
                            continue;
                        }

                        if (p < 5 || failedParticles <= 3)
                        {
                            Debug.LogWarning($"[TrainedAgentVisualizer] Particle {p} failed: {e.Message}");
                        }

                        if (failedParticles >= maxFailedParticles)
                        {
                            Debug.LogError($"[TrainedAgentVisualizer] Too many failed particles ({failedParticles}). Stopping simulation.");
                            Debug.LogError($"[TrainedAgentVisualizer] Last error: {e.Message}");
                            break;
                        }
                    }

                    // Skip this particle and continue
                    continue;
                }

                // Handle LSTM fallback outside of catch block (yield not allowed in catch)
                if (needsLSTMFallback)
                {
                    _isLSTM = false;
                    p = -1; // Restart from beginning
                    failedParticles = 0;
                    _allPositionsX.Clear();
                    _allPositionsY.Clear();
                    _allPositionsZ.Clear();
                    _allEnergies.Clear();
                    _particleBoundaries.Clear();
                    allPathLengths.Clear();
                    allFinalDepths.Clear();
                    allLateralSpreads.Clear();
                    allMeanScatterAngles.Clear();
                    allStepCounts.Clear();
                    boundaryExits = 0;
                    totalSteps = 0;

                    Debug.LogWarning("[TrainedAgentVisualizer] Restarting simulation in MLP mode...");
                    yield return null;
                    continue;
                }

                // Log first few particles for debugging
                if (p < 5)
                {
                    Debug.Log($"[TrainedAgentVisualizer] Particle[{p}]: {result.StepCount} steps, " +
                             $"depth={result.FinalDepth:F3}cm, lateral={result.LateralSpread:F3}cm, " +
                             $"pathLen={result.PathLength:F3}cm, exitBoundary={result.ExitedBoundary}");

                    // Log start and end positions
                    if (result.Positions.Count > 0)
                    {
                        Vector3 start = result.Positions[0];
                        Vector3 end = result.Positions[result.Positions.Count - 1];
                        Debug.Log($"  Start: ({start.x:F3}, {start.y:F3}, {start.z:F3})");
                        Debug.Log($"  End:   ({end.x:F3}, {end.y:F3}, {end.z:F3})");
                    }
                }

                // Periodic worker recreation for LSTM to prevent memory accumulation
                if (_isLSTM && LSTMWorkerRecreateInterval > 0 &&
                    p > 0 && p % LSTMWorkerRecreateInterval == 0)
                {
                    Debug.Log($"[TrainedAgentVisualizer] Recreating LSTM worker at particle {p} to prevent memory issues...");
                    DisposeWorker();
                    System.GC.Collect();

                    // Re-initialize with same settings
                    if (!InitializeNeuralNetwork())
                    {
                        Debug.LogError("[TrainedAgentVisualizer] Failed to recreate worker!");
                        break;
                    }

                    yield return null;
                }

                // Store trajectory data
                for (int i = 0; i < result.Positions.Count && totalSteps < MaxVisualizationSteps; i++)
                {
                    _allPositionsX.Add(result.Positions[i].x);
                    _allPositionsY.Add(result.Positions[i].y);
                    _allPositionsZ.Add(result.Positions[i].z);
                    _allEnergies.Add(result.Energies[i]);
                    totalSteps++;
                }

                // Accumulate statistics
                allPathLengths.Add(result.PathLength);
                allFinalDepths.Add(result.FinalDepth);
                allLateralSpreads.Add(result.LateralSpread);
                allMeanScatterAngles.Add(result.MeanScatterAngle);
                allStepCounts.Add(result.StepCount);
                if (result.ExitedBoundary) boundaryExits++;

                // Progress logging (more frequent for debugging)
                if (ShowProgress && ((p + 1) % ProgressLogInterval == 0 || p == NumParticles - 1))
                {
                    float elapsed = sw.ElapsedMilliseconds / 1000f;
                    float rate = (p + 1) / Mathf.Max(0.001f, elapsed);
                    Debug.Log($"[TrainedAgentVisualizer] Progress: {p + 1}/{NumParticles} " +
                             $"({rate:F0} particles/sec, {totalSteps} total steps)");
                    yield return null;
                }

                // Yield periodically to prevent freezing
                // Use optimized frequency if enabled
                int yieldInterval = ReduceYieldFrequency ? YieldEveryNParticles : 100;
                if (p % yieldInterval == 0)
                {
                    yield return null;
                }
            }

            sw.Stop();
            Debug.Log("-------------------------------------------------------");
            Debug.Log($"[TrainedAgentVisualizer] SIMULATION COMPLETE");
            Debug.Log($"[TrainedAgentVisualizer] Particles: {allPathLengths.Count} successful, {failedParticles} failed");
            Debug.Log($"[TrainedAgentVisualizer] Time: {sw.ElapsedMilliseconds}ms");
            Debug.Log($"[TrainedAgentVisualizer] Total steps collected: {totalSteps}");
            Debug.Log($"[TrainedAgentVisualizer] Boundary exits: {boundaryExits}");
            Debug.Log("-------------------------------------------------------");

            yield return null;

            // Calculate statistics
            _statistics = CalculateStatistics(allPathLengths, allFinalDepths, allLateralSpreads,
                                               allMeanScatterAngles, allStepCounts, boundaryExits);
            Debug.Log(_statistics.ToString());

            yield return null;

            // Analyze coordinate ranges
            AnalyzeCoordinateRanges();

            yield return null;

            // Build visualization
            Debug.Log($"[TrainedAgentVisualizer] Building visualization: Mode={Mode}, Points={_allPositionsX.Count}");

            try
            {
                switch (Mode)
                {
                    case VisualizationMode.PointCloud:
                        BuildPointCloudVisualization();
                        break;
                    case VisualizationMode.LineSegments:
                        BuildLineSegmentsVisualization();
                        break;
                    case VisualizationMode.DensityTexture:
                        BuildDensityTextureVisualization();
                        break;
                }
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[TrainedAgentVisualizer] Exception building visualization: {e.Message}\n{e.StackTrace}");
            }

            _hasData = true;
            _isSimulating = false;

            Debug.Log("[TrainedAgentVisualizer] Visualization complete!");
        }

        // ====================================================================
        // NEURAL NETWORK SETUP
        // ====================================================================

        private bool InitializeNeuralNetwork()
        {
            Debug.Log("=======================================================");
            Debug.Log("[TrainedAgentVisualizer] INITIALIZING NEURAL NETWORK");
            Debug.Log("=======================================================");

            try
            {
                DisposeWorker();

                if (TrainedModel == null)
                {
                    Debug.LogError("[TrainedAgentVisualizer] TrainedModel is NULL!");
                    return false;
                }

                Debug.Log($"[TrainedAgentVisualizer] Loading model: {TrainedModel.name}");
                _model = ModelLoader.Load(TrainedModel);
                Debug.Log($"[TrainedAgentVisualizer] Model loaded successfully");

                // Reset LSTM state
                _isLSTM = false;
                _recurrentInName = null;
                _recurrentOutName = null;

                // Find input names and detect LSTM
                Debug.Log("[TrainedAgentVisualizer] --- Model Inputs ---");
                _inputName = null;
                int inputCount = 0;
                foreach (var input in _model.inputs)
                {
                    string shapeStr = string.Join(", ", input.shape);
                    Debug.Log($"  Input[{inputCount}]: name='{input.name}', shape=[{shapeStr}]");

                    // Check for observation input
                    if (input.name.Contains("obs"))
                    {
                        _inputName = input.name;
                    }

                    // Check for LSTM recurrent input
                    // ML-Agents uses exactly "recurrent_in" for LSTM memory input
                    // Don't trigger on generic "hidden" which could be regular layer names
                    if (input.name == "recurrent_in" || input.name.StartsWith("recurrent_in"))
                    {
                        _isLSTM = true;
                        _recurrentInName = input.name;

                        // Extract hidden size from shape
                        // ML-Agents LSTM input shape is typically [-1, memory_size] or [batch, memory_size]
                        // where -1 means dynamic batch, and memory_size = hidden_units * 2
                        Debug.Log($"  [LSTM] Full shape array: [{string.Join(", ", input.shape)}]");

                        // Find the memory_size dimension (last positive dimension)
                        _lstmHiddenSize = 128; // Default fallback

                        for (int i = input.shape.Length - 1; i >= 0; i--)
                        {
                            int dim = input.shape[i];
                            // Skip dynamic dimensions (-1) and batch dimension (1)
                            if (dim > 1)
                            {
                                _lstmHiddenSize = dim;
                                Debug.Log($"  [LSTM] Found memory_size={dim} at dimension {i}");
                                break;
                            }
                        }

                        // If no valid dimension found, try first dimension if it's positive
                        if (_lstmHiddenSize == 128)
                        {
                            foreach (int dim in input.shape)
                            {
                                if (dim > 1)
                                {
                                    _lstmHiddenSize = dim;
                                    Debug.Log($"  [LSTM] Fallback: using memory_size={dim}");
                                    break;
                                }
                            }
                        }

                        Debug.Log($"  [LSTM] Detected recurrent input: {_recurrentInName}, memory_size={_lstmHiddenSize}");
                    }

                    inputCount++;
                }

                // Find output names
                Debug.Log("[TrainedAgentVisualizer] --- Model Outputs ---");
                _outputName = null;
                int outputCount = 0;
                foreach (var output in _model.outputs)
                {
                    Debug.Log($"  Output[{outputCount}]: name='{output}'");

                    // Check for action output
                    if (output.Contains("continuous") || output.Contains("action"))
                    {
                        _outputName = output;
                    }

                    // Check for LSTM recurrent output
                    // ML-Agents uses exactly "recurrent_out" for LSTM memory output
                    if (output == "recurrent_out" || output.StartsWith("recurrent_out"))
                    {
                        _recurrentOutName = output;
                        Debug.Log($"  [LSTM] Detected recurrent output: {_recurrentOutName}");
                    }

                    outputCount++;
                }

                // Fallback: if no specific names found, use first input/output
                if (_inputName == null && _model.inputs.Count > 0)
                {
                    _inputName = _model.inputs[0].name;
                }
                if (_outputName == null && _model.outputs.Count > 0)
                {
                    _outputName = _model.outputs[0];
                }

                if (_inputName == null || _outputName == null)
                {
                    Debug.LogError($"[TrainedAgentVisualizer] Could not find tensor names! input={_inputName}, output={_outputName}");
                    return false;
                }

                Debug.Log($"[TrainedAgentVisualizer] Selected input: '{_inputName}'");
                Debug.Log($"[TrainedAgentVisualizer] Selected output: '{_outputName}'");
                Debug.Log($"[TrainedAgentVisualizer] Is LSTM model: {_isLSTM}");

                if (_isLSTM)
                {
                    // Check if user wants to force disable LSTM
                    if (ForceDisableLSTM)
                    {
                        Debug.LogWarning($"[TrainedAgentVisualizer] LSTM detected but ForceDisableLSTM=true. Using MLP inference.");
                        Debug.LogWarning($"[TrainedAgentVisualizer] Results will be approximate (no temporal memory).");
                        _isLSTM = false;
                    }
                    else
                    {
                        // Apply manual override if set
                        if (LSTMMemorySizeOverride > 0)
                        {
                            Debug.Log($"[TrainedAgentVisualizer] Using LSTM memory size OVERRIDE: {LSTMMemorySizeOverride} (auto-detected was {_lstmHiddenSize})");
                            _lstmHiddenSize = LSTMMemorySizeOverride;
                        }

                        Debug.Log($"[TrainedAgentVisualizer] === LSTM CONFIGURATION ===");
                        Debug.Log($"[TrainedAgentVisualizer] recurrent_in: '{_recurrentInName}'");
                        Debug.Log($"[TrainedAgentVisualizer] recurrent_out: '{_recurrentOutName}'");
                        Debug.Log($"[TrainedAgentVisualizer] memory_size: {_lstmHiddenSize}");
                        Debug.Log($"[TrainedAgentVisualizer] ========================");
                        Debug.LogWarning("[TrainedAgentVisualizer] LSTM WARNING: If you get 'Expected: 64 == 4096' errors,");
                        Debug.LogWarning("[TrainedAgentVisualizer] this is a known ML-Agents + Barracuda LSTM incompatibility.");
                        Debug.LogWarning("[TrainedAgentVisualizer] The batch_size from training (4096) is embedded in the ONNX model.");
                        Debug.LogWarning("[TrainedAgentVisualizer] SOLUTIONS:");
                        Debug.LogWarning("[TrainedAgentVisualizer]   1. Set ForceDisableLSTM=true (uses MLP inference, approximate results)");
                        Debug.LogWarning("[TrainedAgentVisualizer]   2. Retrain with batch_size=64 in YAML config");
                        Debug.LogWarning("[TrainedAgentVisualizer]   3. Use PPO or SAC without LSTM for visualization");
                    }
                }

                // Create worker
                // For LSTM models, try different backends in order of likelihood to work
                try
                {
                    if (_isLSTM)
                    {
                        // Try CSharpBurst first - often works better with LSTM shapes
                        try
                        {
                            _worker = WorkerFactory.CreateWorker(WorkerFactory.Type.CSharpBurst, _model);
                            Debug.Log("[TrainedAgentVisualizer] Worker created (CSharpBurst - recommended for LSTM)");
                        }
                        catch (System.Exception e1)
                        {
                            Debug.LogWarning($"[TrainedAgentVisualizer] CSharpBurst failed: {e1.Message}");
                            try
                            {
                                _worker = WorkerFactory.CreateWorker(WorkerFactory.Type.CSharpRef, _model);
                                Debug.Log("[TrainedAgentVisualizer] Worker created (CSharpRef fallback)");
                            }
                            catch (System.Exception e2)
                            {
                                Debug.LogWarning($"[TrainedAgentVisualizer] CSharpRef failed: {e2.Message}");
                                _worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, _model);
                                Debug.Log("[TrainedAgentVisualizer] Worker created (ComputePrecompiled last resort)");
                            }
                        }
                    }
                    else
                    {
                        _worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, _model);
                        Debug.Log("[TrainedAgentVisualizer] Worker created (ComputePrecompiled)");
                    }
                }
                catch (System.Exception workerEx)
                {
                    Debug.LogWarning($"[TrainedAgentVisualizer] All backends failed: {workerEx.Message}");
                    _worker = WorkerFactory.CreateWorker(WorkerFactory.Type.CSharpRef, _model);
                    Debug.Log("[TrainedAgentVisualizer] Worker created (CSharpRef emergency fallback)");
                }

                Debug.Log("[TrainedAgentVisualizer] Neural network initialization COMPLETE");
                Debug.Log("=======================================================");
                return true;
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[TrainedAgentVisualizer] FAILED to load model: {e.Message}");
                Debug.LogError($"[TrainedAgentVisualizer] Stack trace: {e.StackTrace}");
                return false;
            }
        }

        // ====================================================================
        // SINGLE PARTICLE SIMULATION
        // ====================================================================

        private struct ParticleResult
        {
            public List<Vector3> Positions;
            public List<float> Energies;
            public float PathLength;
            public float FinalDepth;
            public float LateralSpread;
            public float MeanScatterAngle;
            public int StepCount;
            public bool ExitedBoundary;
        }

        private ParticleResult SimulateSingleParticle(bool verbose = false)
        {
            ParticleResult result = new ParticleResult
            {
                Positions = new List<Vector3>(MaxStepsPerParticle),
                Energies = new List<float>(MaxStepsPerParticle)
            };

            // Initialize particle state
            Vector3 position = ElectronPhysics.GetInitialPosition();
            Vector3 direction = ElectronPhysics.GetInitialDirection();
            float energy = ElectronPhysics.INITIAL_ENERGY;
            float pathLength = 0f;
            List<float> scatterAngles = new List<float>();

            if (verbose)
            {
                Debug.Log($"    [Particle] Initial: pos=({position.x:F3}, {position.y:F3}, {position.z:F3}), " +
                         $"dir=({direction.x:F3}, {direction.y:F3}, {direction.z:F3}), energy={energy:F3}");
            }

            result.Positions.Add(position);
            result.Energies.Add(energy);

            // LSTM hidden state (initialized to zeros at start of each particle)
            float[] lstmHiddenState = null;
            if (_isLSTM)
            {
                lstmHiddenState = new float[_lstmHiddenSize];
                // Initialize to zeros (default LSTM initial state)
                for (int i = 0; i < _lstmHiddenSize; i++)
                {
                    lstmHiddenState[i] = 0f;
                }
            }

            // Observation tensor (11 values as per ElectronAgentPhysics)
            float[] observations = new float[11];

            for (int step = 0; step < MaxStepsPerParticle; step++)
            {
                // Check termination
                if (energy <= 0.01f)
                {
                    if (verbose) Debug.Log($"    [Particle] Terminated at step {step}: energy depleted");
                    break;
                }

                // Build observations (matching ElectronAgentPhysics.CollectObservations)
                observations[0] = position.x / ElectronPhysics.PHANTOM_HALF_SIZE;
                observations[1] = position.y / ElectronPhysics.PHANTOM_HALF_SIZE;
                observations[2] = position.z / ElectronPhysics.PHANTOM_HALF_SIZE;
                observations[3] = direction.x;
                observations[4] = direction.y;
                observations[5] = direction.z;
                observations[6] = energy / ElectronPhysics.INITIAL_ENERGY;
                observations[7] = (float)step / MaxStepsPerParticle;

                // Recent scattering statistics
                float recentMean = 0f;
                float recentVariance = 0f;
                int lookback = Mathf.Min(10, scatterAngles.Count);
                if (lookback > 0)
                {
                    for (int i = scatterAngles.Count - lookback; i < scatterAngles.Count; i++)
                        recentMean += scatterAngles[i];
                    recentMean /= lookback;

                    for (int i = scatterAngles.Count - lookback; i < scatterAngles.Count; i++)
                    {
                        float diff = scatterAngles[i] - recentMean;
                        recentVariance += diff * diff;
                    }
                    recentVariance /= lookback;
                }
                observations[8] = recentMean / 30f;
                observations[9] = Mathf.Sqrt(recentVariance) / 15f;

                // Spiral indicator (simplified - use 0 for batch)
                observations[10] = 0f;

                // Run inference
                float act0, act1, act2, act3;

                if (_isLSTM)
                {
                    // LSTM inference with hidden state
                    var inferenceResult = RunLSTMInference(observations, lstmHiddenState, verbose && step < 3);
                    act0 = inferenceResult.actions[0];
                    act1 = inferenceResult.actions[1];
                    act2 = inferenceResult.actions[2];
                    act3 = inferenceResult.actions[3];

                    // Update hidden state for next step
                    lstmHiddenState = inferenceResult.newHiddenState;
                }
                else
                {
                    // Standard MLP inference
                    var actions = RunMLPInference(observations, verbose && step < 3);
                    act0 = actions[0];
                    act1 = actions[1];
                    act2 = actions[2];
                    act3 = actions[3];
                }

                // Add stochastic noise
                float actionStdDev = ActionNoiseStdDev;
                act0 = Mathf.Clamp(act0 + SampleGaussian(0f, actionStdDev), -1f, 1f);
                act1 = Mathf.Clamp(act1 + SampleGaussian(0f, actionStdDev), -1f, 1f);
                act2 = Mathf.Clamp(act2 + SampleGaussian(0f, actionStdDev), -1f, 1f);
                act3 = Mathf.Clamp(act3 + SampleGaussian(0f, actionStdDev), -1f, 1f);

                // Apply action (matching ElectronAgentPhysics.ApplyPureAgentAction)
                Vector3 directionDelta = new Vector3(act0, act1, act2);
                float stepSizeFactor = (act3 + 1f) / 2f;
                float stepSize = Mathf.Lerp(MinStepSize, MaxStepSize, stepSizeFactor);

                Vector3 scaledDelta = directionDelta * MaxDirectionChange;
                Vector3 proposedDirection = (direction + scaledDelta).normalized;

                if (proposedDirection.magnitude < 0.001f)
                    proposedDirection = direction;

                float proposedAngle = Vector3.Angle(direction, proposedDirection);
                Vector3 newDirection;

                if (proposedAngle > MaxScatterAnglePerStep)
                {
                    Vector3 rotationAxis = Vector3.Cross(direction, proposedDirection).normalized;
                    if (rotationAxis.magnitude < 0.001f)
                    {
                        rotationAxis = Vector3.Cross(direction, Vector3.up).normalized;
                        if (rotationAxis.magnitude < 0.001f)
                            rotationAxis = Vector3.Cross(direction, Vector3.right).normalized;
                    }
                    Quaternion rotation = Quaternion.AngleAxis(MaxScatterAnglePerStep, rotationAxis);
                    newDirection = (rotation * direction).normalized;
                }
                else
                {
                    newDirection = proposedDirection;
                }

                float scatterAngle = Vector3.Angle(direction, newDirection);
                scatterAngles.Add(scatterAngle);

                direction = newDirection;
                position += direction * stepSize;
                pathLength += stepSize;

                // Energy loss with fluctuation
                float energyLoss = ElectronPhysics.CalculateEnergyLoss(energy, stepSize);
                energyLoss *= Random.Range(0.85f, 1.15f);
                energy -= energyLoss;
                energy = Mathf.Max(0f, energy);

                result.Positions.Add(position);
                result.Energies.Add(energy);

                // Check boundary exit (backward only)
                if (position.x < ElectronPhysics.PHANTOM_ENTRY_X - 0.5f)
                {
                    result.ExitedBoundary = true;
                    break;
                }
            }

            // Calculate final statistics
            result.PathLength = pathLength;
            result.FinalDepth = position.x - ElectronPhysics.PHANTOM_ENTRY_X;
            result.LateralSpread = Mathf.Sqrt(position.y * position.y + position.z * position.z);
            result.StepCount = result.Positions.Count;

            if (scatterAngles.Count > 0)
            {
                float sum = 0f;
                foreach (float a in scatterAngles) sum += a;
                result.MeanScatterAngle = sum / scatterAngles.Count;
            }

            return result;
        }

        // ====================================================================
        // INFERENCE METHODS
        // ====================================================================

        /// <summary>
        /// Run standard MLP inference (non-LSTM models).
        /// For LSTM models running in "MLP fallback mode", still provides recurrent_in with zeros.
        /// </summary>
        private float[] RunMLPInference(float[] observations, bool verbose = false)
        {
            float[] actions = new float[4];

            Tensor inputTensor = null;
            Tensor dummyHiddenTensor = null;

            try
            {
                inputTensor = new Tensor(1, observations.Length, observations);

                var inputs = new Dictionary<string, Tensor>();
                inputs[_inputName] = inputTensor;

                // If this is actually an LSTM model being run in "MLP mode" (ForceDisableLSTM=true),
                // we still need to provide the recurrent_in tensor (with zeros)
                // Otherwise Barracuda will throw "Global input is missing: recurrent_in"
                if (_recurrentInName != null)
                {
                    float[] zeroHidden = new float[_lstmHiddenSize];
                    dummyHiddenTensor = new Tensor(1, _lstmHiddenSize, zeroHidden);
                    inputs[_recurrentInName] = dummyHiddenTensor;

                    if (verbose)
                    {
                        Debug.Log($"    [MLP-Fallback] Providing dummy recurrent_in (zeros, size={_lstmHiddenSize})");
                    }
                }

                _worker.Execute(inputs);

                Tensor outputTensor = _worker.PeekOutput(_outputName);

                if (outputTensor.length >= 4)
                {
                    actions[0] = outputTensor[0];
                    actions[1] = outputTensor[1];
                    actions[2] = outputTensor[2];
                    actions[3] = outputTensor[3];

                    if (verbose)
                    {
                        Debug.Log($"    [MLP] Actions: ({actions[0]:F4}, {actions[1]:F4}, {actions[2]:F4}, {actions[3]:F4})");
                    }
                }
                else
                {
                    Debug.LogError($"[TrainedAgentVisualizer] Output tensor too small: {outputTensor.length}");
                }
            }
            finally
            {
                if (inputTensor != null) inputTensor.Dispose();
                if (dummyHiddenTensor != null) dummyHiddenTensor.Dispose();
            }

            return actions;
        }

        /// <summary>
        /// ZMODYFIKOWANA WERSJA: Obsługuje modele ze sztywnym Batch Size = 64
        /// Tworzy sztuczny batch danych, aby zadowolić asercje ONNX.
        /// </summary>
        private (float[] actions, float[] newHiddenState) RunLSTMInference(float[] observations, float[] hiddenState, bool verbose = false)
        {
            // --- FIX KONFIGURACJI ---
            // Model wymaga batch=64. Jeśli error mówi "Expected: XX == 1", wpisz tu XX.
            const int FIXED_BATCH_SIZE = 64;
            // ------------------------

            float[] actions = new float[4];
            float[] newHiddenState = new float[_lstmHiddenSize];

            Tensor obsTensor = null;
            Tensor hiddenTensor = null;

            try
            {
                // KROK 1: Przygotuj duże tablice na sztuczny batch (64 wiersze)
                // Obserwacje: [64, 11] -> spłaszczone do tablicy 64 * 11
                float[] batchedObservations = new float[FIXED_BATCH_SIZE * observations.Length];

                // Pamięć: [64, 64] -> spłaszczone do tablicy 64 * 64
                float[] batchedHidden = new float[FIXED_BATCH_SIZE * _lstmHiddenSize];

                // KROK 2: Skopiuj prawdziwe dane agenta tylko do pierwszego wiersza (index 0)
                // Reszta tablicy pozostaje zerami (to są "martwe" agenty)
                System.Array.Copy(observations, 0, batchedObservations, 0, observations.Length);
                System.Array.Copy(hiddenState, 0, batchedHidden, 0, _lstmHiddenSize);

                // KROK 3: Utwórz Tensory o wymiarze [Batch, Channels]
                // Zamiast (1, length) dajemy (FIXED_BATCH_SIZE, length)
                obsTensor = new Tensor(FIXED_BATCH_SIZE, observations.Length, batchedObservations);
                hiddenTensor = new Tensor(FIXED_BATCH_SIZE, _lstmHiddenSize, batchedHidden);

                if (verbose)
                {
                    Debug.Log($"[LSTM-FIX] Obs tensor shape: {obsTensor.shape}");
                    Debug.Log($"[LSTM-FIX] Hidden tensor shape: {hiddenTensor.shape}");
                }

                var inputs = new Dictionary<string, Tensor>();
                inputs[_inputName] = obsTensor;
                inputs[_recurrentInName] = hiddenTensor;

                _worker.Execute(inputs);

                // KROK 4: Odbierz wyniki
                Tensor outputTensor = _worker.PeekOutput(_outputName);

                // Pobieramy akcje tylko dla pierwszego agenta (offset 0)
                // Tensor outputu ma kształt [64, 4], więc pierwsze 4 floaty to nasz agent
                if (outputTensor.length >= 4)
                {
                    actions[0] = outputTensor[0];
                    actions[1] = outputTensor[1];
                    actions[2] = outputTensor[2];
                    actions[3] = outputTensor[3];
                }

                // Pobieramy nową pamięć
                if (_recurrentOutName != null)
                {
                    Tensor hiddenOutTensor = _worker.PeekOutput(_recurrentOutName);

                    // Znów, bierzemy tylko pierwszy wiersz pamięci
                    int copyLength = Mathf.Min(_lstmHiddenSize, hiddenOutTensor.length);
                    for (int i = 0; i < copyLength; i++)
                    {
                        newHiddenState[i] = hiddenOutTensor[i];
                    }
                }
                else
                {
                    System.Array.Copy(hiddenState, newHiddenState, _lstmHiddenSize);
                }
            }
            catch (System.Exception e)
            {
                // Ignorujemy błąd Assert, jeśli mimo to udało się pobrać dane (czasem Barracuda rzuca false positive)
                if (!e.Message.Contains("Assertion failure"))
                {
                    Debug.LogWarning($"[LSTM] Inference error: {e.Message}");
                    throw;
                }
            }
            finally
            {
                if (obsTensor != null) obsTensor.Dispose();
                if (hiddenTensor != null) hiddenTensor.Dispose();
            }

            return (actions, newHiddenState);
        }

        // Flag for auto-adjustment
        private bool _lstmSizeMismatchDetected = false;

        // Flag to prevent infinite fallback loop
        private bool _triedMLPFallback = false;

        /// <summary>
        /// Sample from Gaussian distribution using Box-Muller transform.
        /// </summary>
        private float SampleGaussian(float mean, float stdDev)
        {
            float u1 = Random.Range(0.0001f, 1f);
            float u2 = Random.Range(0.0001f, 1f);
            float randStdNormal = Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Sin(2f * Mathf.PI * u2);
            return mean + stdDev * randStdNormal;
        }

        // ====================================================================
        // STATISTICS CALCULATION
        // ====================================================================

        private TrainedAgentStatistics CalculateStatistics(
            List<float> pathLengths,
            List<float> finalDepths,
            List<float> lateralSpreads,
            List<float> meanScatterAngles,
            List<int> stepCounts,
            int boundaryExits)
        {
            var stats = new TrainedAgentStatistics();
            stats.NumParticles = pathLengths.Count;
            stats.BoundaryExits = boundaryExits;

            if (pathLengths.Count == 0) return stats;

            // Path length statistics
            stats.MeanPathLength = CalculateMean(pathLengths);
            stats.StdPathLength = CalculateStdDev(pathLengths, stats.MeanPathLength);

            // Penetration depth statistics
            stats.MeanPenetrationDepth = CalculateMean(finalDepths);
            stats.StdPenetrationDepth = CalculateStdDev(finalDepths, stats.MeanPenetrationDepth);

            // Lateral spread statistics
            stats.MeanLateralSpread = CalculateMean(lateralSpreads);
            stats.StdLateralSpread = CalculateStdDev(lateralSpreads, stats.MeanLateralSpread);

            // Scattering angle statistics
            stats.MeanScatterAngle = CalculateMean(meanScatterAngles);
            stats.StdScatterAngle = CalculateStdDev(meanScatterAngles, stats.MeanScatterAngle);

            // Step count statistics
            List<float> stepCountsF = new List<float>();
            foreach (int s in stepCounts) stepCountsF.Add(s);
            stats.MeanStepCount = CalculateMean(stepCountsF);

            return stats;
        }

        private float CalculateMean(List<float> values)
        {
            if (values.Count == 0) return 0f;
            float sum = 0f;
            foreach (float v in values) sum += v;
            return sum / values.Count;
        }

        private float CalculateStdDev(List<float> values, float mean)
        {
            if (values.Count == 0) return 0f;
            float variance = 0f;
            foreach (float v in values)
            {
                float diff = v - mean;
                variance += diff * diff;
            }
            return Mathf.Sqrt(variance / values.Count);
        }

        // ====================================================================
        // COORDINATE ANALYSIS
        // ====================================================================

        private void AnalyzeCoordinateRanges()
        {
            if (_allPositionsX.Count == 0) return;

            Vector3 min = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            Vector3 max = new Vector3(float.MinValue, float.MinValue, float.MinValue);

            for (int i = 0; i < _allPositionsX.Count; i++)
            {
                min.x = Mathf.Min(min.x, _allPositionsX[i]);
                min.y = Mathf.Min(min.y, _allPositionsY[i]);
                min.z = Mathf.Min(min.z, _allPositionsZ[i]);

                max.x = Mathf.Max(max.x, _allPositionsX[i]);
                max.y = Mathf.Max(max.y, _allPositionsY[i]);
                max.z = Mathf.Max(max.z, _allPositionsZ[i]);
            }

            Debug.Log($"[TrainedAgentVisualizer] Coordinate ranges (cm):");
            Debug.Log($"  X: [{min.x:F3}, {max.x:F3}] (range: {max.x - min.x:F3})");
            Debug.Log($"  Y: [{min.y:F3}, {max.y:F3}] (range: {max.y - min.y:F3})");
            Debug.Log($"  Z: [{min.z:F3}, {max.z:F3}] (range: {max.z - min.z:F3})");
        }

        // ====================================================================
        // POINT CLOUD VISUALIZATION
        // ====================================================================

        private void BuildPointCloudVisualization()
        {
            if (_visualizationObject != null)
                Destroy(_visualizationObject);

            _visualizationObject = new GameObject("TrainedAgent_PointCloud");
            _visualizationObject.transform.SetParent(transform);
            _visualizationObject.transform.localPosition = Vector3.zero;

            _pointMesh = new Mesh();
            _pointMesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;

            int vertexCount = _allPositionsX.Count;
            Vector3[] vertices = new Vector3[vertexCount];
            Color[] colors = new Color[vertexCount];
            int[] indices = new int[vertexCount];

            for (int i = 0; i < vertexCount; i++)
            {
                vertices[i] = new Vector3(_allPositionsX[i], _allPositionsY[i], _allPositionsZ[i]);
                float energyFraction = _allEnergies[i] / ElectronPhysics.INITIAL_ENERGY;
                colors[i] = GetEnergyColor(energyFraction);
                indices[i] = i;
            }

            _pointMesh.vertices = vertices;
            _pointMesh.colors = colors;
            _pointMesh.SetIndices(indices, MeshTopology.Points, 0);

            _pointMaterial = CreatePointMaterial();

            MeshFilter mf = _visualizationObject.AddComponent<MeshFilter>();
            mf.mesh = _pointMesh;

            MeshRenderer mr = _visualizationObject.AddComponent<MeshRenderer>();
            mr.material = _pointMaterial;

            Debug.Log($"[TrainedAgentVisualizer] Point cloud created: {vertexCount} points");
        }

        // ====================================================================
        // LINE SEGMENTS VISUALIZATION
        // ====================================================================

        private void BuildLineSegmentsVisualization()
        {
            if (_visualizationObject != null)
                Destroy(_visualizationObject);

            _visualizationObject = new GameObject("TrainedAgent_LineSegments");
            _visualizationObject.transform.SetParent(transform);
            _visualizationObject.transform.localPosition = Vector3.zero;

            _pointMesh = new Mesh();
            _pointMesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;

            int vertexCount = _allPositionsX.Count;
            Vector3[] vertices = new Vector3[vertexCount];
            Color[] colors = new Color[vertexCount];

            for (int i = 0; i < vertexCount; i++)
            {
                vertices[i] = new Vector3(_allPositionsX[i], _allPositionsY[i], _allPositionsZ[i]);
                float energyFraction = _allEnergies[i] / ElectronPhysics.INITIAL_ENERGY;
                colors[i] = GetEnergyColor(energyFraction);
            }

            // Build line indices (connect consecutive steps within each particle)
            List<int> lineIndices = new List<int>();
            HashSet<int> boundarySet = new HashSet<int>(_particleBoundaries);

            for (int i = 0; i < vertexCount - 1; i++)
            {
                if (!boundarySet.Contains(i + 1))
                {
                    lineIndices.Add(i);
                    lineIndices.Add(i + 1);
                }
            }

            _pointMesh.vertices = vertices;
            _pointMesh.colors = colors;
            _pointMesh.SetIndices(lineIndices.ToArray(), MeshTopology.Lines, 0);

            _pointMaterial = CreateLineMaterial();

            MeshFilter mf = _visualizationObject.AddComponent<MeshFilter>();
            mf.mesh = _pointMesh;

            MeshRenderer mr = _visualizationObject.AddComponent<MeshRenderer>();
            mr.material = _pointMaterial;

            Debug.Log($"[TrainedAgentVisualizer] Line segments: {vertexCount} vertices, {lineIndices.Count / 2} lines");
        }

        // ====================================================================
        // DENSITY TEXTURE VISUALIZATION
        // ====================================================================

        private void BuildDensityTextureVisualization()
        {
            _densityMap = new float[TextureResolution, TextureResolution];
            _energyMap = new float[TextureResolution, TextureResolution];

            float phantomHalfSize = ElectronPhysics.PHANTOM_HALF_SIZE;

            float minDepth = -phantomHalfSize;
            float maxDepth = phantomHalfSize;
            float minLateral = -phantomHalfSize;
            float maxLateral = phantomHalfSize;

            float depthRange = maxDepth - minDepth;
            float lateralRange = maxLateral - minLateral;

            Debug.Log($"[TrainedAgentVisualizer] === DENSITY MAP DEBUG ===");
            Debug.Log($"[TrainedAgentVisualizer] Total points: {_allPositionsX.Count}");
            Debug.Log($"[TrainedAgentVisualizer] Using FIXED phantom bounds (10x10 cm):");
            Debug.Log($"  Depth (X): [{minDepth:F1}, {maxDepth:F1}] cm");
            Debug.Log($"  Lateral (Y): [{minLateral:F1}, {maxLateral:F1}] cm");

            int outsideCount = 0;
            for (int i = 0; i < _allPositionsX.Count; i++)
            {
                float depth = _allPositionsX[i];
                float lateral = _allPositionsY[i];

                int binX = Mathf.FloorToInt(((depth - minDepth) / depthRange) * (TextureResolution - 1));
                int binY = Mathf.FloorToInt(((lateral - minLateral) / lateralRange) * (TextureResolution - 1));

                if (binX < 0 || binX >= TextureResolution || binY < 0 || binY >= TextureResolution)
                    outsideCount++;

                binX = Mathf.Clamp(binX, 0, TextureResolution - 1);
                binY = Mathf.Clamp(binY, 0, TextureResolution - 1);

                _densityMap[binX, binY] += 1.0f;
                _energyMap[binX, binY] += _allEnergies[i];
            }

            if (outsideCount > 0)
                Debug.Log($"[TrainedAgentVisualizer] Points outside phantom bounds: {outsideCount}");

            float maxDensity = 0;
            int nonZeroBins = 0;
            for (int x = 0; x < TextureResolution; x++)
            {
                for (int y = 0; y < TextureResolution; y++)
                {
                    if (_densityMap[x, y] > 0) nonZeroBins++;
                    maxDensity = Mathf.Max(maxDensity, _densityMap[x, y]);
                }
            }

            Debug.Log($"[TrainedAgentVisualizer] Max density per bin: {maxDensity}");
            Debug.Log($"[TrainedAgentVisualizer] Non-zero bins: {nonZeroBins} / {TextureResolution * TextureResolution}");

            _densityTexture = new Texture2D(TextureResolution, TextureResolution, TextureFormat.RGBA32, false);
            _densityTexture.filterMode = FilterMode.Bilinear;
            _densityTexture.wrapMode = TextureWrapMode.Clamp;

            for (int x = 0; x < TextureResolution; x++)
            {
                for (int y = 0; y < TextureResolution; y++)
                {
                    float density = _densityMap[x, y];
                    Color color;

                    if (density < 0.5f)
                    {
                        color = PhantomColor;
                    }
                    else
                    {
                        float normalizedDensity;
                        if (UseLogScale && maxDensity > 1)
                        {
                            normalizedDensity = Mathf.Log10(density + 1) / Mathf.Log10(maxDensity + 1);
                        }
                        else
                        {
                            normalizedDensity = density / Mathf.Max(1f, maxDensity);
                        }

                        float avgEnergy = _energyMap[x, y] / density;
                        float energyFraction = avgEnergy / ElectronPhysics.INITIAL_ENERGY;

                        color = GetEnergyColor(energyFraction);
                        color.a = Mathf.Lerp(MinAlpha, MaxAlpha, normalizedDensity);
                    }

                    _densityTexture.SetPixel(x, y, color);
                }
            }

            _densityTexture.Apply();

            // In FastMode, skip visual rendering - only texture is needed for PNG export
            if (!FastModeSkipRendering)
            {
                CreateDensityTextureQuad();
            }
            else
            {
                Debug.Log($"[TrainedAgentVisualizer] FastMode: Skipping Quad rendering (PNG only)");
            }

            Debug.Log($"[TrainedAgentVisualizer] Density texture created: {TextureResolution}x{TextureResolution}");
        }

        private void CreateDensityTextureQuad()
        {
            if (_visualizationObject != null)
                Destroy(_visualizationObject);

            _visualizationObject = GameObject.CreatePrimitive(PrimitiveType.Quad);
            _visualizationObject.name = "TrainedAgentDensityMap";
            _visualizationObject.transform.SetParent(transform);

            var collider = _visualizationObject.GetComponent<Collider>();
            if (collider != null) Destroy(collider);

            _visualizationObject.transform.localPosition = Vector3.zero;
            _visualizationObject.transform.localRotation = Quaternion.Euler(0, 90, 0);
            _visualizationObject.transform.localScale = new Vector3(10f, 10f, 1f);

            MeshRenderer mr = _visualizationObject.GetComponent<MeshRenderer>();
            Material mat = new Material(Shader.Find("Unlit/Transparent"));
            if (mat.shader == null)
                mat = new Material(Shader.Find("Sprites/Default"));
            mat.mainTexture = _densityTexture;
            mr.material = mat;

            Debug.Log($"[TrainedAgentVisualizer] Density texture quad: rotation=(0,90,0), scale=10x10");
        }

        /// <summary>
        /// Get the generated density texture (for PNG export).
        /// </summary>
        public Texture2D GetDensityTexture()
        {
            return _densityTexture;
        }

        /// <summary>
        /// Export density texture to PNG file directly (faster than screen capture).
        /// </summary>
        public void ExportDensityTexturePNG(string filePath)
        {
            if (_densityTexture == null)
            {
                Debug.LogError("[TrainedAgentVisualizer] No density texture to export!");
                return;
            }

            byte[] pngData = _densityTexture.EncodeToPNG();
            System.IO.File.WriteAllBytes(filePath, pngData);
            Debug.Log($"[TrainedAgentVisualizer] PNG exported: {filePath}");
        }

        // ====================================================================
        // MATERIAL HELPERS
        // ====================================================================

        private Material CreatePointMaterial()
        {
            Material mat = new Material(Shader.Find("Particles/Standard Unlit"));
            if (mat.shader == null)
                mat = new Material(Shader.Find("Sprites/Default"));

            mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            mat.SetInt("_ZWrite", 0);
            mat.renderQueue = 3000;

            return mat;
        }

        private Material CreateLineMaterial()
        {
            Material mat = new Material(Shader.Find("Sprites/Default"));
            mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            mat.SetInt("_ZWrite", 0);
            mat.renderQueue = 3000;

            return mat;
        }

        // ====================================================================
        // COLOR HELPERS
        // ====================================================================

        /// <summary>
        /// Get color based on energy fraction (0-1).
        /// Uses physics-standard gradient matching Geant4/CERN visualization:
        ///   Blue (high energy, 10 MeV) → Red (mid, 5 MeV) → White (low/zero)
        /// </summary>
        private Color GetEnergyColor(float energyFraction)
        {
            energyFraction = Mathf.Clamp01(energyFraction);

            if (energyFraction > 0.5f)
            {
                float t = (energyFraction - 0.5f) * 2.0f;
                return Color.Lerp(MidEnergyColor, HighEnergyColor, t);
            }
            else
            {
                float t = energyFraction * 2.0f;
                return Color.Lerp(LowEnergyColor, MidEnergyColor, t);
            }
        }

        // ====================================================================
        // PER-TRAJECTORY CSV EXPORT
        // ====================================================================

        /// <summary>
        /// Export per-trajectory data to CSV for Python analysis.
        /// Uses existing per-step data and boundary detection.
        /// </summary>
        public void ExportTrajectoriesToCSV(string outputFilePath)
        {
            if (!_hasData || _allPositionsX == null || _allPositionsX.Count == 0)
            {
                Debug.LogError("[TrainedAgentVisualizer] No data to export! Run simulation first.");
                return;
            }

            if (_particleBoundaries == null || _particleBoundaries.Count == 0)
            {
                Debug.LogError("[TrainedAgentVisualizer] No particle boundaries detected!");
                return;
            }

            Debug.Log($"[TrainedAgentVisualizer] Exporting {_particleBoundaries.Count} trajectories to CSV...");

            // Build trajectory list
            var trajectories = new System.Collections.Generic.List<TrajectoryRecord>();

            for (int t = 0; t < _particleBoundaries.Count; t++)
            {
                int startIdx = _particleBoundaries[t];
                int endIdx = (t < _particleBoundaries.Count - 1)
                    ? _particleBoundaries[t + 1] - 1
                    : _allPositionsX.Count - 1;

                if (endIdx <= startIdx)
                    continue;

                var record = CalculateTrajectoryMetricsForExport(t, startIdx, endIdx);
                trajectories.Add(record);
            }

            // Export to CSV
            ExportTrajectoryRecords(trajectories, outputFilePath);

            Debug.Log($"[TrainedAgentVisualizer] ✓ Exported {trajectories.Count} trajectories to: {outputFilePath}");
        }

        private struct TrajectoryRecord
        {
            public int TrajectoryID;
            public float PathLength;
            public float PenetrationDepth;
            public float LateralSpread;
            public float LateralY;
            public float LateralZ;
            public float MeanScatterAngle;
            public float FinalEnergy;
            public int NumSteps;
            public bool BoundaryExit;
        }

        private TrajectoryRecord CalculateTrajectoryMetricsForExport(int trajectoryID, int startIdx, int endIdx)
        {
            var record = new TrajectoryRecord();
            record.TrajectoryID = trajectoryID;
            record.NumSteps = endIdx - startIdx + 1;

            // Calculate path length (sum of step distances)
            float pathLength = 0f;
            for (int i = startIdx; i < endIdx; i++)
            {
                float dx = _allPositionsX[i + 1] - _allPositionsX[i];
                float dy = _allPositionsY[i + 1] - _allPositionsY[i];
                float dz = _allPositionsZ[i + 1] - _allPositionsZ[i];
                pathLength += Mathf.Sqrt(dx * dx + dy * dy + dz * dz);
            }
            record.PathLength = pathLength;

            // Penetration depth (final X coordinate - entry point)
            float finalX = _allPositionsX[endIdx];
            record.PenetrationDepth = finalX - ElectronPhysics.PHANTOM_ENTRY_X;

            // Final position (lateral spread at exit)
            float finalY = _allPositionsY[endIdx];
            float finalZ = _allPositionsZ[endIdx];

            record.LateralY = finalY;
            record.LateralZ = finalZ;
            record.LateralSpread = Mathf.Sqrt(finalY * finalY + finalZ * finalZ);

            // Mean scatter angle (average direction change)
            float sumAngles = 0f;
            int angleCount = 0;

            for (int i = startIdx; i < endIdx - 1; i++)
            {
                Vector3 dir1 = new Vector3(
                    _allPositionsX[i + 1] - _allPositionsX[i],
                    _allPositionsY[i + 1] - _allPositionsY[i],
                    _allPositionsZ[i + 1] - _allPositionsZ[i]
                ).normalized;

                Vector3 dir2 = new Vector3(
                    _allPositionsX[i + 2] - _allPositionsX[i + 1],
                    _allPositionsY[i + 2] - _allPositionsY[i + 1],
                    _allPositionsZ[i + 2] - _allPositionsZ[i + 1]
                ).normalized;

                float angle = Vector3.Angle(dir1, dir2);
                if (!float.IsNaN(angle) && angle > 0.1f)
                {
                    sumAngles += angle;
                    angleCount++;
                }
            }

            record.MeanScatterAngle = angleCount > 0 ? sumAngles / angleCount : 0f;

            // Final energy
            record.FinalEnergy = _allEnergies[endIdx];

            // Boundary exit detection (if exited backward)
            record.BoundaryExit = finalX < ElectronPhysics.PHANTOM_ENTRY_X - 0.5f;

            return record;
        }

        private void ExportTrajectoryRecords(System.Collections.Generic.List<TrajectoryRecord> trajectories, string filepath)
        {
            // Ensure directory exists
            string directory = System.IO.Path.GetDirectoryName(filepath);
            if (!string.IsNullOrEmpty(directory) && !System.IO.Directory.Exists(directory))
            {
                System.IO.Directory.CreateDirectory(directory);
            }

            var sb = new System.Text.StringBuilder();

            // Header - EXACT same format as Geant4 for consistency
            sb.AppendLine("TrajectoryID,PathLength,PenetrationDepth,LateralSpread,LateralY,LateralZ,MeanScatterAngle,FinalEnergy,NumSteps,BoundaryExit");

            // Data rows
            foreach (var t in trajectories)
            {
                sb.AppendLine($"{t.TrajectoryID}," +
                             $"{t.PathLength:F4}," +
                             $"{t.PenetrationDepth:F4}," +
                             $"{t.LateralSpread:F4}," +
                             $"{t.LateralY:F4}," +
                             $"{t.LateralZ:F4}," +
                             $"{t.MeanScatterAngle:F4}," +
                             $"{t.FinalEnergy:F4}," +
                             $"{t.NumSteps}," +
                             $"{(t.BoundaryExit ? "True" : "False")}");
            }

            System.IO.File.WriteAllText(filepath, sb.ToString());
        }

        // ====================================================================
        // DEBUG HELPERS
        // ====================================================================

        private void DrawCoordinateAxes()
        {
            float axisLength = 5.0f;
            Debug.DrawRay(Vector3.zero, Vector3.right * axisLength, Color.red, 1000f);
            Debug.DrawRay(Vector3.zero, Vector3.up * axisLength, Color.green, 1000f);
            Debug.DrawRay(Vector3.zero, Vector3.forward * axisLength, Color.blue, 1000f);
        }

        // ====================================================================
        // EDITOR BUTTONS
        // ====================================================================

#if UNITY_EDITOR
        [ContextMenu("Run Simulation")]
        private void EditorRunSimulation()
        {
            RunBatchSimulation();
        }

        [ContextMenu("Export Statistics")]
        private void EditorExportStatistics()
        {
            ExportStatistics();
        }

        [ContextMenu("Clear Visualization")]
        private void EditorClearVisualization()
        {
            ClearVisualization();
        }
#endif
    }

    // ====================================================================
    // STATISTICS STRUCT
    // ====================================================================

    [System.Serializable]
    public class TrainedAgentStatistics
    {
        public int NumParticles;
        public int BoundaryExits;

        public float MeanPathLength;
        public float StdPathLength;

        public float MeanPenetrationDepth;
        public float StdPenetrationDepth;

        public float MeanLateralSpread;
        public float StdLateralSpread;

        public float MeanScatterAngle;
        public float StdScatterAngle;

        public float MeanStepCount;

        public override string ToString()
        {
            return $"[TrainedAgent Statistics]\n" +
                   $"  Particles: {NumParticles} (boundary exits: {BoundaryExits})\n" +
                   $"  Path Length: {MeanPathLength:F3} ± {StdPathLength:F3} cm\n" +
                   $"  Penetration: {MeanPenetrationDepth:F3} ± {StdPenetrationDepth:F3} cm\n" +
                   $"  Lateral Spread: {MeanLateralSpread:F3} ± {StdLateralSpread:F3} cm\n" +
                   $"  Scatter Angle: {MeanScatterAngle:F2} ± {StdScatterAngle:F2}°\n" +
                   $"  Steps/Particle: {MeanStepCount:F1}";
        }

        public string ToDetailedString()
        {
            return $"Trained Agent Batch Simulation Statistics\n" +
                   $"==========================================\n\n" +
                   $"Simulation Parameters:\n" +
                   $"  Total Particles: {NumParticles}\n" +
                   $"  Boundary Exits: {BoundaryExits} ({100f * BoundaryExits / NumParticles:F2}%)\n\n" +
                   $"Trajectory Statistics:\n" +
                   $"  Path Length:\n" +
                   $"    Mean: {MeanPathLength:F4} cm\n" +
                   $"    Std:  {StdPathLength:F4} cm\n\n" +
                   $"  Penetration Depth (X-axis):\n" +
                   $"    Mean: {MeanPenetrationDepth:F4} cm\n" +
                   $"    Std:  {StdPenetrationDepth:F4} cm\n\n" +
                   $"  Lateral Spread (Y-Z plane):\n" +
                   $"    Mean: {MeanLateralSpread:F4} cm\n" +
                   $"    Std:  {StdLateralSpread:F4} cm\n\n" +
                   $"  Scattering Angle:\n" +
                   $"    Mean: {MeanScatterAngle:F4}°\n" +
                   $"    Std:  {StdScatterAngle:F4}°\n\n" +
                   $"  Steps per Particle: {MeanStepCount:F2}\n";
        }
    }
}