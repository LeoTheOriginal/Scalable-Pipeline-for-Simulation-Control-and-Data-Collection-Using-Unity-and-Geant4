using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using Core;

namespace Visualization
{
    /// <summary>
    /// Visualizer for Geant4 batch simulation trajectories.
    /// Renders cumulative trajectory density with energy-based coloring.
    /// 
    /// Energy gradient: Blue (high energy) → Red → White (energy → 0)
    /// 
    /// Coordinate system:
    /// - Geant4: Beam travels in +Z direction (depth into phantom)
    /// - Unity: Left-handed, need coordinate conversion
    /// - Density map: X (lateral spread) vs Z (depth)
    /// </summary>
    public class Geant4TrajectoryVisualizer : MonoBehaviour
    {
        // ====================================================================
        // INSPECTOR SETTINGS
        // ====================================================================

        [Header("Simulation Settings")]
        [Tooltip("Number of particles to simulate")]
        public int NumParticles = 100000;

        [Tooltip("Run simulation on Start")]
        public bool RunOnStart = false;

        [Tooltip("Maximum steps to visualize (memory limit)")]
        public int MaxVisualizationSteps = 10000000;

        [Header("Visualization Mode")]
        [Tooltip("Use point cloud (faster) or line rendering")]
        public VisualizationMode Mode = VisualizationMode.PointCloud;

        [Tooltip("Point/line size")]
        public float PointSize = 0.01f;

        [Header("Coordinate System")]
        [Tooltip("Beam direction in Geant4 coordinates")]
        public BeamDirection BeamAxis = BeamDirection.PositiveZ;

        [Header("Color Settings")]
        [Tooltip("High energy color (10 MeV)")]
        public Color HighEnergyColor = new Color(0.2f, 0.4f, 1.0f, 0.8f);

        [Tooltip("Mid energy color (5 MeV)")]
        public Color MidEnergyColor = new Color(1.0f, 0.3f, 0.3f, 0.7f);

        [Tooltip("Low energy color (approaching 0 MeV) - WHITE")]
        public Color LowEnergyColor = new Color(1.0f, 1.0f, 1.0f, 0.9f);

        [Tooltip("Background/phantom color")]
        public Color PhantomColor = new Color(0.1f, 0.1f, 0.3f, 0.5f);

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

        [Header("Statistics Export")]
        [Tooltip("Path for statistics file")]
        public string StatisticsFilePath = "geant4_statistics.txt";

        [Header("Debug")]
        public bool ShowProgress = false;
        public bool ShowCoordinateAxes = false;

        public enum VisualizationMode
        {
            PointCloud,
            LineSegments,
            DensityTexture
        }

        public enum BeamDirection
        {
            PositiveX,
            PositiveY,
            PositiveZ
        }

        // ====================================================================
        // RUNTIME STATE
        // ====================================================================

        private bool _isSimulating = false;
        private bool _hasData = false;
        private Geant4BatchStatistics _statistics;

        // Visualization data
        private float[] _positionsX;
        private float[] _positionsY;
        private float[] _positionsZ;
        private float[] _energies;
        private int _stepCount;

        // Mesh rendering
        private Mesh _pointMesh;
        private Material _pointMaterial;
        private GameObject _visualizationObject;

        // Density texture (lateral spread vs depth)
        private Texture2D _densityTexture;
        private float[,] _densityMap;
        private float[,] _energyMap;

        // Track particle boundaries for line segments
        private List<int> _particleBoundaries;

        // ====================================================================
        // PUBLIC API
        // ====================================================================

        /// <summary>
        /// Start batch simulation asynchronously.
        /// </summary>
        public void RunBatchSimulation()
        {
            if (_isSimulating)
            {
                Debug.LogWarning("[Geant4Visualizer] Simulation already running!");
                return;
            }

            StartCoroutine(RunSimulationCoroutine());
        }

        /// <summary>
        /// Get the computed statistics.
        /// </summary>
        public Geant4BatchStatistics GetStatistics()
        {
            return _statistics;
        }

        /// <summary>
        /// Export statistics to file.
        /// </summary>
        public void ExportStatistics()
        {
            if (!_hasData)
            {
                Debug.LogWarning("[Geant4Visualizer] No data to export!");
                return;
            }

            string fullPath = System.IO.Path.Combine(Application.dataPath, StatisticsFilePath);
            int result = Geant4Interface.ExportStatisticsToFile(fullPath);

            if (result == 1)
            {
                Debug.Log($"[Geant4Visualizer] Statistics exported to: {fullPath}");
            }
            else
            {
                Debug.LogError($"[Geant4Visualizer] Failed to export statistics!");
            }
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
            _stepCount = 0;
        }

        // ====================================================================
        // LIFECYCLE
        // ====================================================================

        void Start()
        {
            if (RunOnStart)
            {
                RunBatchSimulation();
            }

            if (ShowCoordinateAxes)
            {
                DrawCoordinateAxes();
            }
        }

        void OnDestroy()
        {
            ClearVisualization();

            if (_pointMaterial != null)
            {
                Destroy(_pointMaterial);
            }

            if (_pointMesh != null)
            {
                Destroy(_pointMesh);
            }

            if (_densityTexture != null)
            {
                Destroy(_densityTexture);
            }
        }

        // ====================================================================
        // SIMULATION COROUTINE
        // ====================================================================

        private IEnumerator RunSimulationCoroutine()
        {
            _isSimulating = true;
            _hasData = false;

            Debug.Log($"[Geant4Visualizer] Starting batch simulation: {NumParticles} particles...");

            // Initialize Geant4 if needed
            if (!Geant4Manager.Instance?.IsGeant4Available() ?? true)
            {
                Geant4Interface.InitGeant4();
                yield return null;
            }

            // Run batch simulation
            System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();

            int simulated = Geant4Interface.RunBatchSimulation(NumParticles, 0);

            sw.Stop();
            Debug.Log($"[Geant4Visualizer] Simulated {simulated} particles in {sw.ElapsedMilliseconds}ms");

            yield return null;

            // Get statistics
            float[] statsData = new float[24];
            Geant4Interface.GetBatchStatistics(statsData);
            _statistics = Geant4BatchStatistics.FromArray(statsData);

            Debug.Log(_statistics.ToString());

            yield return null;

            // Get trajectory data
            _stepCount = Geant4Interface.GetBatchStepCount();
            int stepsToLoad = Mathf.Min(_stepCount, MaxVisualizationSteps);

            Debug.Log($"[Geant4Visualizer] Loading {stepsToLoad} of {_stepCount} steps for visualization...");

            _positionsX = new float[stepsToLoad];
            _positionsY = new float[stepsToLoad];
            _positionsZ = new float[stepsToLoad];
            _energies = new float[stepsToLoad];

            int loaded = Geant4Interface.GetBatchTrajectoryData(
                _positionsX, _positionsY, _positionsZ, _energies, stepsToLoad);

            Debug.Log($"[Geant4Visualizer] Loaded {loaded} steps");

            // Analyze coordinate ranges for debugging
            AnalyzeCoordinateRanges();

            yield return null;

            // Build visualization
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

            _hasData = true;
            _isSimulating = false;

            Debug.Log("[Geant4Visualizer] Visualization complete!");
        }

        // ====================================================================
        // COORDINATE ANALYSIS
        // ====================================================================

        private void AnalyzeCoordinateRanges()
        {
            Vector3 min = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            Vector3 max = new Vector3(float.MinValue, float.MinValue, float.MinValue);

            for (int i = 0; i < _positionsX.Length; i++)
            {
                min.x = Mathf.Min(min.x, _positionsX[i]);
                min.y = Mathf.Min(min.y, _positionsY[i]);
                min.z = Mathf.Min(min.z, _positionsZ[i]);

                max.x = Mathf.Max(max.x, _positionsX[i]);
                max.y = Mathf.Max(max.y, _positionsY[i]);
                max.z = Mathf.Max(max.z, _positionsZ[i]);
            }

            Debug.Log($"[Geant4Visualizer] Coordinate ranges (cm):");
            Debug.Log($"  X: [{min.x:F3}, {max.x:F3}] (range: {max.x - min.x:F3})");
            Debug.Log($"  Y: [{min.y:F3}, {max.y:F3}] (range: {max.y - min.y:F3})");
            Debug.Log($"  Z: [{min.z:F3}, {max.z:F3}] (range: {max.z - min.z:F3})");

            // Determine which axis has largest range (likely beam direction)
            float rangeX = max.x - min.x;
            float rangeY = max.y - min.y;
            float rangeZ = max.z - min.z;

            string likelyBeamAxis = "Unknown";
            if (rangeZ > rangeX && rangeZ > rangeY)
                likelyBeamAxis = "Z";
            else if (rangeY > rangeX && rangeY > rangeZ)
                likelyBeamAxis = "Y";
            else if (rangeX > rangeY && rangeX > rangeZ)
                likelyBeamAxis = "X";

            Debug.Log($"  Likely beam direction: +{likelyBeamAxis} (largest range)");
        }

        // ====================================================================
        // POINT CLOUD VISUALIZATION
        // ====================================================================

        private void BuildPointCloudVisualization()
        {
            if (_visualizationObject != null)
            {
                Destroy(_visualizationObject);
            }

            _visualizationObject = new GameObject("Geant4Trajectories_PointCloud");
            _visualizationObject.transform.SetParent(transform);
            _visualizationObject.transform.localPosition = Vector3.zero;

            // Create mesh
            _pointMesh = new Mesh();
            _pointMesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;

            int vertexCount = _positionsX.Length;
            Vector3[] vertices = new Vector3[vertexCount];
            Color[] colors = new Color[vertexCount];
            int[] indices = new int[vertexCount];

            float initialEnergy = 10.0f;

            for (int i = 0; i < vertexCount; i++)
            {
                // Position (cm to Unity units - assuming 1:1 scale)
                vertices[i] = new Vector3(_positionsX[i], _positionsY[i], _positionsZ[i]);

                // Energy-based color
                float energyFraction = _energies[i] / initialEnergy;
                colors[i] = GetEnergyColor(energyFraction);

                indices[i] = i;
            }

            _pointMesh.vertices = vertices;
            _pointMesh.colors = colors;
            _pointMesh.SetIndices(indices, MeshTopology.Points, 0);

            // Create material
            _pointMaterial = CreatePointMaterial();

            // Add mesh components
            MeshFilter mf = _visualizationObject.AddComponent<MeshFilter>();
            mf.mesh = _pointMesh;

            MeshRenderer mr = _visualizationObject.AddComponent<MeshRenderer>();
            mr.material = _pointMaterial;

            Debug.Log($"[Geant4Visualizer] Point cloud created: {vertexCount} points");
        }

        // ====================================================================
        // LINE SEGMENTS VISUALIZATION
        // ====================================================================

        private void BuildLineSegmentsVisualization()
        {
            if (_visualizationObject != null)
            {
                Destroy(_visualizationObject);
            }

            _visualizationObject = new GameObject("Geant4Trajectories_LineSegments");
            _visualizationObject.transform.SetParent(transform);
            _visualizationObject.transform.localPosition = Vector3.zero;

            // Detect particle boundaries (assuming sequential steps per particle)
            DetectParticleBoundaries();

            // Create mesh
            _pointMesh = new Mesh();
            _pointMesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;

            int vertexCount = _positionsX.Length;
            Vector3[] vertices = new Vector3[vertexCount];
            Color[] colors = new Color[vertexCount];

            float initialEnergy = 10.0f;

            // Build vertices and colors
            for (int i = 0; i < vertexCount; i++)
            {
                vertices[i] = new Vector3(_positionsX[i], _positionsY[i], _positionsZ[i]);
                float energyFraction = _energies[i] / initialEnergy;
                colors[i] = GetEnergyColor(energyFraction);
            }

            // Build line indices (connect consecutive steps within each particle)
            List<int> lineIndices = new List<int>();

            for (int i = 0; i < vertexCount - 1; i++)
            {
                // Check if next point belongs to same particle
                bool isBoundary = _particleBoundaries.Contains(i + 1);

                if (!isBoundary)
                {
                    // Add line segment
                    lineIndices.Add(i);
                    lineIndices.Add(i + 1);
                }
            }

            _pointMesh.vertices = vertices;
            _pointMesh.colors = colors;
            _pointMesh.SetIndices(lineIndices.ToArray(), MeshTopology.Lines, 0);

            // Create material
            _pointMaterial = CreateLineMaterial();

            // Add mesh components
            MeshFilter mf = _visualizationObject.AddComponent<MeshFilter>();
            mf.mesh = _pointMesh;

            MeshRenderer mr = _visualizationObject.AddComponent<MeshRenderer>();
            mr.material = _pointMaterial;

            Debug.Log($"[Geant4Visualizer] Line segments created: {vertexCount} vertices, {lineIndices.Count / 2} lines");
        }

        /// <summary>
        /// Detect particle boundaries based on position discontinuities.
        /// Assumes consecutive steps belong to same particle until large jump occurs.
        /// </summary>
        private void DetectParticleBoundaries()
        {
            _particleBoundaries = new List<int>();
            _particleBoundaries.Add(0); // First particle starts at 0

            float maxStepSize = 0.5f; // cm - threshold for detecting new particle

            for (int i = 1; i < _positionsX.Length; i++)
            {
                float dx = _positionsX[i] - _positionsX[i - 1];
                float dy = _positionsY[i] - _positionsY[i - 1];
                float dz = _positionsZ[i] - _positionsZ[i - 1];
                float distance = Mathf.Sqrt(dx * dx + dy * dy + dz * dz);

                // If step is too large or energy resets to high value, it's a new particle
                bool isNewParticle = distance > maxStepSize ||
                                     (_energies[i] > _energies[i - 1] + 1.0f);

                if (isNewParticle)
                {
                    _particleBoundaries.Add(i);
                }
            }

            Debug.Log($"[Geant4Visualizer] Detected {_particleBoundaries.Count} particles");
        }

        // ====================================================================
        // DENSITY TEXTURE VISUALIZATION (Lateral vs Depth projection)
        // ====================================================================

        private void BuildDensityTextureVisualization()
        {
            // Initialize density and energy maps
            _densityMap = new float[TextureResolution, TextureResolution];
            _energyMap = new float[TextureResolution, TextureResolution];

            // Determine projection axes based on beam direction
            // For density map like in reference: lateral (horizontal) vs depth (vertical)

            float[] lateralPositions = new float[_positionsX.Length];
            float[] depthPositions = new float[_positionsX.Length];

            GetProjectionAxes(out lateralPositions, out depthPositions);

            // Find ranges
            float minLateral = float.MaxValue;
            float maxLateral = float.MinValue;
            float minDepth = float.MaxValue;
            float maxDepth = float.MinValue;

            for (int i = 0; i < lateralPositions.Length; i++)
            {
                minLateral = Mathf.Min(minLateral, lateralPositions[i]);
                maxLateral = Mathf.Max(maxLateral, lateralPositions[i]);
                minDepth = Mathf.Min(minDepth, depthPositions[i]);
                maxDepth = Mathf.Max(maxDepth, depthPositions[i]);
            }

            float lateralRange = maxLateral - minLateral;
            float depthRange = maxDepth - minDepth;

            Debug.Log($"[Geant4Visualizer] Density map ranges:");
            Debug.Log($"  Lateral: [{minLateral:F3}, {maxLateral:F3}] cm (range: {lateralRange:F3})");
            Debug.Log($"  Depth: [{minDepth:F3}, {maxDepth:F3}] cm (range: {depthRange:F3})");

            // Use symmetric range for lateral (centered on beam)
            float maxLateralAbs = Mathf.Max(Mathf.Abs(minLateral), Mathf.Abs(maxLateral));
            lateralRange = maxLateralAbs * 2;
            minLateral = -maxLateralAbs;

            // Accumulate density
            for (int i = 0; i < lateralPositions.Length; i++)
            {
                float lateral = lateralPositions[i];
                float depth = depthPositions[i];
                float energy = _energies[i];

                // Map to texture coordinates
                // Lateral: -maxLateralAbs to +maxLateralAbs → 0 to TextureResolution
                // Depth: minDepth to maxDepth → 0 to TextureResolution
                int binX = Mathf.FloorToInt(((lateral - minLateral) / lateralRange) * TextureResolution);
                int binY = Mathf.FloorToInt(((depth - minDepth) / depthRange) * TextureResolution);

                // Clamp to valid range
                binX = Mathf.Clamp(binX, 0, TextureResolution - 1);
                binY = Mathf.Clamp(binY, 0, TextureResolution - 1);

                _densityMap[binX, binY] += 1.0f;
                _energyMap[binX, binY] += energy;
            }

            // Find max density for normalization
            float maxDensity = 0;
            for (int x = 0; x < TextureResolution; x++)
            {
                for (int y = 0; y < TextureResolution; y++)
                {
                    if (_densityMap[x, y] > maxDensity)
                        maxDensity = _densityMap[x, y];
                }
            }

            Debug.Log($"[Geant4Visualizer] Max density per bin: {maxDensity}");

            // Create texture
            _densityTexture = new Texture2D(TextureResolution, TextureResolution, TextureFormat.RGBA32, false);
            _densityTexture.filterMode = FilterMode.Bilinear;

            for (int x = 0; x < TextureResolution; x++)
            {
                for (int y = 0; y < TextureResolution; y++)
                {
                    float density = _densityMap[x, y];

                    Color color;
                    if (density < 0.1f)
                    {
                        // Empty region - use phantom color
                        color = PhantomColor;
                    }
                    else
                    {
                        // Normalize density
                        float normalizedDensity;
                        if (UseLogScale && density > 0)
                        {
                            // Logarithmic scaling for better visualization
                            normalizedDensity = Mathf.Log10(density + 1) / Mathf.Log10(maxDensity + 1);
                        }
                        else
                        {
                            normalizedDensity = density / maxDensity;
                        }

                        // Average energy in this bin
                        float avgEnergy = _energyMap[x, y] / density;
                        float energyFraction = avgEnergy / 10.0f;

                        // Get color based on energy
                        color = GetEnergyColor(energyFraction);

                        // Modulate alpha by density
                        color.a = Mathf.Lerp(MinAlpha, MaxAlpha, normalizedDensity);
                    }

                    _densityTexture.SetPixel(x, y, color);
                }
            }

            _densityTexture.Apply();

            // Create visualization quad
            CreateDensityTextureQuad(lateralRange, depthRange);

            Debug.Log("[Geant4Visualizer] Density texture visualization complete!");
        }

        /// <summary>
        /// Determine lateral and depth axes based on beam direction.
        /// Lateral = perpendicular to beam (radial distance from axis)
        /// Depth = along beam direction
        /// </summary>
        private void GetProjectionAxes(out float[] lateral, out float[] depth)
        {
            lateral = new float[_positionsX.Length];
            depth = new float[_positionsX.Length];

            switch (BeamAxis)
            {
                case BeamDirection.PositiveZ:
                    // Beam travels in +Z
                    // Depth = Z
                    // Lateral = sqrt(X² + Y²) with sign from X
                    for (int i = 0; i < _positionsX.Length; i++)
                    {
                        depth[i] = _positionsZ[i];
                        // Use X as lateral (signed distance from beam axis)
                        lateral[i] = _positionsX[i];
                    }
                    break;

                case BeamDirection.PositiveY:
                    // Beam travels in +Y
                    // Depth = Y
                    // Lateral = X
                    for (int i = 0; i < _positionsX.Length; i++)
                    {
                        depth[i] = _positionsY[i];
                        lateral[i] = _positionsX[i];
                    }
                    break;

                case BeamDirection.PositiveX:
                    // Beam travels in +X
                    // Depth = X
                    // Lateral = Z
                    for (int i = 0; i < _positionsX.Length; i++)
                    {
                        depth[i] = _positionsX[i];
                        lateral[i] = _positionsZ[i];
                    }
                    break;
            }
        }

        private void CreateDensityTextureQuad(float lateralRange, float depthRange)
        {
            if (_visualizationObject != null)
            {
                Destroy(_visualizationObject);
            }

            _visualizationObject = GameObject.CreatePrimitive(PrimitiveType.Quad);
            _visualizationObject.name = "Geant4DensityMap";
            _visualizationObject.transform.SetParent(transform);

            // Fixed positioning: centered at origin with 10x10 scale
            switch (BeamAxis)
            {
                case BeamDirection.PositiveZ:
                    // XY plane, looking from +X direction
                    _visualizationObject.transform.localPosition = Vector3.zero;
                    _visualizationObject.transform.localRotation = Quaternion.Euler(0, 90, 0);
                    _visualizationObject.transform.localScale = new Vector3(10f, 10f, 1f);
                    break;

                case BeamDirection.PositiveY:
                    // XZ plane, looking from -Z direction
                    _visualizationObject.transform.localPosition = Vector3.zero;
                    _visualizationObject.transform.localRotation = Quaternion.Euler(90, 0, 0);
                    _visualizationObject.transform.localScale = new Vector3(10f, 10f, 1f);
                    break;

                case BeamDirection.PositiveX:
                    // YZ plane, looking from +Y direction
                    _visualizationObject.transform.localPosition = Vector3.zero;
                    _visualizationObject.transform.localRotation = Quaternion.Euler(0, 0, 90);
                    _visualizationObject.transform.localScale = new Vector3(10f, 10f, 1f);
                    break;
            }

            // Apply texture
            MeshRenderer mr = _visualizationObject.GetComponent<MeshRenderer>();
            mr.material = new Material(Shader.Find("Unlit/Transparent"));
            mr.material.mainTexture = _densityTexture;
        }

        // ====================================================================
        // MATERIAL HELPERS
        // ====================================================================

        private Material CreatePointMaterial()
        {
            Material mat = new Material(Shader.Find("Particles/Standard Unlit"));
            if (mat.shader == null)
            {
                mat = new Material(Shader.Find("Sprites/Default"));
            }

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
        /// Gradient: Blue (high) → Red (mid) → White (low/zero)
        /// </summary>
        private Color GetEnergyColor(float energyFraction)
        {
            energyFraction = Mathf.Clamp01(energyFraction);

            if (energyFraction > 0.5f)
            {
                // High energy: Blue → Red
                float t = (energyFraction - 0.5f) * 2.0f;
                return Color.Lerp(MidEnergyColor, HighEnergyColor, t);
            }
            else
            {
                // Low energy: White → Red
                float t = energyFraction * 2.0f;
                return Color.Lerp(LowEnergyColor, MidEnergyColor, t);
            }
        }

        // ====================================================================
        // DEBUG HELPERS
        // ====================================================================

        private void DrawCoordinateAxes()
        {
            float axisLength = 5.0f;

            // X axis - Red
            Debug.DrawRay(Vector3.zero, Vector3.right * axisLength, Color.red, 1000f);

            // Y axis - Green
            Debug.DrawRay(Vector3.zero, Vector3.up * axisLength, Color.green, 1000f);

            // Z axis - Blue
            Debug.DrawRay(Vector3.zero, Vector3.forward * axisLength, Color.blue, 1000f);
        }

        // ====================================================================
        // EDITOR BUTTONS (for Inspector)
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

        [ContextMenu("Analyze Coordinates")]
        private void EditorAnalyzeCoordinates()
        {
            if (_hasData)
            {
                AnalyzeCoordinateRanges();
            }
            else
            {
                Debug.LogWarning("No data loaded - run simulation first");
            }
        }
#endif
    }
}