using UnityEngine;
using System.Collections.Generic;
using Agents;

namespace Visualization
{
    /// <summary>
    /// Trajectory visualizer with CUMULATIVE MODE support.
    /// Uses multiple LineRenderers to prevent teleport lines between episodes.
    /// Each episode gets its own LineRenderer = perfect separation!
    /// </summary>
    [RequireComponent(typeof(LineRenderer))]
    public class TrajectoryVisualizer : MonoBehaviour
    {
        // ====================================================================
        // INSPECTOR SETTINGS
        // ====================================================================

        [Header("Agent Reference")]
        [Tooltip("The agent to visualize. If null, will try to find on same GameObject.")]
        public ElectronAgentPhysics Agent;

        [Header("Visualization Mode")]
        [Tooltip("CUMULATIVE: Keep all trajectories, overlay effect. RESET: Clear on episode start.")]
        public bool CumulativeMode = true;

        [Tooltip("Enable/disable visualization")]
        public bool EnableVisualization = true;

        [Header("Cumulative Mode Settings")]
        [Tooltip("Maximum number of episodes to keep (memory limit)")]
        public int MaxEpisodes = 100;

        [Tooltip("Maximum points per episode")]
        public int MaxPointsPerEpisode = 1000;

        [Tooltip("Fade old trajectories (alpha decreases over time)")]
        public bool FadeOldTrajectories = false;

        [Tooltip("Fade speed (0-1, higher = faster fade)")]
        [Range(0f, 1f)]
        public float FadeSpeed = 0.01f;

        [Header("Line Settings")]
        [Tooltip("Line width")]
        public float LineWidth = 0.02f;

        [Header("Color Settings")]
        [Tooltip("Start color (high energy) - Green for thesis")]
        public Color HighEnergyColor = Color.green; // Zmieniono na zielony

        [Tooltip("End color (low energy) - Red for thesis")]
        public Color LowEnergyColor = Color.red;   // Zmieniono na czerwony

        [Tooltip("Use energy-based coloring")]
        public bool UseEnergyGradient = true;

        [Header("Manual Controls")]
        [Tooltip("Clear all trajectories (use in Inspector or via code)")]
        public bool ClearAllTrajectories = false;

        [Header("Debug")]
        [Tooltip("Log trajectory events")]
        public bool DebugLog = false;

        // ====================================================================
        // PRIVATE STATE
        // ====================================================================

        private List<Vector3> _currentPoints;
        private List<float> _currentEnergies;
        private LineRenderer _currentLineRenderer;
        private List<EpisodeSegment> _episodeSegments;
        private bool _isSubscribed = false;
        private int _episodeCount = 0;
        private float _timeSinceLastFade = 0f;
        private GameObject _segmentsContainer;

        private class EpisodeSegment
        {
            public GameObject GameObject;
            public LineRenderer LineRenderer;
            public List<Vector3> Points;
            public List<float> Energies;
            public float Age;

            public EpisodeSegment(GameObject go, LineRenderer lr, List<Vector3> points, List<float> energies)
            {
                GameObject = go;
                LineRenderer = lr;
                Points = new List<Vector3>(points);
                Energies = new List<float>(energies);
                Age = 0f;
            }
        }

        // ====================================================================
        // LIFECYCLE
        // ====================================================================

        void Awake()
        {
            _currentPoints = new List<Vector3>(MaxPointsPerEpisode);
            _currentEnergies = new List<float>(MaxPointsPerEpisode);
            _episodeSegments = new List<EpisodeSegment>();

            _segmentsContainer = new GameObject("EpisodeSegments");
            _segmentsContainer.transform.SetParent(transform);
            _segmentsContainer.transform.localPosition = Vector3.zero;

            _currentLineRenderer = GetComponent<LineRenderer>();
            SetupLineRenderer(_currentLineRenderer);
        }

        void Start()
        {
            if (Agent == null) Agent = GetComponent<ElectronAgentPhysics>();

            if (Agent == null)
            {
                Debug.LogError("[TrajectoryVisualizer] No ElectronAgentPhysics found!");
                enabled = false;
                return;
            }

            SubscribeToAgent();

            // USUNIÊTO: SetAgentColor(); - to nadpisywa³o kolory!

            if (DebugLog)
            {
                string mode = CumulativeMode ? "CUMULATIVE" : "RESET";
                Debug.Log($"[TrajectoryVisualizer] Agent #{Agent.AgentIndex} - Mode: {mode}");
            }
        }

        void Update()
        {
            if (ClearAllTrajectories)
            {
                ClearAllTrajectories = false;
                ClearTrajectory();
            }

            if (CumulativeMode && FadeOldTrajectories && _episodeSegments.Count > 0)
            {
                _timeSinceLastFade += Time.deltaTime;
                if (_timeSinceLastFade > 0.1f)
                {
                    UpdateFading();
                    _timeSinceLastFade = 0f;
                }
            }

            if (_currentPoints.Count > 0)
            {
                UpdateCurrentLineRenderer();
            }
        }

        void OnDestroy()
        {
            UnsubscribeFromAgent();
            if (_segmentsContainer != null) Destroy(_segmentsContainer);
        }

        void OnEnable()
        {
            if (Agent != null && !_isSubscribed) SubscribeToAgent();
        }

        void OnDisable()
        {
            UnsubscribeFromAgent();
            if (!CumulativeMode) ClearTrajectory();
        }

        // ====================================================================
        // EVENT SUBSCRIPTION & HANDLERS
        // ====================================================================

        private void SubscribeToAgent()
        {
            if (Agent != null && !_isSubscribed)
            {
                Agent.OnStepTaken += OnAgentStep;
                Agent.OnEpisodeReset += OnAgentReset;
                _isSubscribed = true;
            }
        }

        private void UnsubscribeFromAgent()
        {
            if (Agent != null && _isSubscribed)
            {
                Agent.OnStepTaken -= OnAgentStep;
                Agent.OnEpisodeReset -= OnAgentReset;
                _isSubscribed = false;
            }
        }

        private void OnAgentStep(Vector3 position)
        {
            if (!EnableVisualization) return;
            AddPointToCurrentEpisode(position, Agent.GetCurrentEnergy());
        }

        private void OnAgentReset()
        {
            if (CumulativeMode)
            {
                if (_currentPoints.Count > 1) SaveCurrentEpisodeAsSegment();

                _currentPoints.Clear();
                _currentEnergies.Clear();
                _currentLineRenderer.positionCount = 0;
                _episodeCount++;
            }
            else
            {
                ClearTrajectory();
            }
        }

        // ====================================================================
        // TRAJECTORY MANAGEMENT
        // ====================================================================

        private void AddPointToCurrentEpisode(Vector3 position, float energy)
        {
            if (_currentPoints.Count >= MaxPointsPerEpisode)
            {
                _currentPoints.RemoveAt(0);
                _currentEnergies.RemoveAt(0);
            }
            _currentPoints.Add(position);
            _currentEnergies.Add(energy);
        }

        private void SaveCurrentEpisodeAsSegment()
        {
            GameObject segmentGO = new GameObject($"Episode_{_episodeCount}");
            segmentGO.transform.SetParent(_segmentsContainer.transform);
            segmentGO.transform.localPosition = Vector3.zero;

            LineRenderer lr = segmentGO.AddComponent<LineRenderer>();
            SetupLineRenderer(lr);

            lr.positionCount = _currentPoints.Count;
            lr.SetPositions(_currentPoints.ToArray());

            UpdateLineRendererGradient(lr, _currentEnergies, 1f);

            EpisodeSegment segment = new EpisodeSegment(segmentGO, lr, _currentPoints, _currentEnergies);
            _episodeSegments.Add(segment);

            if (_episodeSegments.Count > MaxEpisodes) RemoveOldestEpisode();
        }

        private void RemoveOldestEpisode()
        {
            if (_episodeSegments.Count > 0)
            {
                EpisodeSegment oldest = _episodeSegments[0];
                _episodeSegments.RemoveAt(0);
                if (oldest.GameObject != null) Destroy(oldest.GameObject);
            }
        }

        private void UpdateCurrentLineRenderer()
        {
            if (_currentPoints.Count < 2) return;

            _currentLineRenderer.positionCount = _currentPoints.Count;
            _currentLineRenderer.SetPositions(_currentPoints.ToArray());

            if (UseEnergyGradient && _currentEnergies.Count == _currentPoints.Count)
            {
                UpdateLineRendererGradient(_currentLineRenderer, _currentEnergies, 1f);
            }
        }

        public void ClearTrajectory()
        {
            _currentPoints.Clear();
            _currentEnergies.Clear();
            _currentLineRenderer.positionCount = 0;

            foreach (var segment in _episodeSegments)
            {
                if (segment.GameObject != null) Destroy(segment.GameObject);
            }
            _episodeSegments.Clear();
            _episodeCount = 0;
        }

        // ====================================================================
        // RENDERER UPDATES
        // ====================================================================

        private void UpdateFading()
        {
            foreach (var segment in _episodeSegments)
            {
                segment.Age += FadeSpeed;
                float alpha = Mathf.Max(0.1f, 1f - segment.Age);
                UpdateLineRendererGradient(segment.LineRenderer, segment.Energies, alpha);
            }
        }

        private void SetupLineRenderer(LineRenderer lr)
        {
            lr.startWidth = LineWidth;
            lr.endWidth = LineWidth;
            lr.useWorldSpace = true;
            lr.positionCount = 0;

            Shader lineShader = Shader.Find("Particles/Standard Unlit");
            if (lineShader == null) lineShader = Shader.Find("Sprites/Default");

            lr.material = new Material(lineShader);

            if (CumulativeMode)
            {
                lr.material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
                lr.material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
                lr.material.SetInt("_ZWrite", 0);
                lr.material.EnableKeyword("_ALPHABLEND_ON");
                lr.material.renderQueue = 3000;
            }

            lr.startColor = Color.white;
            lr.endColor = Color.white;
        }

        private void UpdateLineRendererGradient(LineRenderer lr, List<float> energies, float alphaMultiplier)
        {
            if (energies.Count < 2) return;

            Gradient gradient = new Gradient();
            int keyCount = Mathf.Min(8, energies.Count);
            GradientColorKey[] colorKeys = new GradientColorKey[keyCount];
            GradientAlphaKey[] alphaKeys = new GradientAlphaKey[keyCount];

            float initialEnergy = Physics.ElectronPhysics.INITIAL_ENERGY;

            for (int i = 0; i < keyCount; i++)
            {
                float t = (float)i / (keyCount - 1);
                int energyIndex = Mathf.FloorToInt(t * (energies.Count - 1));

                float energyFraction = energies[energyIndex] / initialEnergy;

                // Gradient: HighEnergy (Start) -> LowEnergy (End)
                Color color = Color.Lerp(LowEnergyColor, HighEnergyColor, energyFraction);

                float alpha = color.a * alphaMultiplier;
                colorKeys[i] = new GradientColorKey(color, t);
                alphaKeys[i] = new GradientAlphaKey(alpha, t);
            }

            gradient.SetKeys(colorKeys, alphaKeys);
            lr.colorGradient = gradient;
        }

        // ====================================================================
        // PUBLIC API
        // ====================================================================

        public void SetVisualizationEnabled(bool enabled)
        {
            EnableVisualization = enabled;
            _currentLineRenderer.enabled = enabled;
            foreach (var segment in _episodeSegments) segment.LineRenderer.enabled = enabled;
        }

        public void SetCumulativeMode(bool cumulative)
        {
            if (CumulativeMode != cumulative)
            {
                CumulativeMode = cumulative;
                ClearTrajectory();
                SetupLineRenderer(_currentLineRenderer);
                // Usuniêto SetAgentColor()
            }
        }

        public int GetEpisodeCount() => _episodeCount;
        public int GetTotalPointCount()
        {
            int total = _currentPoints.Count;
            foreach (var segment in _episodeSegments) total += segment.Points.Count;
            return total;
        }
    }
}