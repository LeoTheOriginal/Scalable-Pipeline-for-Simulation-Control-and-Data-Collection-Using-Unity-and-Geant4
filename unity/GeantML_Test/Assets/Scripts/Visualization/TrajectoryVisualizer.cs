using UnityEngine;
using System.Collections.Generic;
using Agents;

namespace Visualization
{
    /// <summary>
    /// Trajectory visualizer with CUMULATIVE MODE support.
    /// FIXED: Uses multiple LineRenderers to prevent teleport lines between episodes.
    /// 
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
        [Tooltip("Start color (high energy)")]
        public Color HighEnergyColor = Color.blue;

        [Tooltip("End color (low energy)")]
        public Color LowEnergyColor = Color.red;

        [Tooltip("Use energy-based coloring")]
        public bool UseEnergyGradient = true;

        [Header("Multi-Agent Colors (by AgentIndex)")]
        [Tooltip("Colors for different agents in multi-agent setup")]
        public Color[] AgentColors = new Color[]
        {
            new Color(0.2f, 0.6f, 1.0f, 0.6f),    // Agent 0: Blue (semi-transparent)
            new Color(1.0f, 0.4f, 0.2f, 0.6f),    // Agent 1: Orange
            new Color(0.2f, 1.0f, 0.4f, 0.6f),    // Agent 2: Green
            new Color(1.0f, 0.2f, 0.8f, 0.6f),    // Agent 3: Pink
            new Color(1.0f, 1.0f, 0.2f, 0.6f),    // Agent 4: Yellow
            new Color(0.6f, 0.2f, 1.0f, 0.6f),    // Agent 5: Purple
        };

        [Header("Manual Controls")]
        [Tooltip("Clear all trajectories (use in Inspector or via code)")]
        public bool ClearAllTrajectories = false;

        [Header("Debug")]
        [Tooltip("Log trajectory events")]
        public bool DebugLog = false;

        // ====================================================================
        // PRIVATE STATE - Multiple LineRenderers!
        // ====================================================================

        // Current episode being recorded
        private List<Vector3> _currentPoints;
        private List<float> _currentEnergies;
        private LineRenderer _currentLineRenderer;

        // All completed episodes (for cumulative mode)
        private List<EpisodeSegment> _episodeSegments;

        private bool _isSubscribed = false;
        private int _episodeCount = 0;
        private float _timeSinceLastFade = 0f;

        // Container for episode LineRenderers
        private GameObject _segmentsContainer;

        // ====================================================================
        // HELPER CLASSES
        // ====================================================================

        private class EpisodeSegment
        {
            public GameObject GameObject;
            public LineRenderer LineRenderer;
            public List<Vector3> Points;
            public List<float> Energies;
            public float Age; // For fading

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

            // Create container for episode segments
            _segmentsContainer = new GameObject("EpisodeSegments");
            _segmentsContainer.transform.SetParent(transform);
            _segmentsContainer.transform.localPosition = Vector3.zero;

            // Current LineRenderer (attached to this GameObject)
            _currentLineRenderer = GetComponent<LineRenderer>();
            SetupLineRenderer(_currentLineRenderer);
        }

        void Start()
        {
            // Find agent if not assigned
            if (Agent == null)
            {
                Agent = GetComponent<ElectronAgentPhysics>();
            }

            if (Agent == null)
            {
                Debug.LogError("[TrajectoryVisualizer] No ElectronAgentPhysics found!");
                enabled = false;
                return;
            }

            // Subscribe to agent events
            SubscribeToAgent();

            // Set agent-specific color
            SetAgentColor();

            if (DebugLog)
            {
                string mode = CumulativeMode ? "CUMULATIVE (Multi-LineRenderer)" : "RESET";
                Debug.Log($"[TrajectoryVisualizer] Agent #{Agent.AgentIndex} - Mode: {mode}");
            }
        }

        void Update()
        {
            // Manual clear check
            if (ClearAllTrajectories)
            {
                ClearAllTrajectories = false;
                ClearTrajectory();
            }

            // Fading update
            if (CumulativeMode && FadeOldTrajectories && _episodeSegments.Count > 0)
            {
                _timeSinceLastFade += Time.deltaTime;
                if (_timeSinceLastFade > 0.1f)
                {
                    UpdateFading();
                    _timeSinceLastFade = 0f;
                }
            }

            // Update current episode
            if (_currentPoints.Count > 0)
            {
                UpdateCurrentLineRenderer();
            }
        }

        void OnDestroy()
        {
            UnsubscribeFromAgent();

            // Clean up all episode segments
            if (_segmentsContainer != null)
            {
                Destroy(_segmentsContainer);
            }
        }

        void OnEnable()
        {
            if (Agent != null && !_isSubscribed)
            {
                SubscribeToAgent();
            }
        }

        void OnDisable()
        {
            UnsubscribeFromAgent();

            if (!CumulativeMode)
            {
                ClearTrajectory();
            }
        }

        // ====================================================================
        // EVENT SUBSCRIPTION
        // ====================================================================

        private void SubscribeToAgent()
        {
            if (Agent != null && !_isSubscribed)
            {
                Agent.OnStepTaken += OnAgentStep;
                Agent.OnEpisodeReset += OnAgentReset;
                _isSubscribed = true;

                if (DebugLog)
                {
                    Debug.Log($"[TrajectoryVisualizer] Subscribed to Agent #{Agent.AgentIndex}");
                }
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

        // ====================================================================
        // EVENT HANDLERS
        // ====================================================================

        private void OnAgentStep(Vector3 position)
        {
            if (!EnableVisualization) return;

            AddPointToCurrentEpisode(position, Agent.GetCurrentEnergy());
        }

        private void OnAgentReset()
        {
            if (DebugLog)
            {
                string action = CumulativeMode ? "saving episode and starting new" : "clearing";
                Debug.Log($"[TrajectoryVisualizer] Agent #{Agent.AgentIndex} reset - {action}");
            }

            if (CumulativeMode)
            {
                // Save current episode as separate LineRenderer
                if (_currentPoints.Count > 1) // Need at least 2 points for a line
                {
                    SaveCurrentEpisodeAsSegment();
                }

                // Start fresh episode (no teleport line possible!)
                _currentPoints.Clear();
                _currentEnergies.Clear();
                _currentLineRenderer.positionCount = 0;

                _episodeCount++;
            }
            else
            {
                // Reset mode: just clear current
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
                // Episode too long, trim oldest point
                _currentPoints.RemoveAt(0);
                _currentEnergies.RemoveAt(0);
            }

            _currentPoints.Add(position);
            _currentEnergies.Add(energy);
        }

        private void SaveCurrentEpisodeAsSegment()
        {
            // Create new GameObject for this episode
            GameObject segmentGO = new GameObject($"Episode_{_episodeCount}");
            segmentGO.transform.SetParent(_segmentsContainer.transform);
            segmentGO.transform.localPosition = Vector3.zero;

            // Add LineRenderer
            LineRenderer lr = segmentGO.AddComponent<LineRenderer>();
            SetupLineRenderer(lr);

            // Copy points
            lr.positionCount = _currentPoints.Count;
            lr.SetPositions(_currentPoints.ToArray());

            // Set gradient
            UpdateLineRendererGradient(lr, _currentEnergies, 1f);

            // Store segment
            EpisodeSegment segment = new EpisodeSegment(segmentGO, lr, _currentPoints, _currentEnergies);
            _episodeSegments.Add(segment);

            // Enforce max episodes limit
            if (_episodeSegments.Count > MaxEpisodes)
            {
                RemoveOldestEpisode();
            }

            if (DebugLog)
            {
                Debug.Log($"[TrajectoryVisualizer] Saved episode {_episodeCount} with {_currentPoints.Count} points");
            }
        }

        private void RemoveOldestEpisode()
        {
            if (_episodeSegments.Count > 0)
            {
                EpisodeSegment oldest = _episodeSegments[0];
                _episodeSegments.RemoveAt(0);

                if (oldest.GameObject != null)
                {
                    Destroy(oldest.GameObject);
                }
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

        /// <summary>
        /// Clear all trajectory points and segments.
        /// </summary>
        public void ClearTrajectory()
        {
            // Clear current episode
            _currentPoints.Clear();
            _currentEnergies.Clear();
            _currentLineRenderer.positionCount = 0;

            // Clear all saved episodes
            foreach (var segment in _episodeSegments)
            {
                if (segment.GameObject != null)
                {
                    Destroy(segment.GameObject);
                }
            }
            _episodeSegments.Clear();

            _episodeCount = 0;

            if (DebugLog)
            {
                Debug.Log($"[TrajectoryVisualizer] All trajectories cleared for Agent #{Agent?.AgentIndex}");
            }
        }

        // ====================================================================
        // FADING (Cumulative Mode)
        // ====================================================================

        private void UpdateFading()
        {
            foreach (var segment in _episodeSegments)
            {
                segment.Age += FadeSpeed;
                float alpha = Mathf.Max(0.1f, 1f - segment.Age);

                // Update gradient with faded alpha
                UpdateLineRendererGradient(segment.LineRenderer, segment.Energies, alpha);
            }
        }

        // ====================================================================
        // LINE RENDERER SETUP & UPDATE
        // ====================================================================

        private void SetupLineRenderer(LineRenderer lr)
        {
            lr.startWidth = LineWidth;
            lr.endWidth = LineWidth;
            lr.useWorldSpace = true;
            lr.positionCount = 0;

            // Use shader that supports vertex colors
            Shader lineShader = Shader.Find("Particles/Standard Unlit");
            if (lineShader == null) lineShader = Shader.Find("Mobile/Particles/Additive");
            if (lineShader == null) lineShader = Shader.Find("Unlit/Color");
            if (lineShader == null) lineShader = Shader.Find("Sprites/Default");

            lr.material = new Material(lineShader);

            // Enable transparency
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

        private void SetAgentColor()
        {
            if (Agent == null) return;

            int colorIndex = Agent.AgentIndex % AgentColors.Length;
            Color agentColor = AgentColors[colorIndex];

            HighEnergyColor = agentColor;
            LowEnergyColor = new Color(agentColor.r * 0.3f, agentColor.g * 0.3f, agentColor.b * 0.3f, agentColor.a);

            if (DebugLog)
            {
                Debug.Log($"[TrajectoryVisualizer] Agent #{Agent.AgentIndex} color: {agentColor}");
            }
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

            foreach (var segment in _episodeSegments)
            {
                segment.LineRenderer.enabled = enabled;
            }
        }

        public void SetCumulativeMode(bool cumulative)
        {
            if (CumulativeMode != cumulative)
            {
                CumulativeMode = cumulative;
                ClearTrajectory();
                SetupLineRenderer(_currentLineRenderer);
                SetAgentColor();
            }
        }

        public int GetEpisodeCount()
        {
            return _episodeCount;
        }

        public int GetTotalPointCount()
        {
            int total = _currentPoints.Count;
            foreach (var segment in _episodeSegments)
            {
                total += segment.Points.Count;
            }
            return total;
        }
    }
}