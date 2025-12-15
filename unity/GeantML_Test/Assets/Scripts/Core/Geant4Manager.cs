using UnityEngine;
using System.Collections.Generic;

namespace Core
{
    /// <summary>
    /// Manages Geant4 physics engine lifecycle for multi-agent training.
    /// 
    /// Responsibilities:
    /// - Initialize Geant4 once at application start (if needed)
    /// - Determine if Geant4 is required based on agent configurations
    /// - Clean up resources on application exit
    /// 
    /// Multi-agent support:
    /// - Scans all ElectronAgentPhysics in scene
    /// - Only initializes Geant4 if at least one agent needs it
    /// - Provides centralized access point for Geant4 status
    /// </summary>
    public class Geant4Manager : MonoBehaviour
    {
        // ====================================================================
        // SINGLETON
        // ====================================================================

        public static Geant4Manager Instance { get; private set; }

        // ====================================================================
        // STATE
        // ====================================================================

        private bool _geant4Initialized = false;
        private int _agentCountRequiringGeant4 = 0;
        private int _totalAgentCount = 0;

        // ====================================================================
        // LIFECYCLE
        // ====================================================================

        void Awake()
        {
            // Singleton pattern with DontDestroyOnLoad
            if (Instance != null && Instance != this)
            {
                Debug.LogWarning("[Geant4Manager] Duplicate instance detected, destroying...");
                Destroy(gameObject);
                return;
            }
            Instance = this;
            DontDestroyOnLoad(gameObject);

            // Analyze agent configurations
            AnalyzeAgentConfigurations();

            // Decide whether to initialize Geant4
            if (_agentCountRequiringGeant4 == 0)
            {
                Debug.Log("[Geant4Manager] No agents require Geant4 - initialization SKIPPED");
                Debug.Log($"  Total agents: {_totalAgentCount}");
                Debug.Log($"  All agents in PhysicsBased or Inference mode");
                return;
            }

            // Initialize Geant4
            InitializeGeant4();
        }

        void OnDestroy()
        {
            if (Instance == this)
            {
                CleanupGeant4();
                Instance = null;
            }
        }

        void OnApplicationQuit()
        {
            CleanupGeant4();
        }

        // ====================================================================
        // AGENT ANALYSIS
        // ====================================================================

        /// <summary>
        /// Scan all agents in scene and determine Geant4 requirements.
        /// </summary>
        private void AnalyzeAgentConfigurations()
        {
            var agents = FindObjectsOfType<Agents.ElectronAgentPhysics>();
            _totalAgentCount = agents.Length;
            _agentCountRequiringGeant4 = 0;

            Debug.Log($"[Geant4Manager] Found {_totalAgentCount} ElectronAgentPhysics agents:");

            foreach (var agent in agents)
            {
                bool needsGeant4 = agent.Mode == Agents.TrainingMode.Geant4Statistical;

                Debug.Log($"  - Agent #{agent.AgentIndex} '{agent.name}': Mode={agent.Mode}, NeedsGeant4={needsGeant4}");

                if (needsGeant4)
                {
                    _agentCountRequiringGeant4++;
                }
            }

            Debug.Log($"[Geant4Manager] Summary: {_agentCountRequiringGeant4}/{_totalAgentCount} agents require Geant4");

            // Warning if no agents found
            if (_totalAgentCount == 0)
            {
                Debug.LogWarning("[Geant4Manager] No ElectronAgentPhysics found in scene!");
                Debug.LogWarning("  Make sure agents are in scene before Geant4Manager.Awake() runs");
            }
        }

        // ====================================================================
        // GEANT4 INITIALIZATION
        // ====================================================================

        private void InitializeGeant4()
        {
            Debug.Log("[Geant4Manager] Starting Geant4 initialization...");
            Debug.Log($"  Agents requiring Geant4: {_agentCountRequiringGeant4}");

            // Preventive cleanup (in case previous session didn't clean up properly)
            try
            {
                Geant4Interface.CloseGeant4();
                Debug.Log("[Geant4Manager] Previous session cleanup completed");
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"[Geant4Manager] Cleanup warning (safe to ignore): {e.Message}");
            }

            // Initialize physics engine
            try
            {
                Geant4Interface.InitGeant4();
                _geant4Initialized = true;
                Debug.Log("[Geant4Manager] ✅ Geant4 initialized successfully");
                Debug.Log("  Physics: G4EmLivermorePolarizedPhysics");
                Debug.Log("  Geometry: 10×10×10 cm³ water phantom");
                Debug.Log("  Particle: 10 MeV electron, perpendicular incidence");
            }
            catch (System.Exception e)
            {
                _geant4Initialized = false;
                Debug.LogError($"[Geant4Manager] ❌ Geant4 initialization FAILED");
                Debug.LogError($"  Error: {e.Message}");
                Debug.LogError($"  Stack trace: {e.StackTrace}");
                Debug.LogError("  Agents using Geant4Statistical mode will receive invalid data!");

                // Attempt cleanup after failure
                try
                {
                    Geant4Interface.CloseGeant4();
                }
                catch
                {
                    // Already failed, ignore
                }
            }
        }

        // ====================================================================
        // CLEANUP
        // ====================================================================

        private void CleanupGeant4()
        {
            if (!_geant4Initialized)
            {
                return;
            }

            Debug.Log("[Geant4Manager] Cleaning up Geant4 resources...");

            try
            {
                Geant4Interface.CloseGeant4();
                _geant4Initialized = false;
                Debug.Log("[Geant4Manager] ✅ Geant4 cleanup successful");
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"[Geant4Manager] Cleanup warning: {e.Message}");
            }
        }

        // ====================================================================
        // PUBLIC API
        // ====================================================================

        /// <summary>
        /// Check if Geant4 is available for simulation.
        /// Use this before calling Geant4Interface.RunSimulationBatch().
        /// </summary>
        public bool IsGeant4Available()
        {
            return _geant4Initialized;
        }

        /// <summary>
        /// Get the number of agents that require Geant4.
        /// Useful for logging and debugging.
        /// </summary>
        public int GetAgentCountRequiringGeant4()
        {
            return _agentCountRequiringGeant4;
        }

        /// <summary>
        /// Get total agent count in scene.
        /// </summary>
        public int GetTotalAgentCount()
        {
            return _totalAgentCount;
        }

        /// <summary>
        /// Refresh agent analysis (call if agents are added/removed at runtime).
        /// Does NOT reinitialize Geant4 - that requires application restart.
        /// </summary>
        public void RefreshAgentAnalysis()
        {
            AnalyzeAgentConfigurations();
        }
    }
}