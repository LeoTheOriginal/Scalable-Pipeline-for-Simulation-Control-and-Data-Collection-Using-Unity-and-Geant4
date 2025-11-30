using UnityEngine;

namespace Core
{
    /// <summary>
    /// Manages Geant4 physics engine lifecycle.
    /// Supports both ElectronAgent (legacy) and ElectronAgentPhysics (new).
    /// </summary>
    public class Geant4Manager : MonoBehaviour
    {
        public static Geant4Manager Instance { get; private set; }

        private bool _geant4Initialized = false;

        void Awake()
        {
            // Singleton pattern
            if (Instance != null && Instance != this)
            {
                Destroy(gameObject);
                return;
            }
            Instance = this;
            DontDestroyOnLoad(gameObject);

            // Check if we need Geant4
            bool needsGeant4 = ShouldInitializeGeant4();

            if (!needsGeant4)
            {
                Debug.Log("[Geant4] Running in Inference Mode - Geant4 initialization SKIPPED");
                return;
            }

            // Initialize Geant4 for training
            Debug.Log("[Geant4] Startup sequence (Training Mode)...");

            // Preventive cleanup
            try
            {
                Geant4Interface.CloseGeant4();
            }
            catch
            {
                // Ignore errors - it's just cleanup
            }

            // Initialize physics engine
            Debug.Log("[Geant4] Initializing Physics Engine...");
            try
            {
                Geant4Interface.InitGeant4();
                _geant4Initialized = true;
                Debug.Log("[Geant4] ✅ Initialization Success");
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[Geant4] ❌ Initialization Failed: {e.Message}");
                Debug.LogError($"[Geant4] Stack trace: {e.StackTrace}");

                try
                {
                    Geant4Interface.CloseGeant4();
                }
                catch
                {
                    // Already failed
                }
            }
        }

        /// <summary>
        /// Determines if Geant4 should be initialized based on agent settings.
        /// Supports both legacy ElectronAgent and new ElectronAgentPhysics.
        /// </summary>
        private bool ShouldInitializeGeant4()
        {
            // Check new ElectronAgentPhysics agents
            var physicsAgents = FindObjectsOfType<Agents.ElectronAgentPhysics>();
            foreach (var agent in physicsAgents)
            {
                // Geant4Statistical and Hybrid modes need Geant4
                if (agent.Mode == Agents.TrainingMode.Geant4Statistical ||
                    agent.Mode == Agents.TrainingMode.Hybrid)
                {
                    Debug.Log($"[Geant4] Agent '{agent.name}' requires Geant4 (Mode={agent.Mode})");
                    return true;
                }
            }

            // Check legacy ElectronAgent agents
            var legacyAgents = FindObjectsOfType<Agents.ElectronAgent>();
            foreach (var agent in legacyAgents)
            {
                if (!agent.IsInferenceMode)
                {
                    Debug.Log($"[Geant4] Legacy agent '{agent.name}' requires Geant4");
                    return true;
                }
            }

            // No agents found or all in inference/physics-only mode
            if (physicsAgents.Length == 0 && legacyAgents.Length == 0)
            {
                Debug.LogWarning("[Geant4] No agents found - initializing Geant4 by default");
                return true;
            }

            Debug.Log("[Geant4] All agents in Inference/PhysicsBased mode - Geant4 not required");
            return false;
        }

        void OnDestroy()
        {
            if (Instance == this && _geant4Initialized)
            {
                Debug.Log("[Geant4] Cleaning up resources (OnDestroy)...");
                try
                {
                    Geant4Interface.CloseGeant4();
                    Debug.Log("[Geant4] ✅ Cleanup successful");
                }
                catch (System.Exception e)
                {
                    Debug.LogWarning($"[Geant4] Cleanup warning: {e.Message}");
                }
            }
        }

        public bool IsGeant4Available()
        {
            return _geant4Initialized;
        }
    }
}
