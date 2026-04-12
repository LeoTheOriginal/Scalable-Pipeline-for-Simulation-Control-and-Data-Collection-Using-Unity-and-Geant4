#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;

namespace Visualization
{
    /// <summary>
    /// Custom editor for CheckpointGridManager with convenient buttons and progress display.
    /// </summary>
    [CustomEditor(typeof(CheckpointGridManager))]
    public class CheckpointGridManagerEditor : Editor
    {
        private CheckpointGridManager _manager;

        private void OnEnable()
        {
            _manager = (CheckpointGridManager)target;
        }

        public override void OnInspectorGUI()
        {
            // Draw default inspector
            DrawDefaultInspector();

            EditorGUILayout.Space(10);

            // Status section
            EditorGUILayout.LabelField("=== Status ===", EditorStyles.boldLabel);

            using (new EditorGUI.DisabledGroupScope(true))
            {
                EditorGUILayout.TextField("Status", _manager.Status);
                EditorGUILayout.IntField("Checkpoints Found", _manager.CheckpointsFound);
                EditorGUILayout.IntField("Processed", _manager.CheckpointsProcessed);
            }

            EditorGUILayout.Space(10);

            // Action buttons
            EditorGUILayout.LabelField("=== Actions ===", EditorStyles.boldLabel);

            using (new EditorGUI.DisabledGroupScope(_manager.IsProcessing))
            {
                // Scan button
                if (GUILayout.Button("🔍 Scan Checkpoints (Preview)", GUILayout.Height(30)))
                {
                    _manager.ScanCheckpointsOnly();
                }

                EditorGUILayout.Space(5);

                // Main action button
                GUI.backgroundColor = new Color(0.4f, 0.8f, 0.4f);
                if (GUILayout.Button("▶ SCAN AND GENERATE GRID", GUILayout.Height(40)))
                {
                    _manager.ScanAndGenerateGrid();
                }
                GUI.backgroundColor = Color.white;
            }

            // Stop button (only when processing)
            if (_manager.IsProcessing)
            {
                EditorGUILayout.Space(5);
                GUI.backgroundColor = new Color(0.8f, 0.4f, 0.4f);
                if (GUILayout.Button("⏹ STOP AND CLEAR", GUILayout.Height(30)))
                {
                    _manager.StopAndClear();
                }
                GUI.backgroundColor = Color.white;
            }

            EditorGUILayout.Space(10);

            // Export buttons
            EditorGUILayout.LabelField("=== Export ===", EditorStyles.boldLabel);

            using (new EditorGUI.DisabledGroupScope(_manager.IsProcessing))
            {
                if (GUILayout.Button("📊 Export Statistics CSV", GUILayout.Height(25)))
                {
                    _manager.ExportStatistics();
                }

                EditorGUILayout.Space(5);

                if (GUILayout.Button("📁 Open Output Folder", GUILayout.Height(25)))
                {
                    string path = System.IO.Path.Combine(
                        serializedObject.FindProperty("OutputBasePath").stringValue,
                        serializedObject.FindProperty("DensityTextureSubfolder").stringValue,
                        serializedObject.FindProperty("AlgorithmName").stringValue
                    );

                    if (System.IO.Directory.Exists(path))
                    {
                        System.Diagnostics.Process.Start("explorer.exe", path.Replace("/", "\\"));
                    }
                    else
                    {
                        EditorUtility.DisplayDialog("Folder Not Found", 
                            $"Output folder does not exist yet:\n{path}", "OK");
                    }
                }
            }

            EditorGUILayout.Space(10);

            // Utility buttons
            EditorGUILayout.LabelField("=== Utilities ===", EditorStyles.boldLabel);

            if (GUILayout.Button("🔄 Auto-detect Algorithm Name", GUILayout.Height(25)))
            {
                AutoDetectAlgorithmName();
            }

            if (GUILayout.Button("🗑 Clear Spawned Prefabs", GUILayout.Height(25)))
            {
                _manager.ClearSpawnedPrefabs();
            }

            EditorGUILayout.Space(10);

            // Checkpoint preview
            if (_manager.Checkpoints != null && _manager.Checkpoints.Count > 0)
            {
                EditorGUILayout.LabelField("=== Checkpoint Preview ===", EditorStyles.boldLabel);

                EditorGUILayout.BeginVertical(EditorStyles.helpBox);
                
                int displayCount = Mathf.Min(_manager.Checkpoints.Count, 10);
                for (int i = 0; i < displayCount; i++)
                {
                    var cp = _manager.Checkpoints[i];
                    EditorGUILayout.LabelField($"  {i + 1}. {cp.FormattedSteps} ({cp.StepCount} steps)");
                }

                if (_manager.Checkpoints.Count > 10)
                {
                    EditorGUILayout.LabelField($"  ... and {_manager.Checkpoints.Count - 10} more");
                }

                EditorGUILayout.EndVertical();
            }

            // Repaint while processing
            if (_manager.IsProcessing)
            {
                Repaint();
            }
        }

        private void AutoDetectAlgorithmName()
        {
            string path = serializedObject.FindProperty("CheckpointFolderPath").stringValue;
            string[] parts = path.Split('/');

            for (int i = 0; i < parts.Length - 1; i++)
            {
                if (parts[i] == "results" && i + 1 < parts.Length)
                {
                    serializedObject.FindProperty("AlgorithmName").stringValue = parts[i + 1];
                    serializedObject.ApplyModifiedProperties();
                    Debug.Log($"[CheckpointGridManager] Auto-detected: {parts[i + 1]}");
                    return;
                }
            }

            EditorUtility.DisplayDialog("Auto-detect Failed", 
                "Could not extract algorithm name from path.\nExpected format: .../results/{algorithm_name}/...", "OK");
        }
    }
}
#endif