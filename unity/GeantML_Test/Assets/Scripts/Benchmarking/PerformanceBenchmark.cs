using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Diagnostics;
using Unity.MLAgents.Policies;
using Agents;
using Core;

namespace Benchmarking
{
    /// <summary>
    /// Performance benchmark system for comparing Geant4 Monte Carlo 
    /// simulation with ML Agent inference times.
    /// 
    /// Measures:
    /// - Average trajectory simulation time
    /// - Frames per second (trajectories/sec)
    /// - Speedup factor
    /// - Statistical distribution of timings
    /// 
    /// Output format compatible with Python visualization script.
    /// </summary>
    public class PerformanceBenchmark : MonoBehaviour
    {

        private const string DLL_NAME = "geant4_plugin";

        // ====================================================================
        // GEANT4 NATIVE CALLS
        // ====================================================================

        [DllImport(DLL_NAME)]
        private static extern float MeasureGeant4Performance(int numRuns, int maxSteps);

        [DllImport(DLL_NAME)]
        private static extern void MeasureDetailedPerformance([Out] float[] outMetrics);

        [DllImport(DLL_NAME)]
        private static extern void BenchmarkGeant4Performance(
            int numRuns,
            [Out] float[] outStats,
            int statsSize);

        // ====================================================================
        // INSPECTOR SETTINGS
        // ====================================================================

        [Header("Benchmark Configuration")]
        [Tooltip("Number of trajectories to simulate for averaging")]
        public int NumBenchmarkRuns = 100;

        [Tooltip("Maximum steps per trajectory")]
        public int MaxStepsPerTrajectory = 500;

        [Header("Fair Comparison")]
        public int TargetStepsForComparison = 50;

        [Header("ML Agent Reference")]
        [Tooltip("Reference to trained ML agent for inference benchmarking")]
        public ElectronAgentPhysics MLAgent;

        [Header("Output Settings")]
        [Tooltip("Export results to JSON file")]
        public bool ExportToFile = true;

        [Tooltip("Output file path (relative to Application.dataPath)")]
        public string OutputFileName = "performance_benchmark.json";

        [Header("Automatic Execution")]
        [Tooltip("Run benchmark on Start")]
        public bool RunOnStart = false;

        [Header("Debug")]
        public bool VerboseLogging = true;

        // ====================================================================
        // BENCHMARK RESULTS
        // ====================================================================

        [System.Serializable]
        public class PerformanceMetrics
        {
            public string SystemName;
            public float MeanTimeMs;
            public float StdDevMs;
            public float MinTimeMs;
            public float MaxTimeMs;
            public float MedianTimeMs;
            public float FPS;
            public int TotalSteps;
            public int NumRuns;

            public override string ToString()
            {
                return $"[{SystemName}]\n" +
                       $"  Mean: {MeanTimeMs:F3} ms/trajectory\n" +
                       $"  StdDev: {StdDevMs:F3} ms\n" +
                       $"  Range: [{MinTimeMs:F3}, {MaxTimeMs:F3}] ms\n" +
                       $"  Median: {MedianTimeMs:F3} ms\n" +
                       $"  FPS: {FPS:F2} trajectories/sec\n" +
                       $"  Total steps: {TotalSteps}\n" +
                       $"  Runs: {NumRuns}";
            }
        }

        [System.Serializable]
        public class BenchmarkResults
        {
            public PerformanceMetrics Geant4;
            public PerformanceMetrics MLAgent;
            public float SpeedupFactor;
            public string Timestamp;

            public override string ToString()
            {
                return $"=== PERFORMANCE BENCHMARK ===\n" +
                       $"Timestamp: {Timestamp}\n\n" +
                       $"{Geant4}\n\n" +
                       $"{MLAgent}\n\n" +
                       $"SPEEDUP: {SpeedupFactor:F1}x faster";
            }
        }

        private BenchmarkResults _lastResults;

        // ====================================================================
        // LIFECYCLE
        // ====================================================================

        void Start()
        {
            if (RunOnStart)
            {
                RunBenchmark();
            }
        }

        // ====================================================================
        // PUBLIC API
        // ====================================================================

        /// <summary>
        /// Run complete performance benchmark.
        /// </summary>
        public BenchmarkResults RunBenchmark()
        {
            UnityEngine.Debug.Log($"[PerformanceBenchmark] Starting benchmark with {NumBenchmarkRuns} runs...");

            _lastResults = new BenchmarkResults
            {
                Timestamp = System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")
            };

            // 1. Benchmark Geant4
            _lastResults.Geant4 = BenchmarkGeant4();

            // 2. Benchmark ML Agent
            _lastResults.MLAgent = BenchmarkMLAgent();

            // 3. Calculate speedup
            _lastResults.SpeedupFactor = _lastResults.Geant4.MeanTimeMs / _lastResults.MLAgent.MeanTimeMs;

            // 4. Log results
            UnityEngine.Debug.Log(_lastResults.ToString());

            // 5. Export to file
            if (ExportToFile)
            {
                ExportResults(_lastResults);
            }

            return _lastResults;
        }

        /// <summary>
        /// Get last benchmark results.
        /// </summary>
        public BenchmarkResults GetLastResults()
        {
            return _lastResults;
        }

        // ====================================================================
        // GEANT4 BENCHMARKING
        // ====================================================================

        private PerformanceMetrics BenchmarkGeant4()
        {
            UnityEngine.Debug.Log("[PerformanceBenchmark] Benchmarking Geant4 Monte Carlo...");

            Geant4Interface.InitGeant4();

            // Run detailed benchmark
            float[] stats = new float[6];
            BenchmarkGeant4Performance(NumBenchmarkRuns, stats, stats.Length);

            PerformanceMetrics metrics = new PerformanceMetrics
            {
                SystemName = "Geant4 Monte Carlo",
                MeanTimeMs = stats[0],
                StdDevMs = stats[1],
                MinTimeMs = stats[2],
                MaxTimeMs = stats[3],
                MedianTimeMs = stats[4],
                TotalSteps = (int)stats[5],
                NumRuns = NumBenchmarkRuns,
                FPS = 1000.0f / stats[0]  // Convert ms to FPS
            };

            if (VerboseLogging)
            {
                UnityEngine.Debug.Log(metrics.ToString());
            }

            return metrics;
        }

        // ====================================================================
        // ML AGENT BENCHMARKING
        // ====================================================================

        private PerformanceMetrics BenchmarkMLAgent()
        {
            UnityEngine.Debug.Log("[PerformanceBenchmark] Benchmarking ML Agent inference...");

            if (MLAgent == null)
            {
                UnityEngine.Debug.LogError("[PerformanceBenchmark] ML Agent reference is null!");
                return new PerformanceMetrics { SystemName = "ML Agent (ERROR)" };
            }

            List<float> trajectoryTimes = new List<float>();
            int actualTotalSteps = 0; // Dodajemy licznik faktycznych kroków
            Stopwatch episodeTimer = new Stopwatch();

            for (int run = 0; run < NumBenchmarkRuns; run++)
            {
                // Ważne: Resetujemy agenta przed startem pomiaru
                MLAgent.OnEpisodeBegin();

                episodeTimer.Restart();

                int stepsThisRun = 0;
                // Pętla wykonuje DOKŁADNIE TargetStepsForComparison, 
                // chyba że agent zginie/wyjdzie za mapę wcześniej
                while (stepsThisRun < TargetStepsForComparison)
                {
                    MLAgent.RequestDecision();
                    stepsThisRun++;

                    // Sprawdzamy warunki przerwania (np. utrata energii)
                    if (MLAgent.GetCurrentEnergy() <= 0.01f || MLAgent.DidExitBoundary())
                    {
                        break;
                    }
                }

                episodeTimer.Stop();

                float timeMs = (float)episodeTimer.Elapsed.TotalMilliseconds;
                trajectoryTimes.Add(timeMs);
                actualTotalSteps += stepsThisRun; // Akumulujemy faktyczne kroki

                if (VerboseLogging && run % 10 == 0)
                {
                    UnityEngine.Debug.Log($"[Benchmark] Run {run + 1}/{NumBenchmarkRuns}: {timeMs:F3} ms ({stepsThisRun} steps)");
                }
            }

            // Obliczenia statystyczne (bez zmian, ale na poprawionych danych)
            float mean = trajectoryTimes.Average();
            float variance = trajectoryTimes.Select(t => (t - mean) * (t - mean)).Average();
            float stdDev = Mathf.Sqrt(variance);
            trajectoryTimes.Sort();

            return new PerformanceMetrics
            {
                SystemName = "ML Agent (Trained)",
                MeanTimeMs = mean,
                StdDevMs = stdDev,
                MinTimeMs = trajectoryTimes.First(),
                MaxTimeMs = trajectoryTimes.Last(),
                MedianTimeMs = trajectoryTimes[trajectoryTimes.Count / 2],
                TotalSteps = actualTotalSteps, // TERAZ: Faktyczna liczba kroków zapisana w JSON
                NumRuns = NumBenchmarkRuns,
                FPS = 1000.0f / mean
            };
        }

        // ====================================================================
        // EXPORT
        // ====================================================================

        private void ExportResults(BenchmarkResults results)
        {
            string json = JsonUtility.ToJson(results, true);
            string fullPath = System.IO.Path.Combine(Application.dataPath, OutputFileName);

            try
            {
                System.IO.File.WriteAllText(fullPath, json);
                UnityEngine.Debug.Log($"[PerformanceBenchmark] Results exported to: {fullPath}");
            }
            catch (System.Exception e)
            {
                UnityEngine.Debug.LogError($"[PerformanceBenchmark] Failed to export: {e.Message}");
            }
        }

        // ====================================================================
        // EDITOR UTILITIES
        // ====================================================================

#if UNITY_EDITOR
        [ContextMenu("Run Benchmark")]
        private void EditorRunBenchmark()
        {
            RunBenchmark();
        }

        [ContextMenu("Quick Test (10 runs)")]
        private void EditorQuickTest()
        {
            int originalRuns = NumBenchmarkRuns;
            NumBenchmarkRuns = 10;
            RunBenchmark();
            NumBenchmarkRuns = originalRuns;
        }
#endif
    }
}