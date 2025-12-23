using System.Runtime.InteropServices;
using UnityEngine;

namespace Core
{
    /// <summary>
    /// Interface to Geant4 physics engine via DLL.
    /// Extended with batch simulation capabilities for 100k+ particles.
    /// </summary>
    public static class Geant4Interface
    {
        private const string DLL_NAME = "geant4_plugin";

        // ====================================================================
        // ORIGINAL FUNCTIONS
        // ====================================================================

        [DllImport(DLL_NAME)]
        public static extern void InitGeant4();

        [DllImport(DLL_NAME)]
        public static extern void CloseGeant4();

        /// <summary>
        /// Run single particle simulation.
        /// </summary>
        /// <param name="outData">Buffer [maxSteps * 7] for trajectory data</param>
        /// <param name="maxSteps">Maximum number of steps</param>
        /// <returns>Actual number of steps recorded</returns>
        [DllImport(DLL_NAME)]
        public static extern int RunSimulationBatch([In, Out] float[] outData, int maxSteps);

        // ====================================================================
        // NEW: BATCH SIMULATION FUNCTIONS
        // ====================================================================

        /// <summary>
        /// Run batch simulation of multiple particles.
        /// Stores all trajectory data internally for later retrieval.
        /// </summary>
        /// <param name="numParticles">Number of particles to simulate (e.g., 100000)</param>
        /// <param name="progressCallback">Reserved for future progress reporting</param>
        /// <returns>Number of particles successfully simulated</returns>
        [DllImport(DLL_NAME)]
        public static extern int RunBatchSimulation(int numParticles, int progressCallback);

        /// <summary>
        /// Get computed statistics from batch simulation.
        /// </summary>
        /// <param name="outStats">Buffer of 24 floats for statistics</param>
        [DllImport(DLL_NAME)]
        public static extern void GetBatchStatistics([In, Out] float[] outStats);

        /// <summary>
        /// Get trajectory data for visualization.
        /// </summary>
        /// <param name="outX">X positions buffer</param>
        /// <param name="outY">Y positions buffer</param>
        /// <param name="outZ">Z positions buffer</param>
        /// <param name="outEnergy">Energy values buffer</param>
        /// <param name="maxSteps">Maximum steps to retrieve</param>
        /// <returns>Actual number of steps retrieved</returns>
        [DllImport(DLL_NAME)]
        public static extern int GetBatchTrajectoryData(
            [In, Out] float[] outX,
            [In, Out] float[] outY,
            [In, Out] float[] outZ,
            [In, Out] float[] outEnergy,
            int maxSteps);

        /// <summary>
        /// Get lateral distribution histogram data.
        /// </summary>
        /// <param name="outBins">Output histogram bins</param>
        /// <param name="numBins">Number of bins</param>
        /// <param name="minVal">Minimum value for binning</param>
        /// <param name="maxVal">Maximum value for binning</param>
        [DllImport(DLL_NAME)]
        public static extern void GetLateralDistribution(
            [In, Out] float[] outBins,
            int numBins,
            float minVal,
            float maxVal);

        /// <summary>
        /// Export statistics to file.
        /// </summary>
        /// <param name="filePath">Path to output file</param>
        /// <returns>1 on success, 0 on failure</returns>
        [DllImport(DLL_NAME)]
        public static extern int ExportStatisticsToFile(string filePath);

        /// <summary>
        /// Get total number of steps in batch simulation.
        /// </summary>
        [DllImport(DLL_NAME)]
        public static extern int GetBatchStepCount();

        /// <summary>
        /// Get number of particles in batch simulation.
        /// </summary>
        [DllImport(DLL_NAME)]
        public static extern int GetBatchParticleCount();

        [DllImport(DLL_NAME)]
        public static extern float MeasureGeant4Performance(int numRuns, int maxSteps);

        [DllImport(DLL_NAME)]
        public static extern void MeasureDetailedPerformance([Out] float[] outMetrics);

        [DllImport(DLL_NAME)]
        public static extern void BenchmarkGeant4Performance(
            int numRuns,
            [Out] float[] outStats,
            int statsSize
        );
    }

    /// <summary>
    /// Data structure for batch simulation statistics.
    /// Based on Highland formula, Bethe-Bloch, and CSDA range calculations.
    /// </summary>
    [System.Serializable]
    public struct Geant4BatchStatistics
    {
        // Path length statistics [cm]
        public float PathLengthMean;
        public float PathLengthStdDev;
        public float PathLengthMedian;

        // Penetration depth statistics [cm]
        public float DepthMean;
        public float DepthStdDev;
        public float DepthMedian;

        // Lateral spread Y [cm] with sigma bounds
        public float LateralYMean;
        public float LateralYStdDev;
        public float LateralYSigma2;

        // Lateral spread Z [cm] with sigma bounds
        public float LateralZMean;
        public float LateralZStdDev;
        public float LateralZSigma2;

        // Total lateral spread [cm] with sigma bounds
        public float LateralTotalMean;
        public float LateralTotalStdDev;
        public float LateralTotalSigma3;

        // Final energy statistics [MeV]
        public float FinalEnergyMean;
        public float FinalEnergyStdDev;
        public float FinalEnergyMedian;

        // Scattering angle statistics [degrees]
        public float MeanScatterAngle;
        public float MeanScatterAngleStdDev;
        public float MeanScatterAngleMedian;

        // Scatter angle std dev statistics [degrees]
        public float ScatterStdDevMean;
        public float ScatterStdDevStdDev;
        public float ScatterStdDevMedian;

        /// <summary>
        /// Parse statistics from raw float array (24 values).
        /// </summary>
        public static Geant4BatchStatistics FromArray(float[] data)
        {
            return new Geant4BatchStatistics
            {
                PathLengthMean = data[0],
                PathLengthStdDev = data[1],
                PathLengthMedian = data[2],

                DepthMean = data[3],
                DepthStdDev = data[4],
                DepthMedian = data[5],

                LateralYMean = data[6],
                LateralYStdDev = data[7],
                LateralYSigma2 = data[8],

                LateralZMean = data[9],
                LateralZStdDev = data[10],
                LateralZSigma2 = data[11],

                LateralTotalMean = data[12],
                LateralTotalStdDev = data[13],
                LateralTotalSigma3 = data[14],

                FinalEnergyMean = data[15],
                FinalEnergyStdDev = data[16],
                FinalEnergyMedian = data[17],

                MeanScatterAngle = data[18],
                MeanScatterAngleStdDev = data[19],
                MeanScatterAngleMedian = data[20],

                ScatterStdDevMean = data[21],
                ScatterStdDevStdDev = data[22],
                ScatterStdDevMedian = data[23]
            };
        }

        /// <summary>
        /// Calculate detour factor (path length / penetration depth).
        /// Expected: 1.1-1.3 for 10 MeV electrons in water.
        /// </summary>
        public float GetDetourFactor()
        {
            return DepthMean > 0.01f ? PathLengthMean / DepthMean : 1.0f;
        }

        /// <summary>
        /// Get expected CSDA range for comparison (approximately 4.98 cm for 10 MeV).
        /// </summary>
        public float GetExpectedCSDARange()
        {
            return 4.98f;
        }

        /// <summary>
        /// Get expected RMS scattering angle from Highland formula [degrees].
        /// </summary>
        public float GetExpectedHighlandRMS()
        {
            // θ_RMS = (13.6 MeV / βcp) * √(x/X₀)
            // For 10 MeV electron: β ≈ 0.998, p ≈ 10.5 MeV/c
            // Path length ~ 5 cm, X₀ = 36.08 cm
            float betaCp = 0.998f * 10.5f;
            float xOverX0 = PathLengthMean / 36.08f;
            float theta0Rad = (13.6f / betaCp) * Mathf.Sqrt(xOverX0);
            return theta0Rad * Mathf.Rad2Deg;
        }

        public override string ToString()
        {
            return $"Geant4 Batch Statistics:\n" +
                   $"  Path Length: {PathLengthMean:F3} ± {PathLengthStdDev:F3} cm\n" +
                   $"  Depth: {DepthMean:F3} ± {DepthStdDev:F3} cm (CSDA expected: {GetExpectedCSDARange():F2} cm)\n" +
                   $"  Lateral Y: {LateralYMean:F3} ± {LateralYStdDev:F3} cm (2σ: {LateralYSigma2:F3} cm)\n" +
                   $"  Lateral Z: {LateralZMean:F3} ± {LateralZStdDev:F3} cm (2σ: {LateralZSigma2:F3} cm)\n" +
                   $"  Lateral Total: {LateralTotalMean:F3} ± {LateralTotalStdDev:F3} cm (3σ: {LateralTotalSigma3:F3} cm)\n" +
                   $"  Final Energy: {FinalEnergyMean:F3} ± {FinalEnergyStdDev:F3} MeV\n" +
                   $"  MCS Angle: {MeanScatterAngle:F2} ± {MeanScatterAngleStdDev:F2}° (Highland: {GetExpectedHighlandRMS():F2}°)\n" +
                   $"  Detour Factor: {GetDetourFactor():F3}";
        }
    }
}