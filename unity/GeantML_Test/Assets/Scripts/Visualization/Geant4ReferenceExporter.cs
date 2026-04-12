using UnityEngine;
using System.IO;
using System.Text;
using Core;

/// <summary>
/// Exports Geant4 reference statistics in the same CSV format as checkpoint statistics.
/// Uses existing Geant4Interface batch simulation API.
/// 
/// USAGE:
/// 1. Add this component to any GameObject (e.g., the same one with Geant4TrajectoryVisualizer)
/// 2. Set numParticles (10000 recommended for thesis)
/// 3. Right-click component -> "Export Geant4 Reference CSV"
/// 
/// Output format matches checkpoint CSV:
/// StepCount,CheckpointName,MeanPathLength,StdPathLength,MeanPenetrationDepth,...
/// </summary>
public class Geant4ReferenceExporter : MonoBehaviour
{
    [Header("Export Settings")]
    [Tooltip("Number of particles to simulate (10000 recommended)")]
    public int numParticles = 10000;

    [Tooltip("Output directory path")]
    public string outputPath = @"C:\Thesis\python\data";

    [Tooltip("Output filename (checkpoint format)")]
    public string outputFilename = "geant4_statistics.csv";

    [Header("Status")]
    [SerializeField] private bool isExporting = false;
    [SerializeField] private string lastExportTime = "";

    [ContextMenu("Export Geant4 Reference CSV")]
    public void ExportGeant4Statistics()
    {
        if (isExporting)
        {
            Debug.LogWarning("[Geant4Exporter] Export already in progress!");
            return;
        }

        isExporting = true;
        Debug.Log($"[Geant4Exporter] Starting batch simulation: {numParticles} particles...");

        // Initialize Geant4 if needed
        Geant4Interface.InitGeant4();

        // Run batch simulation
        System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
        sw.Start();

        int simulated = Geant4Interface.RunBatchSimulation(numParticles, 0);

        sw.Stop();
        Debug.Log($"[Geant4Exporter] Simulated {simulated} particles in {sw.ElapsedMilliseconds}ms");

        if (simulated == 0)
        {
            Debug.LogError("[Geant4Exporter] Batch simulation failed!");
            isExporting = false;
            return;
        }

        // Get statistics from Geant4
        float[] statsArray = new float[24];
        Geant4Interface.GetBatchStatistics(statsArray);

        var stats = Geant4BatchStatistics.FromArray(statsArray);
        Debug.Log($"[Geant4Exporter] Statistics retrieved:\n{stats}");

        // Write CSV in checkpoint format
        WriteCheckpointFormatCSV(stats, simulated);

        // Also export to file via DLL (if supported)
        string dllExportPath = Path.Combine(outputPath, "geant4_dll_export.txt");
        Geant4Interface.ExportStatisticsToFile(dllExportPath);

        lastExportTime = System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
        isExporting = false;

        Debug.Log($"[Geant4Exporter] Export complete! Files saved to: {outputPath}");
    }

    private void WriteCheckpointFormatCSV(Geant4BatchStatistics stats, int numSimulated)
    {
        Directory.CreateDirectory(outputPath);
        string filepath = Path.Combine(outputPath, outputFilename);

        var sb = new StringBuilder();

        // Header - EXACT same format as checkpoint CSV from CheckpointGridManager
        sb.AppendLine("StepCount,CheckpointName,MeanPathLength,StdPathLength,MeanPenetrationDepth,StdPenetrationDepth,MeanLateralSpread,StdLateralSpread,MeanScatterAngle,StdScatterAngle,NumParticles,BoundaryExits,BoundaryExitRate");

        // Data row (StepCount = 0 indicates reference, not a training checkpoint)
        // BoundaryExits = 0 for Geant4 (particles stop at energy depletion)
        sb.AppendLine($"0,Geant4-Reference," +
                     $"{stats.PathLengthMean:F4},{stats.PathLengthStdDev:F4}," +
                     $"{stats.DepthMean:F4},{stats.DepthStdDev:F4}," +
                     $"{stats.LateralTotalMean:F4},{stats.LateralTotalStdDev:F4}," +
                     $"{stats.MeanScatterAngle:F4},{stats.MeanScatterAngleStdDev:F4}," +
                     $"{numSimulated},0,0.00");

        File.WriteAllText(filepath, sb.ToString());
        Debug.Log($"[Geant4Exporter] Saved checkpoint-format CSV: {filepath}");
    }

    [ContextMenu("Print Expected Physics Values")]
    public void PrintExpectedValues()
    {
        Debug.Log("=== Expected Physics Values for 10 MeV electrons in water ===\n" +
                 "CSDA Range: 4.98 cm (NIST ESTAR)\n" +
                 "Practical Range: ~4.3-4.5 cm (R80)\n" +
                 "Highland RMS Angle: ~10-12 degrees\n" +
                 "Lateral Spread (1σ): ~1.0-1.5 cm\n" +
                 "Detour Factor: ~1.1-1.3\n" +
                 "=== Use these to validate Geant4 output ===");
    }
}