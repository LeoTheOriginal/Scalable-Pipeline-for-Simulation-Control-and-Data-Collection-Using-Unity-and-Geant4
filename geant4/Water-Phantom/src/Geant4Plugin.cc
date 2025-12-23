#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "ActionInitialization.hh"
#include "EventAction.hh"
#include "G4SystemOfUnits.hh"
#include <chrono>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// Global run manager pointer (DLL state must be persistent)
G4RunManager* g_RunManager = nullptr;

// Statistics storage for batch simulation
struct BatchStatistics {
    // Per-trajectory data
    std::vector<double> pathLengths;
    std::vector<double> finalDepths;        // Penetration depth (X direction)
    std::vector<double> lateralSpreadsY;    // Lateral spread in Y
    std::vector<double> lateralSpreadsZ;    // Lateral spread in Z
    std::vector<double> totalLateralSpreads;// sqrt(Y^2 + Z^2)
    std::vector<double> finalEnergies;
    std::vector<double> meanScatterAngles;
    std::vector<double> scatterStdDevs;

    // Step-level data for density map
    std::vector<float> allStepPositionsX;
    std::vector<float> allStepPositionsY;
    std::vector<float> allStepPositionsZ;
    std::vector<float> allStepEnergies;

    void Clear() {
        pathLengths.clear();
        finalDepths.clear();
        lateralSpreadsY.clear();
        lateralSpreadsZ.clear();
        totalLateralSpreads.clear();
        finalEnergies.clear();
        meanScatterAngles.clear();
        scatterStdDevs.clear();
        allStepPositionsX.clear();
        allStepPositionsY.clear();
        allStepPositionsZ.clear();
        allStepEnergies.clear();
    }

    void Reserve(int numParticles, int avgStepsPerParticle) {
        pathLengths.reserve(numParticles);
        finalDepths.reserve(numParticles);
        lateralSpreadsY.reserve(numParticles);
        lateralSpreadsZ.reserve(numParticles);
        totalLateralSpreads.reserve(numParticles);
        finalEnergies.reserve(numParticles);
        meanScatterAngles.reserve(numParticles);
        scatterStdDevs.reserve(numParticles);

        size_t totalSteps = static_cast<size_t>(numParticles) * avgStepsPerParticle;
        allStepPositionsX.reserve(totalSteps);
        allStepPositionsY.reserve(totalSteps);
        allStepPositionsZ.reserve(totalSteps);
        allStepEnergies.reserve(totalSteps);
    }
};

BatchStatistics g_BatchStats;

// Helper function to get EventAction
EventAction* GetCurrentEventAction() {
    if (!g_RunManager) return nullptr;
    auto eventAction = static_cast<const EventAction*>(g_RunManager->GetUserEventAction());
    return const_cast<EventAction*>(eventAction);
}

// Calculate statistics helper
double CalculateMean(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    double sum = 0.0;
    for (double val : data) sum += val;
    return sum / data.size();
}

double CalculateStdDev(const std::vector<double>& data, double mean) {
    if (data.size() < 2) return 0.0;
    double variance = 0.0;
    for (double val : data) {
        double diff = val - mean;
        variance += diff * diff;
    }
    return std::sqrt(variance / data.size());
}

double CalculatePercentile(std::vector<double> data, double percentile) {
    if (data.empty()) return 0.0;
    std::sort(data.begin(), data.end());
    size_t index = static_cast<size_t>((percentile / 100.0) * (data.size() - 1));
    return data[index];
}

extern "C" {

    // 1. Initialize simulation (called once at Unity start)
    __declspec(dllexport) void InitGeant4() {
        if (g_RunManager != nullptr) {
            return;
        }

        g_RunManager = new G4RunManager();
        g_RunManager->SetUserInitialization(new DetectorConstruction());
        g_RunManager->SetUserInitialization(new PhysicsList());
        g_RunManager->SetUserInitialization(new ActionInitialization());
        g_RunManager->Initialize();

        // Suppress verbose output
        G4UImanager* UI = G4UImanager::GetUIpointer();
        UI->ApplyCommand("/process/em/verbose 0");
        UI->ApplyCommand("/run/verbose 0");
        UI->ApplyCommand("/event/verbose 0");
        UI->ApplyCommand("/tracking/verbose 0");
    }

    // 2. Cleanup (called when Unity closes)
    __declspec(dllexport) void CloseGeant4() {
        // Intentionally empty - avoid cleanup issues
    }

    // 3. Single particle simulation (existing function)
    __declspec(dllexport) int RunSimulationBatch(float* outData, int maxSteps) {
        if (!g_RunManager) return 0;

        g_RunManager->BeamOn(1);

        auto eventAction = GetCurrentEventAction();
        if (!eventAction) return 0;

        const auto& records = eventAction->GetStepRecords();
        int stepsCount = static_cast<int>(records.size());

        if (stepsCount > maxSteps) stepsCount = maxSteps;

        // Format: [x, y, z, px, py, pz, e] (7 floats per step)
        int stride = 7;

        for (int i = 0; i < stepsCount; ++i) {
            const auto& step = records[i];
            int base = i * stride;

            outData[base + 0] = static_cast<float>(step.position.x() / cm);
            outData[base + 1] = static_cast<float>(step.position.y() / cm);
            outData[base + 2] = static_cast<float>(step.position.z() / cm);

            outData[base + 3] = static_cast<float>(step.momentum.x() / MeV);
            outData[base + 4] = static_cast<float>(step.momentum.y() / MeV);
            outData[base + 5] = static_cast<float>(step.momentum.z() / MeV);

            outData[base + 6] = static_cast<float>(step.kineticEnergy / MeV);
        }

        return stepsCount;
    }

    // ========================================================================
    // NEW: Batch simulation for 100,000 particles with statistics
    // ========================================================================

    // 4. Start batch simulation - returns number of particles simulated
    __declspec(dllexport) int RunBatchSimulation(int numParticles, int progressCallback) {
        if (!g_RunManager) return 0;

        g_BatchStats.Clear();
        g_BatchStats.Reserve(numParticles, 300);

        const double PHANTOM_ENTRY_X = -5.0; // cm
        const double INITIAL_ENERGY = 10.0;  // MeV

        for (int p = 0; p < numParticles; ++p) {
            // Run single particle
            g_RunManager->BeamOn(1);

            auto eventAction = GetCurrentEventAction();
            if (!eventAction) continue;

            const auto& records = eventAction->GetStepRecords();
            if (records.empty()) continue;

            // Calculate trajectory statistics
            double pathLength = 0.0;
            std::vector<double> scatterAngles;

            G4ThreeVector prevPos = records[0].position;
            G4ThreeVector prevDir(1.0, 0.0, 0.0);

            for (size_t i = 0; i < records.size(); ++i) {
                const auto& step = records[i];

                // Store step position for density visualization
                g_BatchStats.allStepPositionsX.push_back(static_cast<float>(step.position.x() / cm));
                g_BatchStats.allStepPositionsY.push_back(static_cast<float>(step.position.y() / cm));
                g_BatchStats.allStepPositionsZ.push_back(static_cast<float>(step.position.z() / cm));
                g_BatchStats.allStepEnergies.push_back(static_cast<float>(step.kineticEnergy / MeV));

                // Calculate path length
                G4ThreeVector deltaPos = step.position - prevPos;
                pathLength += deltaPos.mag() / cm;

                // Calculate scattering angle
                if (i > 0) {
                    G4ThreeVector newDir = deltaPos.unit();
                    if (deltaPos.mag() > 0.001 * mm) {
                        double angle = prevDir.angle(newDir) * 180.0 / M_PI; // degrees
                        scatterAngles.push_back(angle);
                        prevDir = newDir;
                    }
                }
                prevPos = step.position;
            }

            // Final position statistics
            const auto& lastStep = records.back();
            double finalX = lastStep.position.x() / cm;
            double finalY = lastStep.position.y() / cm;
            double finalZ = lastStep.position.z() / cm;
            double finalEnergy = lastStep.kineticEnergy / MeV;

            g_BatchStats.pathLengths.push_back(pathLength);
            g_BatchStats.finalDepths.push_back(finalX - PHANTOM_ENTRY_X);
            g_BatchStats.lateralSpreadsY.push_back(finalY);
            g_BatchStats.lateralSpreadsZ.push_back(finalZ);
            g_BatchStats.totalLateralSpreads.push_back(std::sqrt(finalY * finalY + finalZ * finalZ));
            g_BatchStats.finalEnergies.push_back(finalEnergy);

            // Scattering statistics
            if (!scatterAngles.empty()) {
                double meanAngle = CalculateMean(scatterAngles);
                double stdAngle = CalculateStdDev(scatterAngles, meanAngle);
                g_BatchStats.meanScatterAngles.push_back(meanAngle);
                g_BatchStats.scatterStdDevs.push_back(stdAngle);
            }
        }

        return numParticles;
    }

    // 5. Get batch statistics
    __declspec(dllexport) void GetBatchStatistics(float* outStats) {
        // Output format: 24 floats
        // [0-2]: Path length (mean, std, median)
        // [3-5]: Final depth (mean, std, median)
        // [6-8]: Lateral spread Y (mean, std, sigma_2)
        // [9-11]: Lateral spread Z (mean, std, sigma_2)
        // [12-14]: Total lateral spread (mean, std, sigma_3)
        // [15-17]: Final energy (mean, std, median)
        // [18-20]: Mean scatter angle (mean, std, median)
        // [21-23]: Scatter std dev (mean, std, median)

        // Path length
        double plMean = CalculateMean(g_BatchStats.pathLengths);
        outStats[0] = static_cast<float>(plMean);
        outStats[1] = static_cast<float>(CalculateStdDev(g_BatchStats.pathLengths, plMean));
        outStats[2] = static_cast<float>(CalculatePercentile(g_BatchStats.pathLengths, 50));

        // Final depth
        double fdMean = CalculateMean(g_BatchStats.finalDepths);
        outStats[3] = static_cast<float>(fdMean);
        outStats[4] = static_cast<float>(CalculateStdDev(g_BatchStats.finalDepths, fdMean));
        outStats[5] = static_cast<float>(CalculatePercentile(g_BatchStats.finalDepths, 50));

        // Lateral spread Y (with 2-sigma bounds)
        double lyMean = CalculateMean(g_BatchStats.lateralSpreadsY);
        double lyStd = CalculateStdDev(g_BatchStats.lateralSpreadsY, lyMean);
        outStats[6] = static_cast<float>(lyMean);
        outStats[7] = static_cast<float>(lyStd);
        outStats[8] = static_cast<float>(2.0 * lyStd); // 2-sigma bound

        // Lateral spread Z (with 2-sigma bounds)
        double lzMean = CalculateMean(g_BatchStats.lateralSpreadsZ);
        double lzStd = CalculateStdDev(g_BatchStats.lateralSpreadsZ, lzMean);
        outStats[9] = static_cast<float>(lzMean);
        outStats[10] = static_cast<float>(lzStd);
        outStats[11] = static_cast<float>(2.0 * lzStd); // 2-sigma bound

        // Total lateral spread (with 3-sigma bounds)
        double tlMean = CalculateMean(g_BatchStats.totalLateralSpreads);
        double tlStd = CalculateStdDev(g_BatchStats.totalLateralSpreads, tlMean);
        outStats[12] = static_cast<float>(tlMean);
        outStats[13] = static_cast<float>(tlStd);
        outStats[14] = static_cast<float>(3.0 * tlStd); // 3-sigma bound

        // Final energy
        double feMean = CalculateMean(g_BatchStats.finalEnergies);
        outStats[15] = static_cast<float>(feMean);
        outStats[16] = static_cast<float>(CalculateStdDev(g_BatchStats.finalEnergies, feMean));
        outStats[17] = static_cast<float>(CalculatePercentile(g_BatchStats.finalEnergies, 50));

        // Mean scatter angle
        double msMean = CalculateMean(g_BatchStats.meanScatterAngles);
        outStats[18] = static_cast<float>(msMean);
        outStats[19] = static_cast<float>(CalculateStdDev(g_BatchStats.meanScatterAngles, msMean));
        outStats[20] = static_cast<float>(CalculatePercentile(g_BatchStats.meanScatterAngles, 50));

        // Scatter std dev
        double ssMean = CalculateMean(g_BatchStats.scatterStdDevs);
        outStats[21] = static_cast<float>(ssMean);
        outStats[22] = static_cast<float>(CalculateStdDev(g_BatchStats.scatterStdDevs, ssMean));
        outStats[23] = static_cast<float>(CalculatePercentile(g_BatchStats.scatterStdDevs, 50));
    }

    // 6. Get trajectory data for visualization (returns step count)
    __declspec(dllexport) int GetBatchTrajectoryData(
        float* outX, float* outY, float* outZ, float* outEnergy, int maxSteps) {

        int count = static_cast<int>(g_BatchStats.allStepPositionsX.size());
        if (count > maxSteps) count = maxSteps;

        for (int i = 0; i < count; ++i) {
            outX[i] = g_BatchStats.allStepPositionsX[i];
            outY[i] = g_BatchStats.allStepPositionsY[i];
            outZ[i] = g_BatchStats.allStepPositionsZ[i];
            outEnergy[i] = g_BatchStats.allStepEnergies[i];
        }

        return count;
    }

    // 7. Get lateral distribution data for histogram (binned)
    __declspec(dllexport) void GetLateralDistribution(
        float* outBins, int numBins, float minVal, float maxVal) {

        float binWidth = (maxVal - minVal) / numBins;

        // Initialize bins to zero
        for (int i = 0; i < numBins; ++i) {
            outBins[i] = 0;
        }

        // Fill Y distribution
        for (double val : g_BatchStats.lateralSpreadsY) {
            int bin = static_cast<int>((val - minVal) / binWidth);
            if (bin >= 0 && bin < numBins) {
                outBins[bin] += 1.0f;
            }
        }
    }

    // 8. Export statistics to file
    __declspec(dllexport) int ExportStatisticsToFile(const char* filePath) {
        std::ofstream file(filePath);
        if (!file.is_open()) return 0;

        file << "# Geant4 Batch Simulation Statistics\n";
        file << "# 100,000 electrons at 10 MeV in water phantom (10x10x10 cm³)\n";
        file << "# Reference: Highland formula for MCS, ESTAR for CSDA range\n\n";

        // Summary statistics
        double plMean = CalculateMean(g_BatchStats.pathLengths);
        double fdMean = CalculateMean(g_BatchStats.finalDepths);
        double lyMean = CalculateMean(g_BatchStats.lateralSpreadsY);
        double lzMean = CalculateMean(g_BatchStats.lateralSpreadsZ);
        double tlMean = CalculateMean(g_BatchStats.totalLateralSpreads);
        double feMean = CalculateMean(g_BatchStats.finalEnergies);
        double msMean = CalculateMean(g_BatchStats.meanScatterAngles);

        file << "[SUMMARY]\n";
        file << "NumParticles=" << g_BatchStats.pathLengths.size() << "\n";
        file << "NumStepsTotal=" << g_BatchStats.allStepPositionsX.size() << "\n\n";

        file << "[PATH_LENGTH_CM]\n";
        file << "Mean=" << plMean << "\n";
        file << "StdDev=" << CalculateStdDev(g_BatchStats.pathLengths, plMean) << "\n";
        file << "Median=" << CalculatePercentile(g_BatchStats.pathLengths, 50) << "\n";
        file << "P5=" << CalculatePercentile(g_BatchStats.pathLengths, 5) << "\n";
        file << "P95=" << CalculatePercentile(g_BatchStats.pathLengths, 95) << "\n\n";

        file << "[PENETRATION_DEPTH_CM]\n";
        file << "Mean=" << fdMean << "\n";
        file << "StdDev=" << CalculateStdDev(g_BatchStats.finalDepths, fdMean) << "\n";
        file << "Median=" << CalculatePercentile(g_BatchStats.finalDepths, 50) << "\n";
        file << "CSDA_Range_Expected=4.98\n";
        file << "DetourFactor=" << (plMean / fdMean) << "\n\n";

        double lyStd = CalculateStdDev(g_BatchStats.lateralSpreadsY, lyMean);
        double lzStd = CalculateStdDev(g_BatchStats.lateralSpreadsZ, lzMean);
        double tlStd = CalculateStdDev(g_BatchStats.totalLateralSpreads, tlMean);

        file << "[LATERAL_SPREAD_Y_CM]\n";
        file << "Mean=" << lyMean << "\n";
        file << "StdDev=" << lyStd << "\n";
        file << "Sigma2_Bound=" << (2.0 * lyStd) << "\n";
        file << "Sigma3_Bound=" << (3.0 * lyStd) << "\n\n";

        file << "[LATERAL_SPREAD_Z_CM]\n";
        file << "Mean=" << lzMean << "\n";
        file << "StdDev=" << lzStd << "\n";
        file << "Sigma2_Bound=" << (2.0 * lzStd) << "\n";
        file << "Sigma3_Bound=" << (3.0 * lzStd) << "\n\n";

        file << "[LATERAL_SPREAD_TOTAL_CM]\n";
        file << "Mean=" << tlMean << "\n";
        file << "StdDev=" << tlStd << "\n";
        file << "Sigma2_Bound=" << (2.0 * tlStd) << "\n";
        file << "Sigma3_Bound=" << (3.0 * tlStd) << "\n\n";

        file << "[MULTIPLE_COULOMB_SCATTERING_DEG]\n";
        file << "MeanAngle=" << msMean << "\n";
        file << "StdDevAngle=" << CalculateStdDev(g_BatchStats.meanScatterAngles, msMean) << "\n";
        file << "Highland_Expected_RMS=" << (13.6 / (0.998 * 10.5) * std::sqrt(5.0/36.08) * 180.0 / M_PI) << "\n\n";

        file << "[ENERGY_LOSS_MEV]\n";
        file << "InitialEnergy=10.0\n";
        file << "FinalEnergyMean=" << feMean << "\n";
        file << "TotalEnergyLoss=" << (10.0 - feMean) << "\n";
        file << "BetheBloch_Expected_Loss=" << (2.0 * plMean) << "\n\n";

        file.close();
        return 1;
    }

    // 9. Get step count in batch
    __declspec(dllexport) int GetBatchStepCount() {
        return static_cast<int>(g_BatchStats.allStepPositionsX.size());
    }

    // 10. Get particle count in batch
    __declspec(dllexport) int GetBatchParticleCount() {
        return static_cast<int>(g_BatchStats.pathLengths.size());
    }

    // ========================================================================
    // PERFORMANCE MEASUREMENT FUNCTIONS
    // ========================================================================

    /// <summary>
    /// Measure average simulation time per trajectory over N runs.
    /// Returns average time in milliseconds.
    /// </summary>
    __declspec(dllexport) float MeasureGeant4Performance(int numRuns, int maxSteps) {
        if (!g_RunManager) return -1.0f;

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < numRuns; ++i) {
            g_RunManager->BeamOn(1);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Return average time in milliseconds
        float avgTimeMs = (duration.count() / 1000.0f) / numRuns;

        return avgTimeMs;
    }

    /// <summary>
    /// Measure detailed performance metrics for a single trajectory.
    /// Output: [totalTimeMs, numSteps, avgStepTimeMs]
    /// </summary>
    __declspec(dllexport) void MeasureDetailedPerformance(float* outMetrics) {
        if (!g_RunManager) {
            outMetrics[0] = -1.0f;
            outMetrics[1] = 0.0f;
            outMetrics[2] = -1.0f;
            return;
        }

        auto start = std::chrono::high_resolution_clock::now();
        g_RunManager->BeamOn(1);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        float totalTimeMs = duration.count() / 1000.0f;

        auto eventAction = GetCurrentEventAction();
        int numSteps = 0;
        if (eventAction) {
            numSteps = static_cast<int>(eventAction->GetStepRecords().size());
        }

        float avgStepTimeMs = (numSteps > 0) ? (totalTimeMs / numSteps) : 0.0f;

        outMetrics[0] = totalTimeMs;          // Total time (ms)
        outMetrics[1] = static_cast<float>(numSteps); // Number of steps
        outMetrics[2] = avgStepTimeMs;        // Average time per step (ms)
    }

    /// <summary>
    /// Run performance benchmark with detailed statistics.
    /// Returns: [meanTimeMs, stdDevMs, minTimeMs, maxTimeMs, medianTimeMs, totalSteps]
    /// Uses existing CalculateMean() function for consistency.
    /// </summary>
    __declspec(dllexport) void BenchmarkGeant4Performance(
        int numRuns,
        float* outStats,
        int statsSize
    ) {
        if (!g_RunManager || statsSize < 6) return;

        std::vector<double> times;  // Use double to match existing CalculateMean signature
        times.reserve(numRuns);
        int totalSteps = 0;

        for (int i = 0; i < numRuns; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            g_RunManager->BeamOn(1);
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            times.push_back(duration.count() / 1000.0); // Convert to ms as double

            auto eventAction = GetCurrentEventAction();
            if (eventAction) {
                totalSteps += static_cast<int>(eventAction->GetStepRecords().size());
            }
        }

        // Calculate statistics using existing CalculateMean function
        double mean = CalculateMean(times);
        double stdDev = 0.0;

        if (times.size() > 1) {
            double variance = 0.0;
            for (double t : times) {
                double diff = t - mean;
                variance += diff * diff;
            }
            stdDev = std::sqrt(variance / times.size());
        }

        std::sort(times.begin(), times.end());
        double minTime = times.front();
        double maxTime = times.back();
        double medianTime = times[times.size() / 2];

        // Convert to float for output (C# interop uses float arrays)
        outStats[0] = static_cast<float>(mean);
        outStats[1] = static_cast<float>(stdDev);
        outStats[2] = static_cast<float>(minTime);
        outStats[3] = static_cast<float>(maxTime);
        outStats[4] = static_cast<float>(medianTime);
        outStats[5] = static_cast<float>(totalSteps);
    }
} // extern "C"