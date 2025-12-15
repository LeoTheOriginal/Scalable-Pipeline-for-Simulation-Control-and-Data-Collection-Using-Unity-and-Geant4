using UnityEngine;

namespace Physics
{
    /// <summary>
    /// Helper class for calculating rewards based on normal distribution theory.
    /// Used to encourage RL agents to produce realistic lateral spread patterns
    /// matching Geant4 multiple Coulomb scattering behavior.
    /// 
    /// Reference: Highland formula, PDG 2023 Review of Particle Physics Section 34.3
    /// </summary>
    public static class NormalDistributionRewards
    {
        // ====================================================================
        // CONSTANTS FROM GEANT4 STATISTICS (100k particles @ 10 MeV in water)
        // ====================================================================

        /// <summary>Expected mean lateral spread Y [cm] - should be ~0 (symmetric)</summary>
        public const float EXPECTED_LATERAL_Y_MEAN = 0.0f;

        /// <summary>Expected std dev of lateral Y [cm] from Geant4 batch</summary>
        public const float EXPECTED_LATERAL_Y_STD = 0.3f;

        /// <summary>Expected mean lateral spread Z [cm] - should be ~0 (symmetric)</summary>
        public const float EXPECTED_LATERAL_Z_MEAN = 0.0f;

        /// <summary>Expected std dev of lateral Z [cm] from Geant4 batch</summary>
        public const float EXPECTED_LATERAL_Z_STD = 0.3f;

        /// <summary>Expected total lateral spread mean [cm]</summary>
        public const float EXPECTED_LATERAL_TOTAL_MEAN = 0.35f;

        /// <summary>Expected total lateral spread std [cm]</summary>
        public const float EXPECTED_LATERAL_TOTAL_STD = 0.2f;

        /// <summary>2-sigma bound for lateral deviation</summary>
        public const float SIGMA_2_BOUND = 0.6f;

        /// <summary>3-sigma bound for lateral deviation</summary>
        public const float SIGMA_3_BOUND = 0.9f;

        // ====================================================================
        // GAUSSIAN PDF AND REWARD FUNCTIONS
        // ====================================================================

        /// <summary>
        /// Calculate Gaussian probability density function value.
        /// </summary>
        /// <param name="x">Value</param>
        /// <param name="mean">Distribution mean</param>
        /// <param name="std">Distribution standard deviation</param>
        /// <returns>PDF value (not normalized to max 1)</returns>
        public static float GaussianPDF(float x, float mean, float std)
        {
            if (std <= 0) return 0f;

            float z = (x - mean) / std;
            return Mathf.Exp(-0.5f * z * z);
        }

        /// <summary>
        /// Calculate z-score (number of standard deviations from mean).
        /// </summary>
        public static float CalculateZScore(float x, float mean, float std)
        {
            if (std <= 0) return float.MaxValue;
            return Mathf.Abs((x - mean) / std);
        }

        /// <summary>
        /// Calculate reward based on how well a value matches expected normal distribution.
        /// 
        /// Reward structure:
        /// - Within 1σ: Full reward (1.0)
        /// - 1σ to 2σ: Linear decay (1.0 → 0.5)
        /// - 2σ to 3σ: Linear decay (0.5 → 0.0)
        /// - Beyond 3σ: Penalty (negative)
        /// </summary>
        /// <param name="value">Actual value</param>
        /// <param name="mean">Expected mean</param>
        /// <param name="std">Expected standard deviation</param>
        /// <param name="maxReward">Maximum reward for perfect match</param>
        /// <param name="maxPenalty">Maximum penalty for extreme deviation</param>
        /// <returns>Reward value</returns>
        public static float CalculateNormalDistributionReward(
            float value, float mean, float std,
            float maxReward = 1.0f, float maxPenalty = 1.0f)
        {
            float z = CalculateZScore(value, mean, std);

            if (z <= 1.0f)
            {
                // Within 1σ - full reward
                return maxReward;
            }
            else if (z <= 2.0f)
            {
                // 1σ to 2σ - linear decay from 1.0 to 0.5
                float t = (z - 1.0f); // 0 to 1
                return maxReward * (1.0f - 0.5f * t);
            }
            else if (z <= 3.0f)
            {
                // 2σ to 3σ - linear decay from 0.5 to 0
                float t = (z - 2.0f); // 0 to 1
                return maxReward * 0.5f * (1.0f - t);
            }
            else
            {
                // Beyond 3σ - penalty
                float excess = z - 3.0f;
                return -maxPenalty * Mathf.Min(excess, 2.0f); // Cap penalty
            }
        }

        // ====================================================================
        // LATERAL SPREAD REWARDS
        // ====================================================================

        /// <summary>
        /// Calculate lateral deviation reward using Geant4 statistics.
        /// Encourages normal distribution pattern matching reference image.
        /// </summary>
        /// <param name="lateralY">Current Y position relative to entry point [cm]</param>
        /// <param name="lateralZ">Current Z position relative to entry point [cm]</param>
        /// <param name="depthFraction">Fractional depth in phantom (0-1)</param>
        /// <param name="weight">Reward weight multiplier</param>
        /// <returns>Reward value</returns>
        public static float CalculateLateralDeviationReward(
            float lateralY, float lateralZ, float depthFraction, float weight = 1.0f)
        {
            // Scale expected spread by depth (lateral spread increases with depth)
            float depthScale = Mathf.Sqrt(Mathf.Max(0.1f, depthFraction));
            float expectedStdY = EXPECTED_LATERAL_Y_STD * depthScale;
            float expectedStdZ = EXPECTED_LATERAL_Z_STD * depthScale;

            // Calculate rewards for Y and Z separately (should be independent Gaussians)
            float rewardY = CalculateNormalDistributionReward(
                lateralY, EXPECTED_LATERAL_Y_MEAN, expectedStdY, weight, weight);
            float rewardZ = CalculateNormalDistributionReward(
                lateralZ, EXPECTED_LATERAL_Z_MEAN, expectedStdZ, weight, weight);

            // Combined reward (average of Y and Z)
            return (rewardY + rewardZ) * 0.5f;
        }

        /// <summary>
        /// Calculate total lateral spread reward (radial distance from beam axis).
        /// </summary>
        /// <param name="totalLateral">sqrt(Y² + Z²) [cm]</param>
        /// <param name="depthFraction">Fractional depth in phantom (0-1)</param>
        /// <param name="weight">Reward weight multiplier</param>
        /// <returns>Reward value</returns>
        public static float CalculateTotalLateralReward(
            float totalLateral, float depthFraction, float weight = 1.0f)
        {
            // Scale by depth
            float depthScale = Mathf.Sqrt(Mathf.Max(0.1f, depthFraction));
            float expectedMean = EXPECTED_LATERAL_TOTAL_MEAN * depthScale;
            float expectedStd = EXPECTED_LATERAL_TOTAL_STD * depthScale;

            return CalculateNormalDistributionReward(
                totalLateral, expectedMean, expectedStd, weight, weight);
        }

        // ====================================================================
        // STEP-BY-STEP LATERAL CHANGE REWARDS
        // ====================================================================

        /// <summary>
        /// Calculate reward for step-level lateral change (should be Gaussian).
        /// Each step's lateral deviation should follow Highland formula.
        /// </summary>
        /// <param name="deltaLateral">Change in lateral position this step [cm]</param>
        /// <param name="stepSize">Step length [cm]</param>
        /// <param name="energy">Current energy [MeV]</param>
        /// <param name="weight">Reward weight</param>
        /// <returns>Reward value</returns>
        public static float CalculateStepLateralReward(
            float deltaLateral, float stepSize, float energy, float weight = 1.0f)
        {
            // Expected lateral spread per step from Highland formula
            float theta0 = ElectronPhysics.CalculateRMSScatteringAngle(energy, stepSize);
            float expectedLateralStd = stepSize * theta0 / Mathf.Sqrt(3f); // Integrate over step

            // Very small expected values - use absolute tolerance
            float minStd = 0.001f;
            expectedLateralStd = Mathf.Max(expectedLateralStd, minStd);

            return CalculateNormalDistributionReward(
                Mathf.Abs(deltaLateral), 0f, expectedLateralStd, weight, weight * 0.5f);
        }

        // ====================================================================
        // SCATTERING ANGLE REWARDS (HIGHLAND FORMULA)
        // ====================================================================

        /// <summary>
        /// Calculate reward for scattering angle matching Highland formula.
        /// Uses ±2σ and ±3σ bounds for reward/penalty structure.
        /// </summary>
        /// <param name="scatterAngle">Actual scattering angle [degrees]</param>
        /// <param name="energy">Current energy [MeV]</param>
        /// <param name="stepSize">Step length [cm]</param>
        /// <param name="weight">Reward weight</param>
        /// <returns>Reward value</returns>
        public static float CalculateScatteringAngleReward(
            float scatterAngle, float energy, float stepSize, float weight = 1.0f)
        {
            // Highland formula RMS angle
            float theta0Rad = ElectronPhysics.CalculateRMSScatteringAngle(energy, stepSize);
            float theta0Deg = theta0Rad * Mathf.Rad2Deg;

            // Ensure minimum expected scatter (even at high energy)
            theta0Deg = Mathf.Max(theta0Deg, 1.0f);

            // Scattering angle should follow half-normal distribution (angles are positive)
            // We use the standard deviation directly
            float z = scatterAngle / theta0Deg;

            // Reward structure based on sigma
            if (z <= 2.0f)
            {
                // Within 2σ - good, full to half reward
                return weight * (1.0f - 0.25f * z);
            }
            else if (z <= 3.0f)
            {
                // 2σ to 3σ - acceptable, reduced reward
                float t = (z - 2.0f);
                return weight * 0.5f * (1.0f - t);
            }
            else if (z <= 4.0f)
            {
                // 3σ to 4σ - rare but possible, small penalty
                return -weight * 0.5f * (z - 3.0f);
            }
            else
            {
                // Beyond 4σ - very rare, larger penalty
                return -weight * Mathf.Min(z - 3.0f, 3.0f);
            }
        }

        // ====================================================================
        // COMBINED REWARD FOR REALISTIC DISTRIBUTION
        // ====================================================================

        /// <summary>
        /// Calculate combined reward that encourages normal distribution formation.
        /// Call this at episode end to reward overall distribution quality.
        /// </summary>
        /// <param name="lateralValues">Array of lateral positions throughout trajectory</param>
        /// <param name="weight">Overall weight</param>
        /// <returns>Distribution quality reward</returns>
        public static float CalculateDistributionQualityReward(float[] lateralValues, float weight = 1.0f)
        {
            if (lateralValues == null || lateralValues.Length < 10)
                return 0f;

            // Calculate mean and std of agent's lateral distribution
            float sum = 0f;
            foreach (float v in lateralValues) sum += v;
            float mean = sum / lateralValues.Length;

            float variance = 0f;
            foreach (float v in lateralValues)
            {
                float diff = v - mean;
                variance += diff * diff;
            }
            float std = Mathf.Sqrt(variance / lateralValues.Length);

            // Compare to expected distribution
            float meanError = Mathf.Abs(mean - EXPECTED_LATERAL_Y_MEAN);
            float stdError = Mathf.Abs(std - EXPECTED_LATERAL_Y_STD);

            float reward = 0f;

            // Reward for mean close to 0
            if (meanError < 0.1f)
            {
                reward += weight * 0.5f;
            }
            else if (meanError < 0.3f)
            {
                reward += weight * 0.25f;
            }

            // Reward for std close to expected
            if (stdError < 0.1f)
            {
                reward += weight * 0.5f;
            }
            else if (stdError < 0.2f)
            {
                reward += weight * 0.25f;
            }

            return reward;
        }

        // ====================================================================
        // SIGMA BOUNDS HELPERS
        // ====================================================================

        /// <summary>
        /// Check if value is within n-sigma bound.
        /// </summary>
        public static bool IsWithinSigma(float value, float mean, float std, float nSigma)
        {
            return Mathf.Abs(value - mean) <= nSigma * std;
        }

        /// <summary>
        /// Get the sigma level (how many standard deviations from mean).
        /// </summary>
        public static float GetSigmaLevel(float value, float mean, float std)
        {
            if (std <= 0) return float.MaxValue;
            return Mathf.Abs(value - mean) / std;
        }

        /// <summary>
        /// Get categorical sigma region (1, 2, 3, or >3).
        /// </summary>
        public static int GetSigmaRegion(float value, float mean, float std)
        {
            float z = GetSigmaLevel(value, mean, std);

            if (z <= 1.0f) return 1;
            if (z <= 2.0f) return 2;
            if (z <= 3.0f) return 3;
            return 4; // Beyond 3 sigma
        }
    }
}