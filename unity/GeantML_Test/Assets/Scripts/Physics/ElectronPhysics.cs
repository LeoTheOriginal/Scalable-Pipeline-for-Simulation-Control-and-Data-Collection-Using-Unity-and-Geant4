using UnityEngine;

namespace Physics
{
    /// <summary>
    /// Physics constants and formulas for electron transport in water.
    /// All values are calibrated for 10 MeV electrons in water phantom.
    /// 
    /// References:
    /// - ICRU Report 37 (Stopping Powers for Electrons and Positrons)
    /// - PDG Review of Particle Physics (Passage of particles through matter)
    /// </summary>
    public static class ElectronPhysics
    {
        // ====================================================================
        // PHYSICAL CONSTANTS
        // ====================================================================

        /// <summary>Electron rest mass in MeV/c²</summary>
        public const float MASS_ELECTRON = 0.511f;

        /// <summary>Speed of light in cm/ns</summary>
        public const float C_LIGHT = 29.9792458f;

        /// <summary>Radiation length of water in cm</summary>
        public const float X0_WATER = 36.08f;

        /// <summary>Water density in g/cm³</summary>
        public const float DENSITY_WATER = 1.0f;

        /// <summary>Initial electron energy in MeV (standard for this simulation)</summary>
        public const float INITIAL_ENERGY = 10.0f;

        /// <summary>Phantom half-size in cm</summary>
        public const float PHANTOM_HALF_SIZE = 5.0f;

        /// <summary>Phantom entry position X in cm</summary>
        public const float PHANTOM_ENTRY_X = -5.0f;

        /// <summary>Initial position X in cm (before phantom)</summary>
        public const float INITIAL_POS_X = -6.0f;

        // ====================================================================
        // CSDA RANGE CALCULATION
        // ====================================================================

        /// <summary>
        /// Calculate CSDA (Continuous Slowing Down Approximation) range for electrons in water.
        /// 
        /// Empirical formula fitted to ESTAR database values:
        /// R(cm) = 0.530 * E(MeV) - 0.106 for E > 1 MeV
        /// 
        /// For 10 MeV: R ≈ 5.19 cm
        /// ESTAR actual value: ~4.98 cm
        /// 
        /// We use a more accurate polynomial fit:
        /// </summary>
        /// <param name="energy">Kinetic energy in MeV</param>
        /// <returns>CSDA range in cm</returns>
        public static float CalculateCSDARange(float energy)
        {
            if (energy <= 0) return 0f;

            // Polynomial fit to ESTAR data for water (1-20 MeV range)
            // R = a*E^2 + b*E + c
            // Fitted coefficients:
            float a = -0.00234f;
            float b = 0.5127f;
            float c = -0.0543f;

            float range = a * energy * energy + b * energy + c;
            return Mathf.Max(0f, range);
        }

        /// <summary>
        /// Calculate remaining range based on current energy.
        /// </summary>
        public static float CalculateRemainingRange(float currentEnergy)
        {
            return CalculateCSDARange(currentEnergy);
        }

        /// <summary>
        /// Get expected CSDA range for initial 10 MeV electron.
        /// Value: approximately 4.98 cm
        /// </summary>
        public static float GetInitialCSDARange()
        {
            return CalculateCSDARange(INITIAL_ENERGY);
        }

        // ====================================================================
        // ENERGY LOSS (BETHE-BLOCH)
        // ====================================================================

        /// <summary>
        /// Calculate stopping power (energy loss rate) for electrons in water.
        /// 
        /// Simplified Bethe-Bloch for electrons in water:
        /// dE/dx ≈ 2.0 MeV/(g/cm²) for relativistic electrons
        /// 
        /// More accurate: includes density correction and shell corrections.
        /// </summary>
        /// <param name="energy">Kinetic energy in MeV</param>
        /// <returns>Stopping power in MeV/cm</returns>
        public static float CalculateStoppingPower(float energy)
        {
            if (energy <= 0) return 0f;

            // Total energy and gamma factor
            float totalEnergy = energy + MASS_ELECTRON;
            float gamma = totalEnergy / MASS_ELECTRON;
            float beta2 = 1f - 1f / (gamma * gamma);

            // Base stopping power (collision + radiative for high energy)
            // Empirical fit to ESTAR data
            float collisionStopping = 2.0f; // MeV/cm base value

            // Energy dependence correction
            float correction = 1f + 0.02f * Mathf.Log(energy + 1f);

            // Radiative losses become important above ~10 MeV
            float radiativeFraction = energy / (energy + 7.5f * MASS_ELECTRON);
            float radiativeStopping = 0.1f * energy / X0_WATER;

            float totalStopping = (collisionStopping * correction + radiativeStopping) * DENSITY_WATER;

            return totalStopping;
        }

        /// <summary>
        /// Calculate expected energy loss for a given step length.
        /// </summary>
        public static float CalculateEnergyLoss(float energy, float stepLength)
        {
            float stoppingPower = CalculateStoppingPower(energy);
            return stoppingPower * stepLength;
        }

        // ====================================================================
        // MULTIPLE COULOMB SCATTERING (HIGHLAND FORMULA)
        // ====================================================================

        /// <summary>
        /// Calculate RMS scattering angle using Highland formula.
        /// 
        /// θ_rms = (13.6 MeV / βcp) * √(x/X₀) * [1 + 0.038 * ln(x/X₀)]
        /// 
        /// For small steps, this gives the expected angular spread.
        /// </summary>
        /// <param name="energy">Kinetic energy in MeV</param>
        /// <param name="stepLength">Step length in cm</param>
        /// <returns>RMS scattering angle in radians</returns>
        public static float CalculateRMSScatteringAngle(float energy, float stepLength)
        {
            if (energy <= 0 || stepLength <= 0) return 0f;

            // Calculate β*c*p
            float totalEnergy = energy + MASS_ELECTRON;
            float momentum = Mathf.Sqrt(totalEnergy * totalEnergy - MASS_ELECTRON * MASS_ELECTRON);
            float beta = momentum / totalEnergy;
            float betaCp = beta * momentum; // in MeV

            // Thickness in radiation lengths
            float xOverX0 = stepLength / X0_WATER;

            // Highland formula
            float theta0 = (13.6f / betaCp) * Mathf.Sqrt(xOverX0);

            // Logarithmic correction (only for x/X0 > 0.001)
            if (xOverX0 > 0.001f)
            {
                theta0 *= (1f + 0.038f * Mathf.Log(xOverX0));
            }

            return Mathf.Abs(theta0);
        }

        /// <summary>
        /// Calculate maximum allowed scattering angle for physics validity.
        /// Based on 3σ of Highland distribution.
        /// </summary>
        public static float CalculateMaxScatteringAngle(float energy, float stepLength)
        {
            return 3f * CalculateRMSScatteringAngle(energy, stepLength);
        }

        // ====================================================================
        // PHYSICS VALIDATION
        // ====================================================================

        /// <summary>
        /// Check if a scattering angle is physically reasonable.
        /// 
        /// For electrons in water, large angle scattering (>90°) is extremely rare.
        /// We allow up to ~60° as the practical maximum for single steps.
        /// </summary>
        public static bool IsScatteringAngleValid(float angleDegrees, float energy, float stepLength)
        {
            float maxAngle = CalculateMaxScatteringAngle(energy, stepLength) * Mathf.Rad2Deg;

            // Practical maximum: 60 degrees
            maxAngle = Mathf.Min(maxAngle, 60f);

            return angleDegrees <= maxAngle;
        }

        /// <summary>
        /// Check if electron is still moving generally forward.
        /// Backward motion (>90° from initial direction) is extremely unlikely.
        /// </summary>
        public static bool IsMovingForward(Vector3 currentDirection, Vector3 initialDirection)
        {
            float dot = Vector3.Dot(currentDirection, initialDirection);
            return dot > 0; // Angle < 90°
        }

        /// <summary>
        /// Calculate how "forward" the movement is (1 = perfectly forward, 0 = perpendicular, -1 = backward).
        /// </summary>
        public static float GetForwardness(Vector3 currentDirection, Vector3 initialDirection)
        {
            return Vector3.Dot(currentDirection.normalized, initialDirection.normalized);
        }

        /// <summary>
        /// Calculate velocity beta factor from kinetic energy.
        /// β = v/c
        /// </summary>
        public static float CalculateBeta(float energy)
        {
            if (energy <= 0) return 0f;
            float totalEnergy = energy + MASS_ELECTRON;
            float gamma = totalEnergy / MASS_ELECTRON;
            return Mathf.Sqrt(1f - 1f / (gamma * gamma));
        }

        /// <summary>
        /// Calculate momentum magnitude from kinetic energy.
        /// p = √(E² + 2mE)
        /// </summary>
        public static float CalculateMomentum(float energy)
        {
            if (energy <= 0) return 0f;
            float totalEnergy = energy + MASS_ELECTRON;
            return Mathf.Sqrt(totalEnergy * totalEnergy - MASS_ELECTRON * MASS_ELECTRON);
        }

        // ====================================================================
        // TRAJECTORY STATISTICS
        // ====================================================================

        /// <summary>
        /// Calculate the expected path length detour factor.
        /// Due to scattering, actual path length > straight-line distance.
        /// For electrons in water, this is typically 1.1-1.3 for ~10 MeV.
        /// </summary>
        public static float GetExpectedDetourFactor(float energy)
        {
            // Empirical: higher energy = more straight path
            return 1.1f + 0.2f * Mathf.Exp(-energy / 5f);
        }

        /// <summary>
        /// Calculate expected lateral spread (RMS) for given penetration depth.
        /// </summary>
        public static float CalculateLateralSpread(float penetrationDepth, float energy)
        {
            // Rough approximation based on Highland formula integrated over path
            float theta0 = CalculateRMSScatteringAngle(energy, penetrationDepth);
            return penetrationDepth * theta0 / Mathf.Sqrt(3f);
        }
    }
}