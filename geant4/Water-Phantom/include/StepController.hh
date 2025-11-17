#pragma once

#include "G4Types.hh"
#include "G4ThreeVector.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4String.hh"

#include <memory>

/**
 * @brief Step-by-step control interface for Geant4
 * 
 * Allows external control of particle stepping for real-time
 * integration with Unity ML-Agents
 * 
 * Usage:
 *   1. InitializeParticle() - set initial conditions
 *   2. ExecuteStep() - advance one step
 *   3. Repeat step 2 until IsParticleAlive() returns false
 */
class StepController {
public:
    // Singleton instance
    static StepController& Instance();
    
    // Delete copy/move constructors
    StepController(const StepController&) = delete;
    StepController& operator=(const StepController&) = delete;
    StepController(StepController&&) = delete;
    StepController& operator=(StepController&&) = delete;
    
    /**
     * @brief Step result data structure
     */
    struct StepResult {
        G4ThreeVector position;           // Post-step position (cm)
        G4ThreeVector direction;          // Post-step direction (normalized)
        G4double energy;                  // Post-step kinetic energy (MeV)
        G4double energyDeposited;         // Energy deposited this step (MeV)
        G4double stepLength;              // Step length (cm)
        G4String processName;             // Physics process name
        G4bool particleStopped;           // True if particle stopped/killed
        G4int trackID;                    // Track ID
    };
    
    /**
     * @brief Initialize particle with given conditions
     * 
     * @param particleType Particle name ("e-", "gamma", "proton", etc.)
     * @param energy Initial kinetic energy (MeV)
     * @param position Initial position (cm)
     * @param direction Initial direction (normalized)
     * @return true if successful, false otherwise
     */
    G4bool InitializeParticle(
        const G4String& particleType,
        G4double energy,
        const G4ThreeVector& position,
        const G4ThreeVector& direction
    );
    
    /**
     * @brief Execute single step
     * 
     * @return StepResult containing all step information
     */
    StepResult ExecuteStep();
    
    /**
     * @brief Check if particle is still alive
     * 
     * @return true if particle can continue stepping
     */
    G4bool IsParticleAlive() const;
    
    /**
     * @brief Reset controller state
     */
    void Reset();
    
    /**
     * @brief Get current step number
     */
    G4int GetStepNumber() const { return fStepNumber; }
    
    /**
     * @brief Get total energy deposited
     */
    G4double GetTotalEnergyDeposit() const { return fTotalEnergyDeposit; }
    
private:
    // Private constructor (singleton)
    StepController();
    ~StepController();
    
    // Current track (managed by Geant4)
    G4Track* fCurrentTrack;
    
    // Step counter
    G4int fStepNumber;
    
    // Total energy deposited
    G4double fTotalEnergyDeposit;
    
    // Initialization flag
    G4bool fIsInitialized;
};