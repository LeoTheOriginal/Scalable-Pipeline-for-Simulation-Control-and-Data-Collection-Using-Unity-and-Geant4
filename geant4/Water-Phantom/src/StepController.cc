#include "StepController.hh"

#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4DynamicParticle.hh"
#include "G4PrimaryParticle.hh"
#include "G4TrackingManager.hh"
#include "G4SteppingManager.hh"
#include "G4RunManager.hh"
#include "G4EventManager.hh"

#include "G4ios.hh"

// Singleton instance
StepController& StepController::Instance() {
    static StepController instance;
    return instance;
}

StepController::StepController()
    : fCurrentTrack(nullptr)
    , fStepNumber(0)
    , fTotalEnergyDeposit(0.0)
    , fIsInitialized(false)
{
    G4cout << "[StepController] Constructed" << G4endl;
}

StepController::~StepController() {
    Reset();
    G4cout << "[StepController] Destroyed" << G4endl;
}

G4bool StepController::InitializeParticle(
    const G4String& particleType,
    G4double energy,
    const G4ThreeVector& position,
    const G4ThreeVector& direction)
{
    // Reset previous state
    Reset();
    
    // Get particle definition
    G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
    G4ParticleDefinition* particleDef = particleTable->FindParticle(particleType);
    
    if (!particleDef) {
        G4cerr << "[StepController] ERROR: Unknown particle type: " 
               << particleType << G4endl;
        return false;
    }
    
    // Create dynamic particle
    G4ThreeVector normalizedDir = direction.unit();
    G4DynamicParticle* dynamicParticle = new G4DynamicParticle(
        particleDef,
        normalizedDir,
        energy
    );
    
    // Create track
    fCurrentTrack = new G4Track(dynamicParticle, 0.0, position);
    fCurrentTrack->SetTrackID(1);  // Primary track
    
    // Initialize
    fStepNumber = 0;
    fTotalEnergyDeposit = 0.0;
    fIsInitialized = true;
    
    G4cout << "[StepController] Particle initialized:" << G4endl;
    G4cout << "  Type: " << particleType << G4endl;
    G4cout << "  Energy: " << energy << " MeV" << G4endl;
    G4cout << "  Position: " << position << " cm" << G4endl;
    G4cout << "  Direction: " << normalizedDir << G4endl;
    
    return true;
}

StepController::StepResult StepController::ExecuteStep() {
    StepResult result;
    
    if (!fIsInitialized || !fCurrentTrack) {
        G4cerr << "[StepController] ERROR: Not initialized!" << G4endl;
        result.particleStopped = true;
        return result;
    }
    
    if (!IsParticleAlive()) {
        G4cout << "[StepController] Particle already stopped" << G4endl;
        result.particleStopped = true;
        return result;
    }
    
    // Get stepping manager
    G4SteppingManager* steppingManager = G4EventManager::GetEventManager()
                                        ->GetTrackingManager()
                                        ->GetSteppingManager();
    
    if (!steppingManager) {
        G4cerr << "[StepController] ERROR: No stepping manager!" << G4endl;
        result.particleStopped = true;
        return result;
    }
    
    // Set current track
    steppingManager->SetInitialStep(fCurrentTrack);
    
    // Execute one step
    steppingManager->Stepping();
    
    // Get step information
    G4Step* step = fCurrentTrack->GetStep();
    
    if (step) {
        // Post-step point
        G4StepPoint* postPoint = step->GetPostStepPoint();
        
        // Fill result
        result.position = postPoint->GetPosition();
        result.direction = postPoint->GetMomentumDirection();
        result.energy = postPoint->GetKineticEnergy();
        result.energyDeposited = step->GetTotalEnergyDeposit();
        result.stepLength = step->GetStepLength();
        result.trackID = fCurrentTrack->GetTrackID();
        
        // Process name
        const G4VProcess* process = postPoint->GetProcessDefinedStep();
        result.processName = process ? process->GetProcessName() : "Transportation";
        
        // Update totals
        fTotalEnergyDeposit += result.energyDeposited;
        fStepNumber++;
        
        // Check if particle stopped
        result.particleStopped = (fCurrentTrack->GetTrackStatus() != fAlive);
        
        // Debug output
        if (fStepNumber % 10 == 0) {
            G4cout << "[StepController] Step " << fStepNumber 
                   << ": E=" << result.energy << " MeV, "
                   << "dE=" << result.energyDeposited << " MeV, "
                   << "process=" << result.processName << G4endl;
        }
    } else {
        G4cerr << "[StepController] ERROR: No step information!" << G4endl;
        result.particleStopped = true;
    }
    
    return result;
}

G4bool StepController::IsParticleAlive() const {
    if (!fCurrentTrack) {
        return false;
    }
    
    return (fCurrentTrack->GetTrackStatus() == fAlive);
}

void StepController::Reset() {
    if (fCurrentTrack) {
        delete fCurrentTrack;
        fCurrentTrack = nullptr;
    }
    
    fStepNumber = 0;
    fTotalEnergyDeposit = 0.0;
    fIsInitialized = false;
    
    G4cout << "[StepController] Reset" << G4endl;
}