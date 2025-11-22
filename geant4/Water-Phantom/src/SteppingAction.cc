#include "SteppingAction.hh"
#include "EventAction.hh"
#include <G4Step.hh>
#include <G4Track.hh>
#include <G4ParticleDefinition.hh>
#include <G4VProcess.hh>
#include <G4SystemOfUnits.hh>
#include <G4PhysicalConstants.hh>
#include <cmath>

// Funkcja pomocnicza (bez zmian)
static void CalculateParticleInteraction(const G4Step* step,
                                        G4double& outAngle,
                                        G4double& outAcceleration) {
    const G4ThreeVector preDir  = step->GetPreStepPoint()->GetMomentumDirection();
    const G4ThreeVector postDir = step->GetPostStepPoint()->GetMomentumDirection();
    outAngle = preDir.angle(postDir);

    G4double preKE  = step->GetPreStepPoint()->GetKineticEnergy();
    G4double postKE = step->GetPostStepPoint()->GetKineticEnergy();

    const G4Track* track = step->GetTrack();
    G4double mass = track->GetParticleDefinition()->GetPDGMass();

    G4double totalE_pre  = preKE + mass;
    G4double totalE_post = postKE + mass;

    G4double beta_pre  = (totalE_pre > 0.) ? std::sqrt(1. - std::pow(mass/totalE_pre, 2)) : 0.;
    G4double beta_post = (totalE_post > 0.) ? std::sqrt(1. - std::pow(mass/totalE_post, 2)) : 0.;

    G4double v_pre  = beta_pre  * c_light;
    G4double v_post = beta_post * c_light;

    G4double dt = step->GetDeltaTime();
    outAcceleration = (dt > 0) ? (v_post - v_pre) / dt : 0;
}

SteppingAction::SteppingAction(EventAction* eventAction)
 : fEventAction(eventAction)
{
    // Usunięto inicjalizację fStepCounter, bo już go nie używamy
}

SteppingAction::~SteppingAction() = default;

void SteppingAction::UserSteppingAction(const G4Step* step) {

    // ========================================================================
    // NAPRAWA: Używamy wbudowanego licznika Geant4 zamiast własnego zmiennego
    // GetCurrentStepNumber() resetuje się sam dla każdej cząstki.
    // ========================================================================
    G4int currentStep = step->GetTrack()->GetCurrentStepNumber();

    // if (currentStep > MAX_STEPS_PER_PARTICLE) {
    //     step->GetTrack()->SetTrackStatus(fStopAndKill);
    //     return;
    // }

    G4StepPoint* prePoint  = step->GetPreStepPoint();
    G4VPhysicalVolume* preVol  = prePoint->GetPhysicalVolume();
    G4VPhysicalVolume* postVol = step->GetPostStepPoint()->GetPhysicalVolume();

    if (!preVol) return;

    G4String preName  = preVol->GetName();
    G4String postName = postVol ? postVol->GetName() : "OutOfWorld";

    if (preName == "Phantom" && postName != "Phantom") {
        step->GetTrack()->SetTrackStatus(fStopAndKill);
        G4double dE = step->GetTotalEnergyDeposit();
        if (dE > 0) fEventAction->AddEnergyDeposit(dE);
        return;
    }

    if (step->GetTrack()->GetTrackID() != 1) {
        HandleSecondaryParticle(step);
        return;
    }

    bool inPhantom = (preName == "Phantom" || postName == "Phantom");
    if (inPhantom) {
        RecordPrimaryParticleStep(step);
    }
}

void SteppingAction::RecordPrimaryParticleStep(const G4Step* step) {
    G4StepPoint* postPoint = step->GetPostStepPoint();

    G4ThreeVector position = postPoint->GetPosition();
    G4ThreeVector momentum = postPoint->GetMomentum();
    G4double kineticE = postPoint->GetKineticEnergy();
    G4double dE = step->GetTotalEnergyDeposit();
    
    G4String processName = "Transportation";
    if (postPoint->GetProcessDefinedStep() != nullptr) {
        processName = postPoint->GetProcessDefinedStep()->GetProcessName();
    }
    
    if (dE > 0) fEventAction->AddEnergyDeposit(dE);
    
    G4double scatterAngle = 0.0;
    G4double acceleration = 0.0;
    CalculateParticleInteraction(step, scatterAngle, acceleration);
    
    StepRecord record;
    record.position        = position;
    record.momentum        = momentum;
    record.kineticEnergy   = kineticE;
    record.energyDeposited = dE;
    record.scatterAngle    = scatterAngle;
    record.acceleration    = acceleration;
    record.processName     = processName;
    
    fEventAction->AddStepRecord(record);
}

void SteppingAction::HandleSecondaryParticle(const G4Step* step) {
    G4double dE = step->GetTotalEnergyDeposit();
    if (dE > 0) {
        G4VPhysicalVolume* preVol = step->GetPreStepPoint()->GetPhysicalVolume();
        if (preVol && preVol->GetName() == "Phantom") {
            fEventAction->AddEnergyDeposit(dE);
        }
    }
}

bool SteppingAction::ShouldRecordStep(const G4Step*) const { return true; }