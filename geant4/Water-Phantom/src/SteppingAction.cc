#include "SteppingAction.hh"
#include "EventAction.hh"
#include "UnityMLInterface.hh"
#include <G4Step.hh>
#include <G4Track.hh>
#include <G4ParticleDefinition.hh>
#include <G4VProcess.hh>
#include <G4SystemOfUnits.hh>
#include <G4PhysicalConstants.hh>  
#include <cmath>

static void CalculateParticleInteraction(const G4Step* step, G4double& outAngle, G4double& outAcceleration) {
   
    const G4ThreeVector preDir  = step->GetPreStepPoint()->GetMomentumDirection();
    const G4ThreeVector postDir = step->GetPostStepPoint()->GetMomentumDirection();
    
    outAngle = preDir.angle(postDir);  // in radians

    
    G4double preKE  = step->GetPreStepPoint()->GetKineticEnergy();
    G4double postKE = step->GetPostStepPoint()->GetKineticEnergy();
   
    const G4Track* track = step->GetTrack();
    const G4ParticleDefinition* partDef = track->GetParticleDefinition();
    G4double mass = partDef->GetPDGMass();             
    G4double totalE_pre  = preKE + mass;
    G4double totalE_post = postKE + mass;
   
    G4double beta_pre  = (totalE_pre > 0.) ? std::sqrt(1. - std::pow(mass/totalE_pre, 2)) : 0.;
    G4double beta_post = (totalE_post > 0.) ? std::sqrt(1. - std::pow(mass/totalE_post, 2)) : 0.;
    G4double v_pre  = beta_pre  * c_light;  // in mm/ns
    G4double v_post = beta_post * c_light;
   
    G4double dt = step->GetDeltaTime();   
    if (dt > 0) {
        outAcceleration = (v_post - v_pre) / dt;  
    } else {
        outAcceleration = 0;
    }
}

SteppingAction::SteppingAction(EventAction* eventAction)
 : fEventAction(eventAction) { }

SteppingAction::~SteppingAction() = default;

void SteppingAction::UserSteppingAction(const G4Step* step) {

    G4StepPoint* prePoint  = step->GetPreStepPoint();
    G4StepPoint* postPoint = step->GetPostStepPoint();
    G4VPhysicalVolume* preVol  = prePoint->GetPhysicalVolume();
    G4VPhysicalVolume* postVol = postPoint->GetPhysicalVolume();
    if (!preVol) return;  // safety check

    G4String preName  = preVol->GetName();
    G4String postName = postVol ? postVol->GetName() : "";  

    const G4Track* track = step->GetTrack();
    G4int trackID = track->GetTrackID();
    if (trackID != 1) {
        if (step->GetTotalEnergyDeposit() > 0 && preName == "Phantom") {
            fEventAction->AddEnergyDeposit(step->GetTotalEnergyDeposit());
        }
        return;
    }

    
    bool inPhantom = (preName == "Phantom" || postName == "Phantom");
    if (!inPhantom) {
        return;  
    }

    G4ThreeVector position = postPoint->GetPosition();
    G4double kineticE = postPoint->GetKineticEnergy();
    G4double dE = step->GetTotalEnergyDeposit();
    G4String processName = "Transportation";
    if (postPoint->GetProcessDefinedStep() != nullptr) {
        processName = postPoint->GetProcessDefinedStep()->GetProcessName();
    }

    if (dE > 0 && preName == "Phantom") {
        fEventAction->AddEnergyDeposit(dE);
    }

    G4double scatterAngle = 0.0;
    G4double acceleration = 0.0;
    CalculateParticleInteraction(step, scatterAngle, acceleration);

    StepRecord record;
    record.position        = position;
    record.kineticEnergy   = kineticE;
    record.energyDeposited = dE;
    record.scatterAngle    = scatterAngle;
    record.acceleration    = acceleration;
    record.processName     = processName;
    fEventAction->AddStepRecord(record);

    UnityMLInterface::Instance().CollectObservation(step);
}
