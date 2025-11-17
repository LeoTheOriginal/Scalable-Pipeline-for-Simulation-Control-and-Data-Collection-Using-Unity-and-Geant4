#include "UnityMLInterface.hh"
#include <G4Step.hh>
#include <G4ThreeVector.hh>
#include <G4ios.hh>

void UnityMLInterface::CollectObservation(const G4Step* step) {
    G4StepPoint* pre = step->GetPreStepPoint();
    G4StepPoint* post = step->GetPostStepPoint();
    if (!pre || !post) return;
}
