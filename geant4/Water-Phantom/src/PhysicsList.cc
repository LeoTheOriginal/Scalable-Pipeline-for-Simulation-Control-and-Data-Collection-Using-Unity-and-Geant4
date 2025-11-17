#include "PhysicsList.hh"
#include <G4EmLivermorePolarizedPhysics.hh>
#include <G4SystemOfUnits.hh>

PhysicsList::PhysicsList() {
    
    defaultCutValue = 1.0 * mm;
    RegisterPhysics(new G4EmLivermorePolarizedPhysics());
}

PhysicsList::~PhysicsList() = default;

void PhysicsList::SetCuts() {
    G4VUserPhysicsList::SetCuts();
}
