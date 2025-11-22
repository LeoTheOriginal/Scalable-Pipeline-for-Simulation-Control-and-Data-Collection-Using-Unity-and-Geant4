#include "PhysicsList.hh"
#include <G4EmLivermorePolarizedPhysics.hh>
#include <G4SystemOfUnits.hh>

PhysicsList::PhysicsList() {
    // sensowny, dość standardowy cut
    defaultCutValue = 1.0 * mm;
    RegisterPhysics(new G4EmLivermorePolarizedPhysics());
}

PhysicsList::~PhysicsList() = default;

void PhysicsList::SetCuts() {
    // użyj domyślnych cutów opartych na defaultCutValue
    SetCutsWithDefault();

    // jeśli kiedyś będziesz chciał cuts per region:
    // - stwórz G4Region w DetectorConstruction
    // - znajdź Region tutaj przez G4RegionStore
    // - przypisz do niego G4ProductionCuts
}
