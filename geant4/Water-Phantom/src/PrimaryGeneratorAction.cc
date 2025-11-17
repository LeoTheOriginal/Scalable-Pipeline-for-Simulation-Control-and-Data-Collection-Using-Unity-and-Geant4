#include "PrimaryGeneratorAction.hh"

#include <G4Event.hh>
#include <G4ParticleGun.hh>
#include <G4ParticleDefinition.hh>
#include <G4ParticleTable.hh>
#include <G4SystemOfUnits.hh>
#include <G4ThreeVector.hh>

PrimaryGeneratorAction::PrimaryGeneratorAction() {
  
  fParticleGun = new G4ParticleGun(1);

  
  auto* electron = G4ParticleTable::GetParticleTable()->FindParticle("e-");
  fParticleGun->SetParticleDefinition(electron);

  fParticleGun->SetParticleEnergy(10.0 * MeV);
  fParticleGun->SetParticleMomentumDirection(G4ThreeVector(1., 0., 0.));
  fParticleGun->SetParticlePosition(G4ThreeVector(-6.0 * cm, 0.0 * cm, 0.0 * cm));
}

PrimaryGeneratorAction::~PrimaryGeneratorAction() {
  delete fParticleGun;
  fParticleGun = nullptr;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event) {
  fParticleGun->GeneratePrimaryVertex(event);
}
