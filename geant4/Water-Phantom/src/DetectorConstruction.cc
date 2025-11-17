#include "DetectorConstruction.hh"
#include <G4Box.hh>
#include <G4NistManager.hh>
#include <G4LogicalVolume.hh>
#include <G4PVPlacement.hh>
#include <G4VisAttributes.hh>
#include <G4Colour.hh>
#include <G4SystemOfUnits.hh>

DetectorConstruction::DetectorConstruction()
 : fPhantomSize(5.0*cm) 
 , fWorldSize(15.0*cm)  
{ }

DetectorConstruction::~DetectorConstruction() = default;

G4VPhysicalVolume* DetectorConstruction::Construct() {
    G4NistManager* nist = G4NistManager::Instance();
    G4Material* air   = nist->FindOrBuildMaterial("G4_AIR");
    G4Material* water = nist->FindOrBuildMaterial("G4_WATER");

    G4Box* solidWorld = new G4Box("World", fWorldSize, fWorldSize, fWorldSize);
    G4LogicalVolume* logicWorld = new G4LogicalVolume(solidWorld, air, "World");
    
    G4VPhysicalVolume* physWorld = new G4PVPlacement(
        nullptr,                
        G4ThreeVector(),       
        logicWorld,             
        "World",               
        nullptr,               
        false,                  
        0,                     
        true                    
    );

    
    G4Box* solidPhantom = new G4Box("Phantom", fPhantomSize, fPhantomSize, fPhantomSize);
    G4LogicalVolume* logicPhantom = new G4LogicalVolume(solidPhantom, water, "Phantom");
    new G4PVPlacement(
        nullptr,
        G4ThreeVector(0., 0., 0.),  
        logicPhantom,
        "Phantom",
        logicWorld,   
        false,
        0,
        true          
    );

    logicWorld->SetVisAttributes(G4VisAttributes::GetInvisible());            
    G4VisAttributes* phantomVis = new G4VisAttributes(G4Colour::Blue());
    phantomVis->SetForceWireframe(true);   
    phantomVis->SetLineWidth(2.0);
    logicPhantom->SetVisAttributes(phantomVis);
   

    return physWorld;
}
