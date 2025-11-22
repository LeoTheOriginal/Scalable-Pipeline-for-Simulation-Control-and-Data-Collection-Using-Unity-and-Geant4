#include "DetectorConstruction.hh"
#include <G4Box.hh>
#include <G4NistManager.hh>
#include <G4LogicalVolume.hh>
#include <G4PVPlacement.hh>
#include <G4VisAttributes.hh>
#include <G4Colour.hh>
#include <G4SystemOfUnits.hh>

DetectorConstruction::DetectorConstruction()
 : fPhantomSize(5.0*cm)    // 10cm phantom 
 , fWorldSize(15.0*cm)      // 30cm world
 , fLogicWorld(nullptr)
 , fLogicPhantom(nullptr)
{
    G4cout << "[DetectorConstruction] Initialized:" << G4endl;
    G4cout << "  Phantom: " << fPhantomSize*2/cm << " × " 
           << fPhantomSize*2/cm << " × " << fPhantomSize*2/cm << " cm³" << G4endl;
    G4cout << "  World:   " << fWorldSize*2/cm << " × " 
           << fWorldSize*2/cm << " × " << fWorldSize*2/cm << " cm³" << G4endl;
}

DetectorConstruction::~DetectorConstruction() = default;

G4VPhysicalVolume* DetectorConstruction::Construct() {
    // ========================================================================
    // Materials
    // ========================================================================
    G4NistManager* nist = G4NistManager::Instance();
    G4Material* air   = nist->FindOrBuildMaterial("G4_AIR");
    G4Material* water = nist->FindOrBuildMaterial("G4_WATER");
    
    G4cout << "[DetectorConstruction] Materials loaded:" << G4endl;
    G4cout << "  Air:   " << air->GetName() 
           << " (ρ = " << air->GetDensity()/(g/cm3) << " g/cm³)" << G4endl;
    G4cout << "  Water: " << water->GetName() 
           << " (ρ = " << water->GetDensity()/(g/cm3) << " g/cm³)" << G4endl;

    // ========================================================================
    // World volume (30×30×30 cm³ air box)
    // ========================================================================
    G4Box* solidWorld = new G4Box(
        "World",         // name
        fWorldSize,      // half-size X
        fWorldSize,      // half-size Y
        fWorldSize       // half-size Z
    );
    
    fLogicWorld = new G4LogicalVolume(
        solidWorld,      // solid
        air,             // material
        "World"          // name
    );
    
    G4VPhysicalVolume* physWorld = new G4PVPlacement(
        nullptr,                // no rotation
        G4ThreeVector(),        // at origin (0, 0, 0)
        fLogicWorld,            // logical volume
        "World",                // name
        nullptr,                // no mother volume
        false,                  // no boolean operations
        0,                      // copy number
        true                    // check overlaps
    );

    // ========================================================================
    // Water phantom (10×10×10 cm³ water box at center)
    // ========================================================================
    G4Box* solidPhantom = new G4Box(
        "Phantom",       // name
        fPhantomSize,    // half-size X
        fPhantomSize,    // half-size Y
        fPhantomSize     // half-size Z
    );
    
    fLogicPhantom = new G4LogicalVolume(
        solidPhantom,    // solid
        water,           // material
        "Phantom"        // name
    );
    
    new G4PVPlacement(
        nullptr,                // no rotation
        G4ThreeVector(0., 0., 0.),  // at world center
        fLogicPhantom,          // logical volume
        "Phantom",              // name
        fLogicWorld,            // mother volume
        false,                  // no boolean operations
        0,                      // copy number
        true                    // check overlaps
    );

    // ========================================================================
    // Visualization attributes
    // ========================================================================
    
    // World: invisible
    fLogicWorld->SetVisAttributes(G4VisAttributes::GetInvisible());
    
    // Phantom: blue wireframe
    G4VisAttributes* phantomVis = new G4VisAttributes(G4Colour::Blue());
    phantomVis->SetForceWireframe(true);
    phantomVis->SetLineWidth(2.0);
    fLogicPhantom->SetVisAttributes(phantomVis);
    
    G4cout << "[DetectorConstruction] Geometry constructed successfully" << G4endl;
    G4cout << "  Phantom center: (0, 0, 0) cm" << G4endl;
    G4cout << "  Typical particle start: (-6, 0, 0) cm" << G4endl;
    G4cout << "  Expected trajectory: along +X axis into phantom" << G4endl;

    return physWorld;
}