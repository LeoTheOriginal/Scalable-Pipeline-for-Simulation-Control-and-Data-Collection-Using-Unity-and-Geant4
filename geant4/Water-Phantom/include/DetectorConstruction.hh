#ifndef DETECTOR_CONSTRUCTION_HH
#define DETECTOR_CONSTRUCTION_HH

#include <G4VUserDetectorConstruction.hh>
#include <globals.hh>

class G4VPhysicalVolume;
class G4LogicalVolume;

class DetectorConstruction : public G4VUserDetectorConstruction {
public:
    DetectorConstruction();
    ~DetectorConstruction() override;
    
    G4VPhysicalVolume* Construct() override;
    
    G4double GetPhantomSize() const { return fPhantomSize; }
    G4double GetWorldSize() const { return fWorldSize; }
    
private:
    G4double fPhantomSize;
    G4double fWorldSize;
    
    // Te dwie linie MUSZĄ być!
    G4LogicalVolume* fLogicWorld;
    G4LogicalVolume* fLogicPhantom;
};

#endif