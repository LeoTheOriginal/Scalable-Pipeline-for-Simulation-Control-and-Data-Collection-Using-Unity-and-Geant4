#include <G4VUserDetectorConstruction.hh>
#include <globals.hh>

class G4VPhysicalVolume; 

class DetectorConstruction : public G4VUserDetectorConstruction {
public:
    DetectorConstruction();
    ~DetectorConstruction() override;
    G4VPhysicalVolume* Construct() override; 
private:
    G4double fPhantomSize; 
    G4double fWorldSize;
};
