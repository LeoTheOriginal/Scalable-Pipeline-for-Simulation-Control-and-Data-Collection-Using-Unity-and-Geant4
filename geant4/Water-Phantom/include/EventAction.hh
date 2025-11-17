#include <G4UserEventAction.hh>
#include <vector>
#include <G4ThreeVector.hh>

struct StepRecord {
    G4ThreeVector position;
    G4double kineticEnergy;
    G4double energyDeposited;
    G4double scatterAngle;
    G4double acceleration;
    G4String processName;
};

class EventAction : public G4UserEventAction {
public:
    EventAction();
    ~EventAction() override;
    void BeginOfEventAction(const G4Event*) override;
    void EndOfEventAction(const G4Event*) override;
    void AddStepRecord(const StepRecord& step);
    void AddEnergyDeposit(G4double edep);

    // NEW: Export methods for Python interface
    void ExportResults(G4int eventID, const G4String& outputDir);
    G4double GetTotalEnergyDeposit() const { return fTotalEnergyDeposit; }
    const std::vector<StepRecord>& GetStepRecords() const { return fStepRecords; }

private:
    G4double fTotalEnergyDeposit;
    std::vector<StepRecord> fStepRecords;
};