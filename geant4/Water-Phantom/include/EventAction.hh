#ifndef EVENT_ACTION_HH
#define EVENT_ACTION_HH

#include <G4UserEventAction.hh>
#include <vector>
#include <G4ThreeVector.hh>

// Twoja bogata struktura danych (zachowana)
struct StepRecord {
    G4ThreeVector position;
    G4ThreeVector momentum;
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
    
    void BeginOfEventAction(const G4Event* event) override;
    void EndOfEventAction(const G4Event* event) override;
    
    void AddStepRecord(const StepRecord& step);
    void AddEnergyDeposit(G4double edep);
    
    // Gettery dla Pythona
    const std::vector<StepRecord>& GetStepRecords() const { return fStepRecords; }
    
    // Setter do sterowania zapisem CSV (Domyślnie false)
    void SetCsvExport(bool enable) { fEnableCsvExport = enable; }
    void ClearRecords() { fStepRecords.clear(); }

private:
    G4double fTotalEnergyDeposit;
    std::vector<StepRecord> fStepRecords;
    
    // Flaga sterująca zapisem
    bool fEnableCsvExport;

    // Metody pomocnicze do CSV (zachowane, ale prywatne)
    G4String GetOutputDirectory() const;
    void ExportResults(G4int eventID, const G4String& outputDir);
};

#endif