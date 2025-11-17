#include "EventAction.hh"
#include <G4Event.hh>
#include <G4SystemOfUnits.hh>
#include <G4AnalysisManager.hh> 
#include <G4ios.hh>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>

EventAction::EventAction() : fTotalEnergyDeposit(0.) {}
EventAction::~EventAction() = default;

void EventAction::BeginOfEventAction(const G4Event* /*event*/) {
    fTotalEnergyDeposit = 0.0;
    fStepRecords.clear();
}

void EventAction::EndOfEventAction(const G4Event* event) {
    G4int eventID = event->GetEventID();
    G4cout << ">>> Event " << eventID
        << " finished. Total energy deposited in phantom = "
        << fTotalEnergyDeposit / MeV << " MeV." << G4endl;

    // Export results to CSV file
    const char* outputDirEnv = std::getenv("G4_OUTPUT_DIR");
    G4String outputDir = outputDirEnv ? G4String(outputDirEnv) : G4String(".");
    ExportResults(eventID, outputDir);
}

void EventAction::AddStepRecord(const StepRecord& step) {
    fStepRecords.push_back(step);

    const G4ThreeVector& pos = step.position;
    G4cout
        << "  Step at (" << pos.x() / cm << ", " << pos.y() / cm << ", " << pos.z() / cm << ") [cm]"
        << ": KE=" << step.kineticEnergy / MeV << " MeV"
        << ", dE=" << step.energyDeposited / MeV << " MeV"
        << ", scatterθ=" << step.scatterAngle / deg << " deg"
        << ", accel=" << step.acceleration << " mm/ns^2"
        << ", process=\"" << step.processName << "\""
        << G4endl;
}

void EventAction::AddEnergyDeposit(G4double edep) {
    fTotalEnergyDeposit += edep;
}

void EventAction::ExportResults(G4int eventID, const G4String& outputDir) {
    // Create output filename
    std::stringstream filename;
    filename << outputDir << "/event_"
        << std::setw(6) << std::setfill('0') << eventID << ".csv";

    std::ofstream outFile(filename.str());
    if (!outFile.is_open()) {
        G4cerr << "ERROR: Cannot open output file: " << filename.str() << G4endl;
        return;
    }

    // Write summary header
    outFile << "# Event Summary\n";
    outFile << "EventID," << eventID << "\n";
    outFile << "TotalEnergyDeposit," << fTotalEnergyDeposit / MeV << "\n";
    outFile << "NumberOfSteps," << fStepRecords.size() << "\n";
    outFile << "\n";

    // Write step data header
    outFile << "# Step Data\n";
    outFile << "StepID,PosX_cm,PosY_cm,PosZ_cm,KineticEnergy_MeV,EnergyDeposited_MeV,";
    outFile << "ScatterAngle_deg,Acceleration,ProcessName\n";

    // Write step records
    for (size_t i = 0; i < fStepRecords.size(); ++i) {
        const StepRecord& step = fStepRecords[i];
        outFile << i << ","
            << step.position.x() / cm << ","
            << step.position.y() / cm << ","
            << step.position.z() / cm << ","
            << step.kineticEnergy / MeV << ","
            << step.energyDeposited / MeV << ","
            << step.scatterAngle / deg << ","
            << step.acceleration << ","
            << step.processName << "\n";
    }

    outFile.close();
    G4cout << "✅ Results exported to: " << filename.str() << G4endl;
}