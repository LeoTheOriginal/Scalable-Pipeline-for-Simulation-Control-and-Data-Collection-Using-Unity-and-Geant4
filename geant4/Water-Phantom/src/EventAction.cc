#include "EventAction.hh"
#include <G4Event.hh>
#include <G4SystemOfUnits.hh>
#include <G4AnalysisManager.hh>
#include <G4ios.hh>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <sys/stat.h>
#include <vector>

// Windows/Linux cross-platform mkdir
#ifdef _WIN32
#include <direct.h>
#endif

EventAction::EventAction() 
 : fTotalEnergyDeposit(0.)
 , fEnableCsvExport(false) // WAŻNE: Domyślnie WYŁĄCZONE dla wydajności >1000 FPS
{
    // Rezerwacja pamięci, żeby uniknąć realokacji przy każdym kroku
    fStepRecords.reserve(1000);
    G4cout << "[EventAction] Initialized (CSV Export: Disabled by default)" << G4endl;
}

EventAction::~EventAction() = default;

void EventAction::BeginOfEventAction(const G4Event* /*event*/) {
    fTotalEnergyDeposit = 0.0;
    fStepRecords.clear();
}

void EventAction::EndOfEventAction(const G4Event* event) {
    // 1. Logika dla Pythona (Zawsze aktywna - dane w RAM)
    // (Dane już są w fStepRecords dzięki AddStepRecord)

    // 2. Logika dla CSV (Tylko jeśli włączona flagą)
    if (fEnableCsvExport) {
        G4int eventID = event->GetEventID();
        
        // Logowanie do konsoli (tylko przy CSV, żeby nie śmiecić przy szybkim treningu)
        if (eventID % 100 == 0 || fStepRecords.size() > 0) {
            G4cout << "[EventAction] Event " << eventID << " finished:" << G4endl;
            G4cout << "  Steps:  " << fStepRecords.size() << G4endl;
            G4cout << "  Energy: " << fTotalEnergyDeposit / MeV << " MeV deposited" << G4endl;
        }
        
        G4String outputDir = GetOutputDirectory();
        ExportResults(eventID, outputDir);
    }
}

void EventAction::AddStepRecord(const StepRecord& step) {
    fStepRecords.push_back(step);
    
    // Debug output (Tylko jeśli CSV włączone, bo spowalnia)
    if (fEnableCsvExport && fStepRecords.size() % 50 == 0) {
        // (Opcjonalne logowanie kroków)
    }
}

void EventAction::AddEnergyDeposit(G4double edep) {
    fTotalEnergyDeposit += edep;
}

// ============================================================================
// Legacy CSV Code (Zachowany, ale ukryty za flagą)
// ============================================================================

G4String EventAction::GetOutputDirectory() const {
    const char* outputDirEnv = std::getenv("G4_OUTPUT_DIR");
    if (outputDirEnv) {
        return G4String(outputDirEnv);
    }
    return G4String(".");
}

void EventAction::ExportResults(G4int eventID, const G4String& outputDir) {
    struct stat info;
    if (stat(outputDir.c_str(), &info) != 0) {
    #ifdef _WIN32
        _mkdir(outputDir.c_str());
    #else
        mkdir(outputDir.c_str(), 0755);
    #endif
    }

    std::stringstream filename;
    filename << outputDir << "/event_"
             << std::setw(6) << std::setfill('0') << eventID << ".csv";

    std::ofstream outFile(filename.str());
    if (!outFile.is_open()) {
        G4cerr << "[EventAction] ERROR: Cannot open output file" << G4endl;
        return;
    }
    
    outFile << "# Event Summary\n";
    outFile << "EventID," << eventID << "\n";
    outFile << "TotalEnergyDeposit," << fTotalEnergyDeposit / MeV << "\n";
    outFile << "NumberOfSteps," << fStepRecords.size() << "\n\n";
    
    outFile << "StepID,PosX_cm,PosY_cm,PosZ_cm,"
            << "PmomX_MeVc,PmomY_MeVc,PmomZ_MeVc,"
            << "KineticEnergy_MeV,EnergyDeposited_MeV,"
            << "ScatterAngle_deg,Acceleration,"
            << "ProcessName\n";

    for (size_t i = 0; i < fStepRecords.size(); ++i) {
        const StepRecord& step = fStepRecords[i];
        outFile << i << ","
                << step.position.x() / cm << ","
                << step.position.y() / cm << ","
                << step.position.z() / cm << ","
                << step.momentum.x() / MeV << ","
                << step.momentum.y() / MeV << ","
                << step.momentum.z() / MeV << ","
                << step.kineticEnergy / MeV << ","
                << step.energyDeposited / MeV << ","
                << step.scatterAngle / deg << ","
                << step.acceleration << ","
                << step.processName << "\n";
    }
    outFile.close();
}