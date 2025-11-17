#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "ActionInitialization.hh"
#include <G4RunManagerFactory.hh>
#include <G4UIExecutive.hh>
#include <G4UImanager.hh>
#include <G4VisExecutive.hh>
#include <G4SystemOfUnits.hh>

int main(int argc, char** argv) {
    // Determine if running in batch mode or interactive mode
    bool isBatch = (argc > 1);
    G4UIExecutive* uiExecutive = nullptr;

    if (!isBatch) {
        uiExecutive = new G4UIExecutive(argc, argv);
    }

    // Create run manager
    auto* runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);

    // Set mandatory initialization classes
    runManager->SetUserInitialization(new DetectorConstruction());
    runManager->SetUserInitialization(new PhysicsList());
    runManager->SetUserInitialization(new ActionInitialization());

    // Initialize G4 kernel
    runManager->Initialize();

    // Visualization manager (only for interactive mode)
    G4VisManager* visManager = nullptr;
    if (!isBatch) {
        visManager = new G4VisExecutive();
        visManager->Initialize();
    }

    // Get UI manager
    G4UImanager* UImanager = G4UImanager::GetUIpointer();

    if (isBatch) {
        // BATCH MODE: Execute macro file from command line
        G4String command = "/control/execute ";
        G4String macroFile = argv[1];
        UImanager->ApplyCommand(command + macroFile);

        G4cout << "\n========================================\n";
        G4cout << "Batch mode completed successfully!\n";
        G4cout << "========================================\n";
    }
    else {
        // INTERACTIVE MODE: Start visualization and UI
        UImanager->ApplyCommand("/control/execute ../macros/init_vis.mac");
        uiExecutive->SessionStart();
        delete uiExecutive;
    }

    // Cleanup
    if (visManager) delete visManager;
    delete runManager;

    return 0;
}