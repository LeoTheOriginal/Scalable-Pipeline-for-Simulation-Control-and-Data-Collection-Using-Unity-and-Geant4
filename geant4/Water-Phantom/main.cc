#define _CRT_SECURE_NO_WARNINGS

#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "ActionInitialization.hh"
#include <G4RunManagerFactory.hh>
#include <G4UIExecutive.hh>
#include <G4UImanager.hh>
#include <G4VisExecutive.hh>
#include <G4SystemOfUnits.hh>
#include <cstdlib>
#include <iostream>

void PrintUsage() {
    G4cout << "\n==========================================================\n";
    G4cout << "WaterPhantomSim - Geant4 simulation for ML-Agents\n";
    G4cout << "==========================================================\n";
    G4cout << "\nUsage:\n";
    G4cout << "  Interactive mode:  WaterPhantomSim\n";
    G4cout << "  Batch mode:        WaterPhantomSim <macro_file>\n";
    G4cout << "\nEnvironment variables:\n";
    G4cout << "  G4_OUTPUT_DIR      Output directory for CSV files\n";
    G4cout << "                     (default: current directory)\n";
    G4cout << "\nExample:\n";
    G4cout << "  set G4_OUTPUT_DIR=C:\\output\n";
    G4cout << "  WaterPhantomSim run.mac\n";
    G4cout << "\n==========================================================\n\n";
}

void PrintConfiguration() {
    G4cout << "\n==========================================================\n";
    G4cout << "CONFIGURATION\n";
    G4cout << "==========================================================\n";
    
    // Check output directory
    const char* outputDir = std::getenv("G4_OUTPUT_DIR");
    if (outputDir) {
        G4cout << "Output directory: " << outputDir << G4endl;
    } else {
        G4cout << "Output directory: . (current directory)" << G4endl;
        G4cout << "TIP: Set G4_OUTPUT_DIR environment variable to change" << G4endl;
    }
    
    G4cout << "Phantom size:     10 × 10 × 10 cm³" << G4endl;
    G4cout << "World size:       30 × 30 × 30 cm³" << G4endl;
    G4cout << "Default particle: e- (electron)" << G4endl;
    G4cout << "Default energy:   10 MeV" << G4endl;
    G4cout << "Default position: (-6, 0, 0) cm" << G4endl;
    G4cout << "Default direction: (1, 0, 0)" << G4endl;
    G4cout << "==========================================================\n\n";
}

int main(int argc, char** argv) {
    // ========================================================================
    // Parse command line
    // ========================================================================
    bool isBatch = (argc > 1);
    G4UIExecutive* uiExecutive = nullptr;
    
    if (!isBatch) {
        uiExecutive = new G4UIExecutive(argc, argv);
    }
    
    // Show usage if --help
    if (argc > 1 && (G4String(argv[1]) == "--help" || G4String(argv[1]) == "-h")) {
        PrintUsage();
        return 0;
    }
    
    G4cout << "\n";
    G4cout << "████████████████████████████████████████████████████████\n";
    G4cout << "██                                                    ██\n";
    G4cout << "██    WaterPhantomSim - Geant4 11.3.2                ██\n";
    G4cout << "██    ML-Agents Integration                          ██\n";
    G4cout << "██                                                    ██\n";
    G4cout << "████████████████████████████████████████████████████████\n";
    G4cout << "\n";
    
    PrintConfiguration();
    
    // ========================================================================
    // Create run manager
    // ========================================================================
    G4cout << "[Main] Creating run manager...\n";
    auto* runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);
    G4cout << "[Main] ✅ Run manager created\n";
    
    // ========================================================================
    // Set mandatory initialization classes
    // ========================================================================
    G4cout << "[Main] Initializing detector construction...\n";
    runManager->SetUserInitialization(new DetectorConstruction());
    G4cout << "[Main] ✅ Detector construction set\n";
    
    G4cout << "[Main] Initializing physics list...\n";
    runManager->SetUserInitialization(new PhysicsList());
    G4cout << "[Main] ✅ Physics list set\n";
    
    G4cout << "[Main] Initializing user actions...\n";
    runManager->SetUserInitialization(new ActionInitialization());
    G4cout << "[Main] ✅ User actions set\n";
    
    // ========================================================================
    // Initialize G4 kernel
    // ========================================================================
    G4cout << "[Main] Initializing Geant4 kernel...\n";
    runManager->Initialize();
    G4cout << "[Main] ✅ Geant4 kernel initialized\n\n";
    
    // ========================================================================
    // Visualization (interactive mode only)
    // ========================================================================
    G4VisManager* visManager = nullptr;
    if (!isBatch) {
        G4cout << "[Main] Initializing visualization...\n";
        visManager = new G4VisExecutive();
        visManager->Initialize();
        G4cout << "[Main] ✅ Visualization ready\n";
    }
    
    // ========================================================================
    // Get UI manager
    // ========================================================================
    G4UImanager* UImanager = G4UImanager::GetUIpointer();
    
    if (isBatch) {
        // ====================================================================
        // BATCH MODE: Execute macro file
        // ====================================================================
        G4String macroFile = argv[1];
        
        G4cout << "\n==========================================================\n";
        G4cout << "BATCH MODE\n";
        G4cout << "==========================================================\n";
        G4cout << "Macro file: " << macroFile << G4endl;
        G4cout << "==========================================================\n\n";
        
        // Check if file exists
        std::ifstream testFile(macroFile);
        if (!testFile.good()) {
            G4cerr << "ERROR: Macro file not found: " << macroFile << G4endl;
            delete runManager;
            return 1;
        }
        testFile.close();
        
        G4String command = "/control/execute ";
        UImanager->ApplyCommand(command + macroFile);
        
        G4cout << "\n==========================================================\n";
        G4cout << "BATCH MODE COMPLETED\n";
        G4cout << "==========================================================\n";
        G4cout << "Check output directory for CSV files:\n";
        
        const char* outputDir = std::getenv("G4_OUTPUT_DIR");
        if (outputDir) {
            G4cout << "  " << outputDir << "/event_*.csv\n";
        } else {
            G4cout << "  ./event_*.csv\n";
        }
        
        G4cout << "==========================================================\n\n";
    }
    else {
        // ====================================================================
        // INTERACTIVE MODE: Start visualization and UI
        // ====================================================================
        G4cout << "\n==========================================================\n";
        G4cout << "INTERACTIVE MODE\n";
        G4cout << "==========================================================\n";
        G4cout << "Starting Qt GUI...\n";
        G4cout << "==========================================================\n\n";
        
        // Execute initialization macro if it exists
        std::ifstream initMacro("../macros/init_vis.mac");
        if (initMacro.good()) {
            initMacro.close();
            UImanager->ApplyCommand("/control/execute ../macros/init_vis.mac");
        } else {
            G4cout << "Note: init_vis.mac not found, skipping visualization setup\n";
        }
        
        uiExecutive->SessionStart();
        delete uiExecutive;
    }
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    G4cout << "\n[Main] Cleaning up...\n";
    
    if (visManager) {
        delete visManager;
        G4cout << "[Main] ✅ Visualization manager deleted\n";
    }
    
    delete runManager;
    G4cout << "[Main] ✅ Run manager deleted\n";
    
    G4cout << "\n==========================================================\n";
    G4cout << "Simulation completed successfully!\n";
    G4cout << "==========================================================\n\n";
    
    return 0;
}