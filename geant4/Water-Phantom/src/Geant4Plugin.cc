#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "ActionInitialization.hh"
#include "EventAction.hh"
#include "G4SystemOfUnits.hh"

// Globalny wskaźnik na managera (w DLL stan musi być gdzieś trzymany)
G4RunManager* g_RunManager = nullptr;

// Funkcje pomocnicze do wyciągania wskaźników
EventAction* GetCurrentEventAction() {
    if (!g_RunManager) return nullptr;
    const auto* actionInit = static_cast<const ActionInitialization*>(g_RunManager->GetUserActionInitialization());
    auto eventAction = static_cast<const EventAction*>(g_RunManager->GetUserEventAction());
    return const_cast<EventAction*>(eventAction);
}

extern "C" {

    // 1. Inicjalizacja Symulacji (Wołane raz na starcie Unity)
    __declspec(dllexport) void InitGeant4() {
        // ZMIANA: Jeśli manager już istnieje (został z poprzedniego Play), NIE twórz nowego.
        if (g_RunManager != nullptr) {
            // Opcjonalnie: Można tu zresetować jakieś parametry runu, jeśli trzeba
            return;
        }

        g_RunManager = new G4RunManager();
        g_RunManager->SetUserInitialization(new DetectorConstruction());
        g_RunManager->SetUserInitialization(new PhysicsList());
        g_RunManager->SetUserInitialization(new ActionInitialization());
        g_RunManager->Initialize();

        // Wyciszamy logi
        G4UImanager* UI = G4UImanager::GetUIpointer();
        UI->ApplyCommand("/process/em/verbose 0");
        UI->ApplyCommand("/run/verbose 0");
        UI->ApplyCommand("/event/verbose 0");
        UI->ApplyCommand("/tracking/verbose 0");
    }

    // 2. Sprzątanie (Wołane przy zamykaniu Unity)
    __declspec(dllexport) void CloseGeant4() {
//        if (g_RunManager) {
//            delete g_RunManager;
//            g_RunManager = nullptr;
//        }
    }

    // 3. Główna funkcja symulacji
    // outData: Wskaźnik do tablicy floatów w C# (musi być zaalokowana w C#)
    // maxSteps: Rozmiar bufora (liczba kroków), żeby nie wyjść poza pamięć
    // Zwraca: Liczbę faktycznie zapisanych kroków
    __declspec(dllexport) int RunSimulationBatch(float* outData, int maxSteps) {
        if (!g_RunManager) return 0;

        // Uruchom 1 event
        g_RunManager->BeamOn(1);

        auto eventAction = GetCurrentEventAction();
        if (!eventAction) return 0;

        const auto& records = eventAction->GetStepRecords();
        int stepsCount = records.size();

        // Zabezpieczenie przed przepełnieniem bufora C#
        if (stepsCount > maxSteps) stepsCount = maxSteps;

        // Kopiowanie danych do tablicy C#
        // Format: [x, y, z, px, py, pz, e] (7 floatów na krok)
        int stride = 7;

        for (int i = 0; i < stepsCount; ++i) {
            const auto& step = records[i];
            int base = i * stride;

            outData[base + 0] = (float)(step.position.x() / cm);
            outData[base + 1] = (float)(step.position.y() / cm);
            outData[base + 2] = (float)(step.position.z() / cm);

            outData[base + 3] = (float)(step.momentum.x() / MeV);
            outData[base + 4] = (float)(step.momentum.y() / MeV);
            outData[base + 5] = (float)(step.momentum.z() / MeV);

            outData[base + 6] = (float)(step.kineticEnergy / MeV);
        }

        return stepsCount;
    }
}