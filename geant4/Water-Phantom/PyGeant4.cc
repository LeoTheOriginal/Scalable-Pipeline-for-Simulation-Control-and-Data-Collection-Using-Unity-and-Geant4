#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include <pybind11/numpy.h>

#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "ActionInitialization.hh"
#include "EventAction.hh"
#include "RunAction.hh"
// WAŻNE: Dołączamy jednostki, żeby móc dzielić przez cm i MeV
#include "G4SystemOfUnits.hh"

namespace py = pybind11;

class SimulationManager {
public:
    SimulationManager() {
        // 1. Setup Geant4
        runManager = new G4RunManager();
        runManager->SetUserInitialization(new DetectorConstruction());
        runManager->SetUserInitialization(new PhysicsList());

        auto actionInit = new ActionInitialization();
        runManager->SetUserInitialization(actionInit);
        runManager->Initialize();

        // 2. Silence Geant4 console output
        G4UImanager* UI = G4UImanager::GetUIpointer();
        UI->ApplyCommand("/process/em/verbose 0");
        UI->ApplyCommand("/run/verbose 0");
        UI->ApplyCommand("/event/verbose 0");
        UI->ApplyCommand("/tracking/verbose 0");
    }

    ~SimulationManager() {
        delete runManager;
    }

    void SetCsvLogging(bool enable) {
        auto eventAction = GetCurrentEventAction();
        if (eventAction) {
            eventAction->SetCsvExport(enable);
        }
    }

    py::dict RunSingleSimulation() {
        runManager->BeamOn(1);

        auto eventAction = GetCurrentEventAction();
        const auto& records = eventAction->GetStepRecords();

        size_t n_steps = records.size();

        std::vector<double> x, y, z;
        std::vector<double> px, py, pz;
        std::vector<double> energy;

        x.reserve(n_steps); y.reserve(n_steps); z.reserve(n_steps);
        px.reserve(n_steps); py.reserve(n_steps); pz.reserve(n_steps);
        energy.reserve(n_steps);

        for(const auto& step : records) {
            // ================================================================
            // TU JEST ZMIANA: Konwersja jednostek
            // Dzielimy przez cm, żeby dostać centymetry
            // Dzielimy przez MeV, żeby dostać Megaelektronowolty
            // ================================================================
            x.push_back(step.position.x() / cm);
            y.push_back(step.position.y() / cm);
            z.push_back(step.position.z() / cm);

            px.push_back(step.momentum.x() / MeV);
            py.push_back(step.momentum.y() / MeV);
            pz.push_back(step.momentum.z() / MeV);

            energy.push_back(step.kineticEnergy / MeV);
        }

        py::dict result;
        result["x"] = py::array(n_steps, x.data());
        result["y"] = py::array(n_steps, y.data());
        result["z"] = py::array(n_steps, z.data());
        result["px"] = py::array(n_steps, px.data());
        result["py"] = py::array(n_steps, py.data());
        result["pz"] = py::array(n_steps, pz.data());
        result["energy"] = py::array(n_steps, energy.data());

        return result;
    }

private:
    G4RunManager* runManager;

    EventAction* GetCurrentEventAction() {
        const auto* actionInit = static_cast<const ActionInitialization*>(runManager->GetUserActionInitialization());
        auto eventAction = static_cast<const EventAction*>(runManager->GetUserEventAction());
        return const_cast<EventAction*>(eventAction);
    }
};

PYBIND11_MODULE(geant4_sim, m) {
    py::class_<SimulationManager>(m, "SimulationManager")
        .def(py::init<>())
        .def("run_single", &SimulationManager::RunSingleSimulation)
        .def("set_csv_logging", &SimulationManager::SetCsvLogging);
}