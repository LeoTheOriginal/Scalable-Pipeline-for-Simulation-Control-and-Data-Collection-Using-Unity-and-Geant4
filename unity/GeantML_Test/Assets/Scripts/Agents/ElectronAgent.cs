using Core;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Agents
{
    public class ElectronAgent : Agent
    {
        [Header("Simulation Settings")]
        public int MaxSteps = 500;
        public bool ShowVisualization = true;

        // Bufor na dane z C++: [x, y, z, px, py, pz, e]
        // 7 floats per step
        private float[] _trajectoryBuffer;
        private int _trajectoryLength = 0;
        private int _currentStep = 0;

        // Stałe fizyczne
        private const float MASS_E = 0.511f; // MeV

        // Wagi nagród (Te same co ustaliliśmy w Pythonie)
        private const float W_POS = 20.0f;
        private const float W_MOM = 5.0f;
        private const float W_PHYSICS = 10.0f;
        private const float W_DIR = 2.0f;

        public override void Initialize()
        {
            // Alokujemy pamięć raz
            _trajectoryBuffer = new float[MaxSteps * 7];
        }

        public override void OnEpisodeBegin()
        {
            // 1. Pobierz Ground Truth z C++
            // To działa jak 'reset()' w Pythonie
            // Pętla retry, żeby nie dostać pustej cząstki
            int attempts = 0;
            do
            {
                _trajectoryLength = Geant4Interface.RunSimulationBatch(_trajectoryBuffer, MaxSteps);
                attempts++;
            } while (_trajectoryLength < 2 && attempts < 10);

            if (_trajectoryLength < 2)
            {
                Debug.LogWarning("Geant4 zwrócił puste dane po 10 próbach.");
                EndEpisode();
                return;
            }

            _currentStep = 0;

            // Ustawiamy agenta fizycznie na pozycji startowej (dla wizualizacji)
            // Dane: [0=x, 1=y, 2=z]
            transform.localPosition = new Vector3(_trajectoryBuffer[0], _trajectoryBuffer[1], _trajectoryBuffer[2]);
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            // Agent widzi swój AKTUALNY stan (z bufora C++)
            // Teacher Forcing: Agent dostaje "Gdzie jest teraz naprawdę"
            int baseIdx = _currentStep * 7;

            // 7 obserwacji
            for (int i = 0; i < 7; i++)
            {
                sensor.AddObservation(_trajectoryBuffer[baseIdx + i]);
            }
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            if (_currentStep >= _trajectoryLength - 1)
            {
                EndEpisode();
                return;
            }

            // --- 1. Pobranie Ground Truth (Cel) ---
            int currIdx = _currentStep * 7;
            int nextIdx = (_currentStep + 1) * 7;

            // Pozycja obecna i następna (Prawdziwa)
            Vector3 posGT_Curr = new Vector3(_trajectoryBuffer[currIdx], _trajectoryBuffer[currIdx + 1], _trajectoryBuffer[currIdx + 2]);
            Vector3 posGT_Next = new Vector3(_trajectoryBuffer[nextIdx], _trajectoryBuffer[nextIdx + 1], _trajectoryBuffer[nextIdx + 2]);

            // Delta Prawdziwa (Target)
            Vector3 deltaTrue_Pos = posGT_Next - posGT_Curr;

            // --- 2. Akcja Agenta (Predykcja) ---
            var act = actions.ContinuousActions;
            // Agent zwraca 7 wartości: [dx, dy, dz, dpx, dpy, dpz, de]
            Vector3 deltaPred_Pos = new Vector3(act[0], act[1], act[2]);

            // --- 3. Obliczanie Nagrody ---
            float reward = 0.0f;

            // A. Błąd Pozycji (MSE)
            float distError = Vector3.Distance(deltaPred_Pos, deltaTrue_Pos);
            reward -= (distError * W_POS);

            // B. Kierunek (Wymuszanie osi X - idź w głąb)
            if (deltaPred_Pos.x > 0) reward += W_DIR;
            else reward -= W_DIR;

            // C. Fizyka (E^2 = p^2 + m^2)
            // Predykcja pędu i energii przez Agenta
            float pred_px = act[3];
            float pred_py = act[4];
            float pred_pz = act[5];
            float pred_e = act[6];

            float p_sq = pred_px * pred_px + pred_py * pred_py + pred_pz * pred_pz;
            float e_physical = Mathf.Sqrt(p_sq + MASS_E * MASS_E) - MASS_E;
            float phys_error = Mathf.Abs(pred_e - e_physical);

            reward -= (phys_error * W_PHYSICS);

            // D. Bonus za precyzję
            if (distError < 0.05f) reward += 5.0f;

            SetReward(reward);

            // --- 4. Wizualizacja ---
            if (ShowVisualization)
            {
                // Przesuwamy kulkę tam, gdzie Agent CHCIAŁ iść (Predykcja)
                // Uwaga: W następnym kroku CollectObservations cofnie go na "dobrą drogę" (Teacher Forcing)
                // Więc kulka będzie "drżeć" wokół prawdziwej ścieżki.
                transform.localPosition = posGT_Curr + deltaPred_Pos;
            }

            _currentStep++;
        }

        // Heurystyka do ręcznego sterowania (opcjonalne, do testów w edytorze)
        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var continuousActionsOut = actionsOut.ContinuousActions;
            continuousActionsOut[0] = 0.1f; // Idź do przodu
            // Reszta 0
        }
    }
}