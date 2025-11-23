using UnityEngine;

namespace Core
{
    public class Geant4Manager : MonoBehaviour
    {
        public static Geant4Manager Instance { get; private set; }

        void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(gameObject);
                return;
            }
            Instance = this;
            // Nie używamy DontDestroyOnLoad w Edytorze, bo to czasem utrudnia debugowanie
            // Ale w buildzie jest ok. Zostawmy.
            DontDestroyOnLoad(gameObject);

            Debug.Log("[Geant4] Startup sequence...");

            // --- POPRAWKA 1: PREWENCYJNE CZYSZCZENIE ---
            // Zanim spróbujemy cokolwiek stworzyć, upewnijmy się, że C++ jest czysty.
            // Jeśli poprzednia sesja zostawiła śmieci, to je usunie.
            try
            {
                Geant4Interface.CloseGeant4();
            }
            catch { /* Ignorujemy błędy przy czyszczeniu, bo może być już czysto */ }

            // --- INICJALIZACJA ---
            Debug.Log("[Geant4] Initializing Physics Engine...");
            try
            {
                Geant4Interface.InitGeant4();
                Debug.Log("[Geant4] ✅ Initialization Success");
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[Geant4] ❌ Initialization Failed: {e.Message}");
                // Jeśli init się nie udał, od razu posprzątajmy
                Geant4Interface.CloseGeant4();
            }
        }

        // --- POPRAWKA 2: OnDestroy zamiast OnApplicationQuit ---
        // OnDestroy jest wołane zawsze gdy obiekt jest niszczony (np. przy Stop w Edytorze).
        // Jest pewniejsze w trybie edycji niż OnApplicationQuit.
        void OnDestroy()
        {
            // Upewniamy się, że to my niszczymy instancję (a nie duplikat)
            if (Instance == this)
            {
                Debug.Log("[Geant4] Cleaning up resources (OnDestroy)...");
                Geant4Interface.CloseGeant4();
            }
        }
    }
}