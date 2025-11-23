using System.Runtime.InteropServices;
using UnityEngine;

namespace Core
{
    public static class Geant4Interface
    {
        // Nazwa musi pasować do OUTPUT_NAME w CMakeLists.txt
        private const string DLL_NAME = "geant4_plugin";

        // Inicjalizacja silnika fizycznego (Wołane w Awake)
        [DllImport(DLL_NAME)]
        public static extern void InitGeant4();

        // Sprzątanie pamięci (Wołane w OnApplicationQuit)
        [DllImport(DLL_NAME)]
        public static extern void CloseGeant4();

        // Główna funkcja: Pobiera całą trajektorię jednej cząstki
        // outData: Tablica floatów [maxSteps * 7]
        // maxSteps: Limit kroków
        // Zwraca: Liczbę faktycznych kroków
        [DllImport(DLL_NAME)]
        public static extern int RunSimulationBatch([In, Out] float[] outData, int maxSteps);
    }
}