using Core;
using System.Collections.Generic;
using UnityEngine;

namespace Performance
{
    public class BatchRenderer : MonoBehaviour
    {
        [Header("Rendering")]
        public Mesh particleMesh;
        public Material particleMat;

        private List<Matrix4x4[]> _matricesBatches = new List<Matrix4x4[]>();
        private const int INSTANCE_BATCH_SIZE = 1000;

        // Licznik do ograniczania spamu w konsoli
        private int _debugLogCounter = 0;

        // ZMIANA: Usunąłem argument 'count' (był mylący/nieużywany), teraz mamy jasne 'offset' i 'numParticles'
        public void PrepareBatch(float[] flatPositions, int steps, int offsetParticles, int numParticlesToDraw)
        {
            _matricesBatches.Clear();

            if (flatPositions == null || flatPositions.Length == 0) return;

            Matrix4x4[] currentBatch = new Matrix4x4[INSTANCE_BATCH_SIZE];
            int inBatchCount = 0;

            bool shouldLog = (_debugLogCounter++ % 60 == 0); // Loguj rzadziej

            // --- ZMIANA KLUCZOWA: Pętla po cząstkach ---
            // Iterujemy od wskazanego offsetu przez zadaną liczbę cząstek
            for (int i = offsetParticles; i < offsetParticles + numParticlesToDraw; i++)
            {
                // Pętla po krokach czasowych dla cząstki 'i'
                for (int s = 0; s < steps; s++)
                {
                    // Oblicz indeks w płaskiej tablicy dla cząstki 'i' w kroku 's'
                    int baseIdx = (i * steps * 3) + (s * 3);

                    // Zabezpieczenie przed wyjściem poza tablicę
                    if (baseIdx + 2 >= flatPositions.Length) break;

                    float x = flatPositions[baseIdx];
                    float y = flatPositions[baseIdx + 1];
                    float z = flatPositions[baseIdx + 2];

                    // --- FILTROWANIE ZER (PADDINGU) ---
                    // Ignorujemy puste dane (0,0,0), chyba że to punkt startowy
                    if (s > 0 && Mathf.Abs(x) < 0.001f && Mathf.Abs(y) < 0.001f && Mathf.Abs(z) < 0.001f)
                    {
                        continue;
                    }

                    // LOGOWANIE (Tylko dla pierwszej rysowanej cząstki w tej grupie, żeby nie spamować)
                    if (shouldLog && i == offsetParticles)
                    {
                        if (s == 0) Debug.Log($"[BatchRenderer] START (Cząstka {i}): ({x:F2}, {y:F2}, {z:F2})");
                        // Ostatni punkt jest trudniejszy do złapania w pętli z continue, więc odpuszczamy logowanie końca dla wydajności
                    }

                    currentBatch[inBatchCount] = Matrix4x4.TRS(
                        new Vector3(x, y, z),
                        Quaternion.identity,
                        Vector3.one * 0.05f // Skala kulki
                    );

                    inBatchCount++;

                    // Jeśli zapełniliśmy batch (1000 kulek), tworzymy nowy
                    if (inBatchCount >= INSTANCE_BATCH_SIZE)
                    {
                        _matricesBatches.Add(currentBatch);
                        currentBatch = new Matrix4x4[INSTANCE_BATCH_SIZE];
                        inBatchCount = 0;
                    }
                }
            }
            // -------------------------------------------

            // Dodajemy ostatni, niepełny batch
            if (inBatchCount > 0)
            {
                _matricesBatches.Add(currentBatch);
            }
        }

        public void Render()
        {
            foreach (var batch in _matricesBatches)
            {
                Graphics.DrawMeshInstanced(particleMesh, 0, particleMat, batch);
            }
        }
    }
}