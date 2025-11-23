using System;
using System.Collections.Concurrent;
using UnityEngine;
using Core;
using Networking;
using Performance;

namespace Visualization
{
    public class ParticleVisualizer : MonoBehaviour
    {
        [Header("Modules")]
        public ServerConnection server;

        [Header("Visuals")]
        public BatchRenderer realRenderer; // Renderer dla Prawdy (Zielony)
        public BatchRenderer aiRenderer;   // Renderer dla AI (Czerwony)

        private ConcurrentQueue<TrajectoryBatch> _incomingBatches = new ConcurrentQueue<TrajectoryBatch>();
        private float[] _positionBuffer;

        void Start()
        {
            server.OnBatchReceived += HandleNewBatch;
            server.Connect();
        }

        private void HandleNewBatch(TrajectoryBatch batch)
        {
            _incomingBatches.Enqueue(batch);
        }

        void Update()
        {
            if (_incomingBatches.TryDequeue(out TrajectoryBatch batch))
            {
                ProcessData(batch);
            }

            if (realRenderer != null) realRenderer.Render();
            if (aiRenderer != null) aiRenderer.Render();
        }

        private void ProcessData(TrajectoryBatch batch)
        {
            int floatCount = batch.RawData.Length / 4;
            if (_positionBuffer == null || _positionBuffer.Length != floatCount)
            {
                _positionBuffer = new float[floatCount];
            }
            Buffer.BlockCopy(batch.RawData, 0, _positionBuffer, 0, batch.RawData.Length);

            // DZIELIMY DANE NA PÓŁ
            // Python wysyła [Real... AI...]
            int totalParticles = batch.Count;
            int halfParticles = totalParticles / 2;

            // 1. Rysuj Prawdziwe (Pierwsza połowa)
            // Przekazujemy do renderera całą tablicę, ale mówimy:
            // "Rysuj od indeksu 0, tyle sztuk: halfParticles"
            // (Wymaga modyfikacji BatchRenderer, którą zaraz podam)
            realRenderer.PrepareBatch(_positionBuffer, batch.Steps, 0, halfParticles);

            // 2. Rysuj AI (Druga połowa)
            aiRenderer.PrepareBatch(_positionBuffer, batch.Steps, halfParticles, halfParticles);
        }
    }
}