using Core;
using K4os.Compression.LZ4;
using MessagePack;
using System;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace Networking
{
    public class ServerConnection : MonoBehaviour
    {
        [Header("Settings")]
        public string serverUrl = "ws://localhost:8000/ws";

        public event Action<TrajectoryBatch> OnBatchReceived;

        private ClientWebSocket _ws;
        private CancellationTokenSource _cts;

        // Zwiększamy bufory dla bezpieczeństwa
        private byte[] _receiveBuffer = new byte[4 * 1024 * 1024]; // 4MB na skompresowane dane
        private byte[] _lz4Buffer = new byte[16 * 1024 * 1024];    // 16MB na pozycje (duży zapas)

        public async void Connect()
        {
            // Zawsze twórz nową instancję przy ponownym łączeniu
            _ws?.Dispose();
            _ws = new ClientWebSocket();
            _cts = new CancellationTokenSource();

            try
            {
                Debug.Log($"[Network] Connecting to {serverUrl}...");
                await _ws.ConnectAsync(new Uri(serverUrl), _cts.Token);
                Debug.Log("[Network] Connected!");

                _ = ReceiveLoop();
            }
            catch (Exception e)
            {
                Debug.LogError($"[Network] Connection failed: {e.Message}");
            }
        }

        private async Task ReceiveLoop()
        {
            while (_ws.State == WebSocketState.Open)
            {
                try
                {
                    int totalBytesReceived = 0;

                    // ZMIANA: Inicjalizujemy nullem, żeby kompilator nie krzyczał
                    WebSocketReceiveResult result = null;

                    do
                    {
                        int freeSpace = _receiveBuffer.Length - totalBytesReceived;

                        if (freeSpace <= 0)
                        {
                            Debug.LogError("[Network] Receive Buffer Overflow! Zwiększ _receiveBuffer.");
                            // Jeśli tu przerwiemy, result pozostanie nullem
                            break;
                        }

                        result = await _ws.ReceiveAsync(
                            new ArraySegment<byte>(_receiveBuffer, totalBytesReceived, freeSpace),
                            _cts.Token
                        );

                        totalBytesReceived += result.Count;

                    } while (!result.EndOfMessage);

                    // ZMIANA: Sprawdzamy czy result nie jest nullem przed dostępem do właściwości
                    if (result != null)
                    {
                        if (result.MessageType == WebSocketMessageType.Binary)
                        {
                            // Opcjonalny debug (odkomentuj jeśli potrzebujesz)
                            // DebugLogHeader(totalBytesReceived);

                            ProcessMessage(totalBytesReceived);
                        }
                        else if (result.MessageType == WebSocketMessageType.Close)
                        {
                            Debug.Log("[Network] Server requested close.");
                            await _ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "OK", CancellationToken.None);
                        }
                    }
                }
                catch (Exception e)
                {
                    if (e is OperationCanceledException) break;

                    Debug.LogError($"[Network] Loop error: {e.Message}\n{e.StackTrace}");
                    await Task.Delay(1000);
                }
            }
        }

        private void DebugLogHeader(int count)
        {
            // Wypisz rozmiar i pierwsze 4 bajty (magic numbers)
            if (count > 0)
            {
                string hex = BitConverter.ToString(_receiveBuffer, 0, Math.Min(4, count));
                // Debug.Log($"[Network] Received {count} bytes. Header (hex): {hex}");
                // Normalnie to zakomentuj, odkomentuj jak nadal będzie błąd
            }
        }

        private void ProcessMessage(int count)
        {
            try
            {
                if (count == 0) return;

                // 1. Dekompresja LZ4
                var compressedData = new Span<byte>(_receiveBuffer, 0, count);

                // Używamy Decode z podaniem rozmiaru wyjściowego bufferu, zwraca ile faktycznie odkodował
                // UWAGA: Jeśli Python używa block.compress(store_size=False), Unity NIE WIE jaki jest rozmiar wyjściowy.
                // LZ4Codec.Decode(input, output) po prostu jedzie do końca inputu.

                int decodedSize = LZ4Codec.Decode(compressedData, _lz4Buffer);

                if (decodedSize < 0)
                {
                    Debug.LogError($"[Network] LZ4 Decode failed (returned {decodedSize}). Może zły format danych?");
                    return;
                }

                // 2. Deserializacja MessagePack
                var batch = MessagePackSerializer.Deserialize<TrajectoryBatch>(
                    new ReadOnlyMemory<byte>(_lz4Buffer, 0, decodedSize));

                // 3. Powiadomienie
                OnBatchReceived?.Invoke(batch);
            }
            catch (Exception e)
            {
                Debug.LogError($"[Network] Processing error: {e.Message}");
                // Dodatkowy debug w przypadku błędu
                string header = BitConverter.ToString(_receiveBuffer, 0, Math.Min(10, count));
                Debug.LogError($"[Network] Dane wejściowe ({count} bytes): {header}...");
            }
        }

        private void OnDestroy()
        {
            _cts?.Cancel();
            _ws?.Dispose();
        }
    }
}