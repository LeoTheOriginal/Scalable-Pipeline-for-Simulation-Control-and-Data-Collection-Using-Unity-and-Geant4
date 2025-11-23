using UnityEngine;
using MessagePack;

namespace Core
{
    // To musi pasować do tego, co wysyła Python (msgpack)
    [MessagePackObject]
    public struct TrajectoryBatch
    {
        [Key("count")]
        public int Count;
        [Key("steps")]
        public int Steps;
        [Key("data")]
        public byte[] RawData; // Skompresowane floaty [x, y, z, x, y, z...]
    }
}