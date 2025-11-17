#ifndef UNITY_ML_INTERFACE_HH
#define UNITY_ML_INTERFACE_HH

#include <G4Step.hh>

class UnityMLInterface {
public:
    static UnityMLInterface& Instance() {
        static UnityMLInterface instance;
        return instance;
    }
    void CollectObservation(const G4Step* step);
private:
    UnityMLInterface() = default;
    ~UnityMLInterface() = default;
    UnityMLInterface(const UnityMLInterface&) = delete;
    UnityMLInterface& operator=(const UnityMLInterface&) = delete;
};

#endif
