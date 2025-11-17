#include <G4VModularPhysicsList.hh>

class PhysicsList : public G4VModularPhysicsList {
public:
    PhysicsList();
    ~PhysicsList() override;
    void SetCuts() override;
};
