#include "ActionInitialization.hh"

#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "EventAction.hh"
#include "SteppingAction.hh"

ActionInitialization::ActionInitialization() = default;
ActionInitialization::~ActionInitialization() = default;

void ActionInitialization::BuildForMaster() const {
  SetUserAction(new RunAction());
}

void ActionInitialization::Build() const {

  auto* primary = new PrimaryGeneratorAction();
  SetUserAction(primary);

  auto* runAction = new RunAction();
  SetUserAction(runAction);

  auto* eventAction = new EventAction();
  SetUserAction(eventAction);

  auto* steppingAction = new SteppingAction(eventAction);
  SetUserAction(steppingAction);
}
