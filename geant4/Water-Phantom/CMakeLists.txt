cmake_minimum_required(VERSION 3.12)
project(WaterPhantomSim)

find_package(Geant4 REQUIRED ui_all vis_all)
include(${Geant4_USE_FILE})

add_executable(WaterPhantomSim
  src/ActionInitialization.cc
  src/DetectorConstruction.cc
  src/EventAction.cc
  src/PhysicsList.cc
  src/PrimaryGeneratorAction.cc
  src/RunAction.cc
  src/SteppingAction.cc
  src/UnityMLInterface.cc
  main.cc
)
target_include_directories(WaterPhantomSim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(WaterPhantomSim ${Geant4_LIBRARIES})
