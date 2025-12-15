cmake_minimum_required(VERSION 3.16...3.27)

# ============================================================================
# Project Definition
# ============================================================================
project(WaterPhantomSim VERSION 2.0.0)

# ============================================================================
# C++ Standard
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Find Dependencies
# ============================================================================

# Find Geant4 (static libraries)
find_package(Geant4 REQUIRED ui_all vis_all)
include(${Geant4_USE_FILE})

# ============================================================================
# Source Management
# ============================================================================

set(COMMON_SOURCES
    src/ActionInitialization.cc
    src/DetectorConstruction.cc
    src/EventAction.cc
    src/PhysicsList.cc
    src/PrimaryGeneratorAction.cc
    src/RunAction.cc
    src/SteppingAction.cc
)

set(COMMON_HEADERS
    include/ActionInitialization.hh
    include/DetectorConstruction.hh
    include/EventAction.hh
    include/PhysicsList.hh
    include/PrimaryGeneratorAction.hh
    include/RunAction.hh
    include/SteppingAction.hh
)

# ============================================================================
# Target: Unity Native Plugin (Shared Library / DLL)
# ============================================================================

# Tworzymy bibliotekę współdzieloną dla Unity
add_library(geant4_plugin SHARED
    src/Geant4Plugin.cc
    ${COMMON_SOURCES}
    ${COMMON_HEADERS}
)

target_include_directories(geant4_plugin PRIVATE ${PROJECT_SOURCE_DIR}/include)

# WAŻNE: Linkowanie static libraries Geant4
target_link_libraries(geant4_plugin PRIVATE ${Geant4_LIBRARIES})

# ============================================================================
# Output Configuration (Deployment to Unity)
# ============================================================================

# Ścieżka do folderu Plugins w Unity
get_filename_component(UNITY_PLUGIN_DIR "${CMAKE_SOURCE_DIR}/../../unity/GeantML_Test/Assets/Plugins" ABSOLUTE)

# Stwórz folder jeśli nie istnieje
file(MAKE_DIRECTORY ${UNITY_PLUGIN_DIR})

message(STATUS "Unity Plugin Directory: ${UNITY_PLUGIN_DIR}")

set_target_properties(geant4_plugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${UNITY_PLUGIN_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${UNITY_PLUGIN_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${UNITY_PLUGIN_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${UNITY_PLUGIN_DIR}"
    PREFIX ""
    OUTPUT_NAME "geant4_plugin"
)

# ============================================================================
# USUNIĘTE: Kopiowanie DLL
# ============================================================================
# Przy static libraries NIE MA żadnych DLL do kopiowania!
# Wszystko jest zlinkowane wewnątrz geant4_plugin.dll

message(STATUS "Building with STATIC Geant4 libraries")
message(STATUS "geant4_plugin.dll will contain all Geant4 code (~50-80 MB)")