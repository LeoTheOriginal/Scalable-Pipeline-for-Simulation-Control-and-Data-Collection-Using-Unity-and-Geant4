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

# 1. Find Geant4 (Tylko to jest potrzebne, Python i pybind11 idą do kosza)
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

# Tworzymy bibliotekę współdzieloną, a nie moduł Pythona
add_library(geant4_plugin SHARED
    src/Geant4Plugin.cc  # <-- TO BĘDZIE NOWY PLIK (Most C++/C#)
    ${COMMON_SOURCES}
    ${COMMON_HEADERS}
)

target_include_directories(geant4_plugin PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(geant4_plugin PRIVATE ${Geant4_LIBRARIES})

# ============================================================================
# Output Configuration (Deployment to Unity)
# ============================================================================

# Ścieżka do folderu Plugins w Unity (Dostosuj jeśli nazwa folderu Unity jest inna)
# Wychodzimy z geant4/Water-Phantom (..) -> geant4 (..) -> Thesis -> unity/GeantML_Test/Assets/Plugins
get_filename_component(UNITY_PLUGIN_DIR "${CMAKE_SOURCE_DIR}/../../unity/GeantML_Test/Assets/Plugins" ABSOLUTE)

# Jeśli folder nie istnieje, stwórz go (żeby CMake nie krzyczał)
file(MAKE_DIRECTORY ${UNITY_PLUGIN_DIR})

message(STATUS "Unity Plugin Directory: ${UNITY_PLUGIN_DIR}")

# ... (cała góra pliku bez zmian) ...

set_target_properties(geant4_plugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${UNITY_PLUGIN_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${UNITY_PLUGIN_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${UNITY_PLUGIN_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${UNITY_PLUGIN_DIR}"
    PREFIX ""
    OUTPUT_NAME "geant4_plugin"
)

# === POPRAWIONA SEKCJA KOPIOWANIA ===
# Zamiast bawić się w relatywne ścieżki, wpiszmy to na sztywno, bo wiemy gdzie to jest.
set(GEANT4_BIN_DIR "C:/Geant4/install/bin")

# Komenda kopiująca G4global.dll (i inne krytyczne, jeśli chcesz dodać)
add_custom_command(TARGET geant4_plugin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${GEANT4_BIN_DIR}/G4global.dll"
    "${UNITY_PLUGIN_DIR}/"

    # Geant4 potrzebuje też tych bibliotek (dodaję najwazniejsze):
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${GEANT4_BIN_DIR}/G4run.dll" "${UNITY_PLUGIN_DIR}/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${GEANT4_BIN_DIR}/G4event.dll" "${UNITY_PLUGIN_DIR}/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${GEANT4_BIN_DIR}/G4tracking.dll" "${UNITY_PLUGIN_DIR}/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${GEANT4_BIN_DIR}/G4processes.dll" "${UNITY_PLUGIN_DIR}/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${GEANT4_BIN_DIR}/G4particles.dll" "${UNITY_PLUGIN_DIR}/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${GEANT4_BIN_DIR}/G4geometry.dll" "${UNITY_PLUGIN_DIR}/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${GEANT4_BIN_DIR}/G4materials.dll" "${UNITY_PLUGIN_DIR}/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${GEANT4_BIN_DIR}/G4intercoms.dll" "${UNITY_PLUGIN_DIR}/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${GEANT4_BIN_DIR}/G4track.dll" "${UNITY_PLUGIN_DIR}/"

    COMMENT "Kopiowanie bibliotek DLL Geant4 do Unity..."
)