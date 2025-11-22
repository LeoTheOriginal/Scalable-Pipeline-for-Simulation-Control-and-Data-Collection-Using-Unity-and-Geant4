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

# 1. Find Geant4
find_package(Geant4 REQUIRED ui_all vis_all)
include(${Geant4_USE_FILE})

# 2. Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# 3. Find Pybind11
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)

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
# Target 1: Standalone Executable
# ============================================================================
add_executable(WaterPhantomApp
    main.cc
    ${COMMON_SOURCES}
    ${COMMON_HEADERS}
)

target_include_directories(WaterPhantomApp PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(WaterPhantomApp ${Geant4_LIBRARIES})

# ============================================================================
# Target 2: Python Module
# ============================================================================
pybind11_add_module(geant4_sim
    PyGeant4.cc
    ${COMMON_SOURCES}
    ${COMMON_HEADERS}
)

target_include_directories(geant4_sim PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(geant4_sim PRIVATE ${Geant4_LIBRARIES})

# --- AUTOMATYCZNE UMIESZCZANIE W STRUKTURZE PYTHONA ---

# Obliczamy ścieżkę bezwzględną do python/src/simulation
# Relatywnie: ../../python/src/simulation
get_filename_component(PYTHON_SIM_DIR "${CMAKE_SOURCE_DIR}/../../python/src/simulation" ABSOLUTE)

message(STATUS "Target Python Directory: ${PYTHON_SIM_DIR}")

set_target_properties(geant4_sim PROPERTIES
    # Wymuszamy zapis pliku .pyd bezpośrednio do folderu źródłowego Pythona
    LIBRARY_OUTPUT_DIRECTORY "${PYTHON_SIM_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${PYTHON_SIM_DIR}"

    # To samo dla konfiguracji Release (Visual Studio tworzy podfoldery Release/Debug)
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${PYTHON_SIM_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PYTHON_SIM_DIR}"

    PREFIX ""
    OUTPUT_NAME "geant4_sim"
    SUFFIX ".pyd"
)
# ------------------------------------------------------

# ============================================================================
# Macros Copy
# ============================================================================
file(GLOB macro_files ${PROJECT_SOURCE_DIR}/macros/*.mac)
foreach(_file ${macro_files})
    configure_file(${_file} ${CMAKE_BINARY_DIR}/COPYONLY)
endforeach()

message(STATUS "Python Found: ${Python_EXECUTABLE}")
message(STATUS "Pybind11 Found: ${pybind11_DIR}")