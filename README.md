# Scalable Pipeline for Simulation Control and Data Collection Using Unity and Geant4

Engineering thesis project · **Faculty of Physics and Applied Computer Science · AGH University of Science and Technology** · academic year 2024/2025

Author: **Dawid Piotrowski** ([@LeoTheOriginal](https://github.com/LeoTheOriginal))
Supervisor: **prof. dr hab. inż. Tomasz Szumlak** — Department of Particle Interactions and Detection (KOiDC), AGH WFiS

---

## Overview

Monte Carlo simulations of radiation–matter interactions (e.g. **Geant4**) are the gold standard for detector physics studies, but they are **expensive**: a single high-fidelity track can take milliseconds, and detector efficiency studies routinely require millions of them. This project explores whether a **Reinforcement Learning agent**, trained inside **Unity ML Agents**, can act as a **surrogate generator** — producing statistically similar particle trajectories at a small fraction of the cost.

The contribution of this thesis is **the pipeline itself**: a working bridge between Geant4 (the ground-truth physics simulator) and Unity (the RL training environment), so that the *physics question* (can RL learn radiation transport?) can be tackled without first having to build the entire infrastructure.

A toy detector — a **water phantom (Water-Phantom)** — is used as the testbed: a simple cubic volume of water through which charged particles propagate and ionise.

---

## Architecture

```
┌───────────────────────┐        DLL bridge          ┌──────────────────────────┐
│   Geant4 (C++)        │  ─── per-step 4-vector ──▶ │  Unity 3D + ML Agents    │
│   Water-Phantom       │     (E, px, py, pz, x,y,z) │  (C# observation +       │
│   primary generator   │                            │   policy network in      │
│   step-by-step output │ ◀── ready/next-event ───── │   PyTorch via mlagents)  │
└───────────────────────┘                            └──────────────────────────┘
                                                                  │
                                                                  ▼
                                                     ┌──────────────────────────┐
                                                     │  Python analysis stack   │
                                                     │  (matplotlib, pandas,    │
                                                     │   metrics, TikZ figures) │
                                                     └──────────────────────────┘
```

Key design decisions:

- **On-the-fly data flow** — particles are generated and consumed step-by-step rather than buffered to disk. Avoids the 50-format zoo of HEP data formats and keeps the system stream-shaped.
- **Native DLL bridge (Windows)** — chosen for low overhead vs. gRPC/Protobuf alternatives explored in the project.
- **Unity ML Agents as the training framework** — gives a "world for the agent" essentially for free, lets the project focus on the physics rather than building an RL framework from scratch.

---

## Repository structure

```
.
├── geant4/
│   └── Water-Phantom/        # Geant4 application: cubic water volume,
│                             # primary particle gun, per-step instrumentation
│                             # exposing (energy, momentum, position) over the bridge
│
├── unity/
│   └── GeantML_Test/         # Unity project + ML Agents integration
│                             # (C# Agent, observation parser, training scenes)
│
├── python/
│   ├── data/                 # processed datasets and exports
│   ├── metrics/              # evaluation metrics (track length, energy deposit, …)
│   └── figures/              # plots and TikZ figures used in the thesis report
│
├── environment.yaml          # conda env (mlagents + scientific Python stack)
└── .gitignore                # ignores Unity Library/Temp/Build, Geant4 build/,
                              # IDE clutter, large regenerable datasets
```

The `.gitignore` is deliberate: large regenerable artefacts (raw point clouds, density textures, intermediate `*.csv` event dumps, ROOT/HDF5 shared data) are kept **out of git**. Anything tracked is intended to be reproducible from the simulation itself.

---

## Tech stack

| Layer | Technology |
|---|---|
| Physics ground truth | **Geant4** (Monte Carlo radiation transport) — C++ |
| Build system (Geant4) | CMake |
| Training environment | **Unity 3D** + **Unity ML Agents** — C# scripts + Python `mlagents` trainer |
| ML backend | **PyTorch** (via the `mlagents` package) |
| Data / analysis | **Python** (pandas, NumPy, matplotlib) |
| Inter-process bridge | Native **DLL** on Windows (low-latency, in-process) |
| Reporting figures | matplotlib + **TikZ** export (`*.tex` for direct inclusion in LaTeX) |

---

## Setup

```bash
# 1. Conda environment (includes Python, PyTorch, mlagents, analysis stack)
conda env create -f environment.yaml
conda activate ml-agents

# 2. Geant4 — build the Water-Phantom application
cd geant4/Water-Phantom
mkdir build && cd build
cmake .. && cmake --build . --config Release

# 3. Unity — open unity/GeantML_Test in Unity Hub (matching ML Agents version)
#    Trigger training from the included scene; the DLL bridge wires Geant4 → Agent.
```

Tested on Windows (the DLL bridge is Windows-specific in this iteration; Linux/Docker ports are explored in the master's thesis follow-up — see *Related work* below).

---

## Status

Engineering thesis **completed** in academic year 2024/2025. The repository is preserved as the **baseline** for the master's thesis continuation, where the same pipeline is being:
- ported to Linux/Docker (containerised Geant4 + headless Unity),
- evaluated against alternative transports (gRPC, raw TCP, ZeroMQ),
- extended with richer detector geometries (beyond the water cube) and adaptive time-stepping for the Unity ↔ Geant4 step-rate mismatch.

---

## Related work

- **Master's thesis** (continuation, in progress, 2025/2026 →): `LeoTheOriginal/is-mgr-kod`, `…-docs`, `…-meetings` (private).
- **Research group:** [`RL4Phy-AGH`](https://github.com/RL4Phy-AGH) — *Reinforcement Learning for Physics simulation*, group around prof. Szumlak. Engineering theses of all current group members are archived in [`RL4Phy-AGH/legacy_code`](https://github.com/RL4Phy-AGH/legacy_code), each in its own subfolder.

---

## License & use

Private repository — all rights reserved (academic work, AGH UST). For collaboration or citation requests, please contact the author.
