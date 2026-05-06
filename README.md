# Scalable Pipeline for Simulation Control and Data Collection<br/>Using Unity and Geant4

> **A bridge between Geant4 (Monte Carlo physics) and Unity ML Agents (Reinforcement Learning) — built so that the *physics question* (can RL learn radiation transport?) can be tackled without first building the entire infrastructure.**

<sub>Engineering thesis · Faculty of Physics and Applied Computer Science · AGH University of Science and Technology · 2025/2026</sub>
<sub>Author: **Dawid Piotrowski** ([@LeoTheOriginal](https://github.com/LeoTheOriginal))</sub>

[![Geant4](https://img.shields.io/badge/Geant4-Livermore_EM-005C9C?style=flat-square)](https://geant4.web.cern.ch/)
[![Unity](https://img.shields.io/badge/Unity-ML_Agents-000000?style=flat-square&logo=unity)](https://github.com/Unity-Technologies/ml-agents)
[![PyTorch](https://img.shields.io/badge/PyTorch-PPO_·_PPO+LSTM_·_SAC-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?style=flat-square&logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![C#](https://img.shields.io/badge/C%23-Unity-239120?style=flat-square&logo=csharp)](https://learn.microsoft.com/dotnet/csharp/)

---

## At a glance

- **Problem.** Detector design studies need millions of Monte Carlo tracks. Geant4 is the gold standard — and it's slow.
- **Idea.** Train a Reinforcement Learning agent inside Unity ML Agents to act as a **surrogate generator** that reproduces Geant4-quality electron tracks at a fraction of the cost.
- **Contribution of this thesis.** A working **pipeline**: a low-latency native bridge between Geant4 (C++) and Unity ML Agents (C# + PyTorch), a physics-informed RL agent (`ElectronAgentPhysics`), and a comparison harness for three RL algorithms (PPO, PPO+LSTM, SAC). Validated on a toy water phantom.
- Maintained as a baseline for ongoing master's-thesis work.

---

## Concrete setup (what's actually simulated)

| | Value |
|---|---|
| Phantom | **10 × 10 × 10 cm³** water box (NIST `G4_WATER`), centred at origin in a 30³ cm³ air world |
| Primary | **Single electron**, **10 MeV** kinetic energy, perpendicular incidence at `(−6, 0, 0)` cm |
| Geant4 physics list | `G4EmLivermorePolarizedPhysics` (precise low-energy EM) with default 1 mm production cut |
| Per-step record | `position`, `momentum`, `kineticEnergy`, `energyDeposited`, `scatterAngle`, `processName` |
| Wire format | **7 floats per step** — `[x, y, z, px, py, pz, E_kin]` in cm / MeV |

---

## System architecture

```mermaid
flowchart LR
    subgraph G["Geant4 — ground truth physics"]
        direction TB
        G1["Water-Phantom<br/>10×10×10 cm³ (C++)"]
        G2["Particle gun:<br/>10 MeV e⁻ at (−6,0,0)"]
        G3["G4EmLivermorePolarizedPhysics<br/>step-by-step tracking"]
        G1 --> G2 --> G3
    end

    subgraph B["Native Bridge (geant4_plugin.dll)"]
        direction TB
        B1["11 exported C functions:<br/>InitGeant4 · RunSimulationBatch<br/>RunBatchSimulation · GetBatchStatistics<br/>GetBatchTrajectoryData · GetLateralDistribution<br/>ExportStatisticsToFile · BenchmarkGeant4Performance · …"]
    end

    subgraph U["Unity 3D + ML Agents"]
        direction TB
        U1["ElectronAgentPhysics (C#)<br/>3 modes: PhysicsBased ·<br/>Geant4Statistical · Inference"]
        U2["Observation: 7-dim<br/>(pos x,y,z · dir x,y,z · E)"]
        U3["Action: 7-dim continuous<br/>(Δpos · Δmom · ΔE)"]
        U4["Three-phase, energy-dependent reward<br/>(initial penetration → transition → deep scattering)"]
        U5["Policy network (PyTorch via mlagents)<br/>+ Curiosity intrinsic signal<br/>+ Scheduled Sampling / Teacher Forcing"]
        U1 --> U2 & U3
        U2 --> U5
        U3 --> U4 --> U5
    end

    subgraph A["Python analysis"]
        direction TB
        A1["pandas / NumPy<br/>+ NIST water reference data"]
        A2["track, lateral spread, energy<br/>and convergence metrics"]
        A3["matplotlib + TikZ export<br/>→ thesis figures"]
        A1 --> A2 --> A3
    end

    G3 -- "RunSimulationBatch<br/>(7 floats × N steps)" --> B1
    B1 --> U1
    U5 -. "ElectronBehavior.onnx<br/>(trained, runs without Geant4)" .-> A1
    G3 -. "reference tracks (CSV)" .-> A1

    classDef g fill:#0d3b66,stroke:#0d3b66,color:#fff;
    classDef b fill:#7d4f50,stroke:#7d4f50,color:#fff;
    classDef u fill:#222,stroke:#222,color:#fff;
    classDef a fill:#1f5f3f,stroke:#1f5f3f,color:#fff;
    class G1,G2,G3 g;
    class B1 b;
    class U1,U2,U3,U4,U5 u;
    class A1,A2,A3 a;
```

The native DLL (`geant4_plugin.dll`, called from C# via `DllImport`) was chosen over gRPC + Protobuf alternatives explored during the project — direct in-process float-buffer exchange keeps inter-process overhead in the µs range, which matters when an episode produces hundreds of physics steps per particle. A secondary **MessagePack + LZ4** path (`Core.TrajectoryBatch` in C#, the `MessagePack` and `K4os.Compression.LZ4` NuGet packages) is wired for batch loading of pre-recorded Python ground-truth datasets.

---

## Training loop (per particle)

```mermaid
sequenceDiagram
    autonumber
    participant G as Geant4 (C++)
    participant D as DLL bridge
    participant A as ElectronAgentPhysics (C#)
    participant T as mlagents trainer (Python / PyTorch)

    G->>D: BeamOn(1)  spawn 10 MeV e⁻
    D->>A: trajectory buffer (7 floats × N steps)
    Note over A: Scheduled Sampling decides:<br/>ground-truth observation (teacher) vs<br/>agent's own predicted state
    loop while particle alive in phantom
        A->>T: observation (pos, dir, E)
        T-->>A: action (Δpos, Δmom, ΔE) ∈ ℝ⁷
        A->>A: three-phase reward<br/>(initial penetration · transition ·<br/>deep scattering — energy-gated)<br/>+ angular diversity · progressive boundary
    end
    A->>A: episode done (energy depleted /<br/>boundary exit / max steps)
    A->>T: episode return + curiosity bonus
    T->>T: backprop · policy update
```

The reward function is **physics-informed** rather than purely imitation-based and is split into three energy-gated phases (*initial penetration* → *transition* → *deep scattering*, switched at `E > 0.75 E₀` and `E ≤ 0.4 E₀`). Each phase shapes a different aspect of the trajectory: forward progress at high energy, balanced scattering during transition, and survival/diversity at low energy. Two cross-cutting mechanisms — *angular-diversity* (anti-mode-collapse) and *progressive boundary penalties* — operate in all phases.

**Scheduled Sampling / Teacher Forcing** is annealed over 10 000 episodes from full ground-truth observations down to 10 % — letting the policy gradually take over while keeping a small ground-truth signal to avoid drift.

---

## Repository structure

```
.
├── geant4/
│   └── Water-Phantom/                 # Geant4 application:
│       ├── src/Geant4Plugin.cc        #   the DLL bridge (11 C exports)
│       ├── src/SteppingAction.cc      #   per-step instrumentation
│       ├── src/PhysicsList.cc         #   G4EmLivermorePolarizedPhysics
│       ├── src/DetectorConstruction.cc#   10×10×10 cm³ water phantom
│       └── src/PrimaryGeneratorAction.cc # 10 MeV electron, +X
│
├── unity/
│   └── GeantML_Test/Assets/Scripts/
│       ├── Agents/ElectronAgentPhysics.cs  # main RL agent (V6)
│       ├── Agents/NormalDistributionRewards.cs # Gaussian-PDF reward helpers
│       ├── Core/Geant4Interface.cs    # P/Invoke into geant4_plugin.dll
│       ├── Core/Geant4Manager.cs      # Geant4 lifecycle (singleton)
│       ├── Core/DataModels.cs         # MessagePack types for batch ingest
│       ├── Visualization/             # trajectory + density visualisers
│       └── Benchmarking/              # PerformanceBenchmark.cs
│
├── python/
│   ├── data/                          # geant4 + ppo/sac trajectories,
│   │                                  # nist_water_data.txt, performance JSON
│   ├── metrics/                       # mode-collapse + JSON metric extractors
│   └── figures/                       # plotting scripts + PDF/PNG/TikZ outputs
│
├── environment.yaml                   # conda env (mlagents + scientific Python)
└── .gitignore                         # excludes Unity Library/Temp/Build,
                                       # Geant4 build/, IDE clutter,
                                       # large regenerable datasets
```

Anything tracked in git is **reproducible from simulation**. Large regenerable artefacts (raw point clouds, density textures, intermediate `*.csv` event dumps, ROOT/HDF5 shared data) are kept out by design.

---

## Tech stack

| Layer | Technology | Role |
|---|---|---|
| Physics | [**Geant4 11.3.0**](https://geant4.web.cern.ch/) (C++17), static-linked, with `G4EmLivermorePolarizedPhysics` | Ground-truth radiation transport |
| Build | CMake | Geant4 application + DLL build |
| Bridge | Native **`geant4_plugin.dll`** (Windows, `extern "C"`, 11 exports) | Geant4 ↔ Unity inter-process |
| Secondary bridge | **MessagePack** + **K4os.Compression.LZ4** | Python-recorded ground-truth ingest |
| Environment | [**Unity 2022.3 LTS**](https://unity.com/) + [**ML Agents**](https://github.com/Unity-Technologies/ml-agents) | RL world + framework |
| Agent code | C# (`ElectronAgentPhysics`, three training modes) | Observations, action space, reward shaping |
| ML backend | [**PyTorch**](https://pytorch.org/) via `mlagents` trainer | Policy network training |
| Trained model | **ONNX** (`ElectronBehavior.onnx`) | Inference inside Unity, no Python needed |
| Analysis | Python (pandas, NumPy, matplotlib) | Metrics, plots, mode-collapse detection |
| Reporting | TikZ figure export (`*.tex`) | Direct inclusion in LaTeX thesis |

---

## RL algorithms compared

Three algorithms train on the **same** observation / action interface (configs under `unity/GeantML_Test/Assets/Configs/`) and are compared over **1 million training steps**. The comparison surfaces a fundamental trade-off: standard RL **reward maximisation** vs the Monte Carlo requirement for **stochasticity preservation**.

### Algorithms

| Config | Trainer | Family | Why it's tested |
|---|---|---|---|
| `electron_ppo_v1.yaml` | **PPO** | On-policy, clipped surrogate `L^CLIP` | Stable baseline; manual entropy via β = 0.15; clipping prevents destructive policy updates |
| `electron_sac_v1.yaml` | **SAC** | Off-policy, max-entropy framework | Sample-efficient via 500 k replay buffer; *automatic* α tuning theoretically built for exploration |
| `electron_ppo_lstm_v1.yaml` | **PPO + LSTM** | On-policy + recurrent (hidden state h_t) | Carries trajectory history across steps — designed for patterns like *"after N straight steps, scatter"* |

The PPO and SAC feed-forward networks share a 3-layer × 256-unit MLP with normalised inputs; the LSTM variant adds a 128-unit memory state with sequence length 64.

### Final results (1 M steps, smoothed)

| Metric | PPO | SAC | PPO + LSTM |
|---|---:|---:|---:|
| **Training performance** | | | |
| Final cumulative reward | 5,025 | **7,161** | 5,237 |
| Final episode length (steps) | 237 | 179 | **178** |
| Steps to 90 % of max reward | 340 k | **75 k** | 125 k |
| **Stability & exploration** | | | |
| Reward variance (last 10 %) | 732 | 438 \* | 791 |
| Final policy entropy | 1.44 | −0.04 | **1.53** |
| Mode collapse? | No | **Yes (diversity)** | No |
| **Physical fidelity** | | | |
| Angular coverage | ~85 % | <20 % | **~90 %** |
| Stochasticity preserved | High | None | **Very high** |
| Physical plausibility | Moderate | Low ("laser beam") | **High** |

<sub>\* Low SAC variance reflects deterministic-policy collapse, not stable exploration.</sub>

### Verdict (from the thesis)

The "best" algorithm depends on what is asked of it:

- **SAC — the geometric optimiser.** Highest reward, ~4.5× more sample-efficient than PPO (90 % of max in 75 k vs 340 k steps), shortest paths. But the automatic entropy temperature collapsed to ≈ 0 by ~300 k steps — the policy converged to a deterministic *"laser beam"* through the phantom. Excellent at score maximisation; **fails as a physical simulator** because real electron transport requires angular diversity.

- **PPO — the physically-plausible baseline.** Stable entropy (β = 0.15 fixed), conical beam profile (~±30°), no mode collapse, ~85 % angular coverage. Slower convergence and longer trajectories (237 steps) — the agent struggled to optimise path length while preserving scattering behaviour.

- **PPO + LSTM — the recommended hybrid.** Combines SAC-level trajectory efficiency (178 steps — the shortest) with the *highest* entropy of all three (1.53) and the best angular coverage (~90 %). The recurrent hidden state lets the agent stay globally coherent (low step count) while remaining locally stochastic — exactly the regime that matches Monte Carlo behaviour.

The headline insight from the thesis: **reward-maximising ≠ physically-faithful.** SAC finds the deepest, straightest path; PPO + LSTM produces the closest match to Geant4's *"dandelion"* density profile.

Density-texture comparison plots, full hyperparameter tables, and per-metric training curves are in [`thesis.pdf`](https://github.com/LeoTheOriginal/Scalable-Pipeline-Thesis/blob/main/thesis.pdf) (Chapter 6 — *Results and Validation*).

---

## Build & run

```bash
# 1. Conda environment — Python, PyTorch, mlagents, analysis stack
conda env create -f environment.yaml
conda activate ml-agents

# 2. Geant4 — build the Water-Phantom application + the bridge DLL
cd geant4/Water-Phantom
mkdir build && cd build
cmake .. && cmake --build . --config Release
#   produces geant4_plugin.dll, picked up by Unity at runtime

# 3. Unity — open unity/GeantML_Test in Unity Hub (matching ML Agents version)
#    pick a training scene, then from a separate shell:
mlagents-learn unity/GeantML_Test/Assets/Configs/electron_ppo_v1.yaml --run-id=ppo_v1
```

Tested on **Windows**. The DLL bridge is Windows-specific in this iteration.

---

## Thesis paper

Full engineering thesis (LaTeX sources + compiled PDF):

- [**Scalable-Pipeline-Thesis**](https://github.com/LeoTheOriginal/Scalable-Pipeline-Thesis) — LaTeX sources
- [**thesis.pdf**](https://github.com/LeoTheOriginal/Scalable-Pipeline-Thesis/blob/main/thesis.pdf) — compiled PDF

---

## License & use

All rights reserved (academic work, AGH UST). For collaboration or citation, please contact the author.
