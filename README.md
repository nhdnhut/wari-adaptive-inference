# WARI-L: Working-set Adaptive Residual Inference (Layer Granule)

> Hardware-aware inference scheduling policy optimizing working-set memory locality during neural network inference on CPU.

[![Python](https://img.shields.io/badge/Python-3.10+-brightgreen?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-7%20Passed-brightgreen?style=flat-square)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

**Author:** Nguyen Huynh Dang Nhut  
**Affiliation:** Faculty of Computer Science — Program in Artificial Intelligence, University of Information Technology, VNU-HCM  

> **Mandatory Framing:**  
> WARI-L is a hardware-aware inference POLICY that composes existing ideas. It is NOT a new neural architecture. If ablations show it ties skip+pin (B6), the work must be reframed as empirical systems / measurement research.

---

## 1. Overview
WARI-L investigates whether a layer-granularity inference policy can maintain dense language model quality on a CPU-only commodity PC under a hard peak-RAM budget cap by minimizing unique bytes moved per token.

### Core Architectural Invariants:
1. **Disk Storage:** Disk stores **ONLY** original layer blobs $W_\ell$ (FP16/Q4).
2. **RAM Storage:** RAM stores all low-rank cores ($U_\ell, V_\ell$), a pinned subset of $W_\ell$, KV cache, and runtime memory.
3. **No Dense Residual Matrix:** Dense residual $R = W - UV$ is **NEVER** persisted. Storing $R$ would duplicate the full layer shape on disk and degrade I/O.
4. **Sublayer Tri-State Gate:**
   - `skip`: Residual-stream identity.
   - `core`: $y = U(Vx) + b$ (RAM-resident thin GEMMs).
   - `full`: Load $W$ if unpinned; $y = Wx + b$ (never adds $UV$).
5. **Session-Adaptive Controller:** Pinned layers are dynamically managed via an Exponential Moving Average (EMA) of token-level full-layer demand under a strict byte budget.

---

## 2. Quickstart & Verification

### Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### Run Unit Tests
```bash
pytest tests/ -v
```
All 7 unit tests verify SVD reconstruction accuracy, storage constraints, gate mechanics, controller budget invariants, memory simulation formulas, and end-to-end MiniGPT integration.

### Reproduce Experiments on MiniGPT Lab
```bash
# Experiment 0: Locality and Go/No-Go Decision
python experiments/exp0_locality.py --seed 42 --output_dir results/raw/exp0

# Experiment 1: Quality Protocol Pilot (KL and Teacher Agreement)
python experiments/exp1_quality.py --seed 42 --output_dir results/raw/exp1

# Experiment 2: Systems Latency and Cost Model Validation (Seeds 0-4)
python experiments/exp2_systems.py --seeds 0 1 2 3 4 --tokens 16

# Experiment 3: Full Ablation Matrix (Seeds 0-4)
python experiments/exp3_ablation.py --seeds 0 1 2 3 4 --tokens 16

# Experiment 4: Sensitivity Sweeps (Rank, Budget, Tau)
python experiments/exp4_sensitivity.py --seed 42 --tokens 16

# Generate Figures from Raw JSON Telemetry
python experiments/plot_pilot.py
```

---

## 3. How to Run Exp-0 on a 16GB+ Laptop (Real LM Evaluation)

When deploying on a machine with $\ge 16\text{ GB}$ RAM:

1. **Install HuggingFace Transformers and Datasets:**
   ```bash
   pip install transformers datasets accelerate
   ```

2. **Run Exp-0 on TinyLlama-1.1B or Qwen2-0.5B:**
   ```bash
   python experiments/exp0_locality.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --seed 42
   ```

3. **Pre-Registered Go / No-Go Criteria:**
   - **GO (Keep Mode A - Full WARI-L):**
     There exists a rank $r$ with useful core quality ($\text{KL} \le 0.5$, $\text{Argmax} \ge 80\%$), core bytes < 50% of pageable dense bytes, and session Jaccard overlap of FULL-demand $\ge 0.50$.
   - **NO-GO (Switch to Mode B):**
     Drop residual-core. Retain `skip+pin` and systems measurement apparatus. Reframe title and paper to empirical systems research.

---

## 4. Repository Structure
```
wari/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   └── mvp.yaml
├── src/wari/
│   ├── __init__.py
│   ├── model.py          # MiniGPT laboratory model
│   ├── decompose.py      # SVD cores & storage invariants
│   ├── gates.py          # Tri-state dispatch (skip/core/full)
│   ├── controller.py     # EMA session working-set controller
│   ├── memory_sim.py     # Analytical DRAM & SSD cost model
│   ├── runtime.py        # Dispatch modes (dense, svd, wari, etc.)
│   ├── metrics.py        # RSS, Jaccard, hardware telemetry
│   └── hf_wrapper.py     # Real LM adapter & RAM safety guards
├── experiments/
│   ├── exp0_locality.py
│   ├── exp1_quality.py
│   ├── exp2_systems.py
│   ├── exp3_ablation.py
│   ├── exp4_sensitivity.py
│   └── plot_pilot.py
├── benchmarks/
│   └── prompts.jsonl
├── results/raw/          # Verifiable JSON telemetry records
├── figures/              # Generated publication figures
├── tests/
│   └── test_wari.py      # PyTest suite (7/7 passed)
├── notes/
│   ├── LITERATURE_REVIEW.md
│   ├── HYPOTHESES.md
│   ├── EXPERIMENT_LOG.md
│   └── CLAIMS.md
├── proposal/
│   └── RESEARCH_PROPOSAL.md
└── paper/
    ├── main.tex
    ├── references.bib
    ├── figures/
    ├── tables/
    └── supplementary/
```
