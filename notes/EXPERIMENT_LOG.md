# Experiment Log: WARI-L Empirical Execution

## Go / No-Go Decision Record (Experiment 0)
On the MiniGPT mechanism laboratory (random initialization, 8 layers, hidden dim 256, FFN dim 1024), evaluation across ranks {16, 32, 64} yielded a rank-16 mean relative Frobenius error of $0.9038$, core-to-dense byte ratio of $0.2000$, next-token KL divergence of $7.0179$, greedy argmax agreement of $0.0078$, and a mean windowed session Jaccard overlap of $0.0408$ [exp: results/raw/exp0/20260916_180355Z_seed42.json]. In accordance with pre-registered criteria ($\text{Rel-Frob} \le 0.50$ and $\text{Jaccard} \ge 0.50$), `exp0_go_on_minigpt = false`. This behavior is mathematically expected for Gaussian random weight matrices whose singular value spectrum decays sublinearly. Because the active execution host possesses $< 8\text{ GB}$ physical RAM ($7.70\text{ GB}$ total, $\approx 248\text{ MB}$ free at run time), attempting to download and evaluate TinyLlama-1.1B or Qwen2-0.5B would violate memory safety constraints and induce severe operating system thrashing; real model evaluation is strictly deferred to a $16\text{GB}+$ host as specified in Section 5.

---

## Logged Runs

### Experiment 0: Locality and Go/No-Go Pilot
- **File:** [results/raw/exp0/20260916_180355Z_seed42.json](file:///c:/Users/nguye/Desktop/wari/results/raw/exp0/20260916_180355Z_seed42.json)
- **Model:** MiniGPT (random-init laboratory)
- **Rank 16:** Rel-Frobenius = 0.9038, Byte Ratio = 0.2000, KL = 7.0179, Argmax Agree = 0.0078, Jaccard = 0.0408
- **Rank 32:** Rel-Frobenius = 0.8123, Byte Ratio = 0.4000, KL = 6.9530, Argmax Agree = 0.0078, Jaccard = 0.0526
- **Rank 64:** Rel-Frobenius = 0.6393, Byte Ratio = 0.8000, KL = 6.6438, Argmax Agree = 0.0391, Jaccard = 0.0526
- **Status:** Complete.

### Experiment 1: Quality Protocol Pilot
- **File:** [results/raw/exp1/20260916_180500Z_seed42.json](file:///c:/Users/nguye/Desktop/wari/results/raw/exp1/20260916_180500Z_seed42.json)
- **Model:** MiniGPT (random-init laboratory)
- **Results:** KL Divergence vs Dense Teacher = 4.9349, Greedy Argmax Agreement = 0.0352.
- **Real LM (WikiText-2 / LAMBADA):** Marked TBD (requires 16GB+ host).

### Experiment 2: Systems Latency and Cost Model Validation
- **Files:**
  - Seed 0: [results/raw/exp2/20260916_180627Z_seed0.json](file:///c:/Users/nguye/Desktop/wari/results/raw/exp2/20260916_180627Z_seed0.json)
  - Seed 1: [results/raw/exp2/20260916_180629Z_seed1.json](file:///c:/Users/nguye/Desktop/wari/results/raw/exp2/20260916_180629Z_seed1.json)
  - Seed 2: [results/raw/exp2/20260916_180632Z_seed2.json](file:///c:/Users/nguye/Desktop/wari/results/raw/exp2/20260916_180632Z_seed2.json)
  - Seed 3: [results/raw/exp2/20260916_180634Z_seed3.json](file:///c:/Users/nguye/Desktop/wari/results/raw/exp2/20260916_180634Z_seed3.json)
  - Seed 4: [results/raw/exp2/20260916_180636Z_seed4.json](file:///c:/Users/nguye/Desktop/wari/results/raw/exp2/20260916_180636Z_seed4.json)
- **Observations:** Mode comparison (dense, svd_only, skip_pin, mmap, wari) cold and warm passes executed. Measured wall-clock time aligns with analytical predictions while reflecting CPU threading runtime overheads.

### Experiment 3: Full Ablation Matrix
- **Files:**
  - Seed 0: [results/raw/exp3/20260916_180806Z_seed0.json](file:///c:/Users/nguye/Desktop/wari/results/raw/exp3/20260916_180806Z_seed0.json)
  - Seed 1: [results/raw/exp3/20260916_180815Z_seed1.json](file:///c:/Users/nguye/Desktop/wari/results/raw/exp3/20260916_180815Z_seed1.json)
  - Seed 2: [results/raw/exp3/20260916_180826Z_seed2.json](file:///c:/Users/nguye/Desktop/wari/results/raw/exp3/20260916_180826Z_seed2.json)
  - Seed 3: [results/raw/exp3/20260916_180834Z_seed3.json](file:///c:/Users/nguye/Desktop/wari/results/raw/exp3/20260916_180834Z_seed3.json)
  - Seed 4: [results/raw/exp3/20260916_180842Z_seed4.json](file:///c:/Users/nguye/Desktop/wari/results/raw/exp3/20260916_180842Z_seed4.json)
- **Variants Evaluated:** B4_dense, B5_svd_only, B6_skip_pin, B7_mmap, WARI_no_skip, WARI_no_core, WARI_full.

### Experiment 4: Sensitivity Sweeps
- **File:** [results/raw/exp4/20260916_181007Z_seed42.json](file:///c:/Users/nguye/Desktop/wari/results/raw/exp4/20260916_181007Z_seed42.json)
- **Sweeps:** Rank $r \in \{16, 32, 64\}$, Pin Budget $R_{\max} \in \{0.5, 1.0, 2.0, 4.0\}\text{ MB}$, Thresholds $(\tau_{\text{skip}}, \tau_{\text{core}})$.
