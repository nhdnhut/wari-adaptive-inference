# Pre-Registered Research Hypotheses for WARI-L

## Hypothesis 1: Bandwidth Reduction via Low-Rank Cores
Low-rank RAM-resident cores ($U_r, V_r$) reduce required DRAM read bandwidth per token proportional to the compression ratio $2r / \min(m, n)$ for all sublayers evaluated in `core` mode, mitigating memory bus saturation on commodity CPUs.

## Hypothesis 2: Temporal Working-Set Locality (Exp-0)
Autoregressive token generation exhibits temporal locality in layer demand. 
- *Target metric:* Consecutive token-window FULL-demand sets will demonstrate a mean Jaccard overlap $\ge 0.50$ on trained language models.
- *Falsification condition:* If Jaccard overlap $\approx 0$ (noise), static or session-adaptive pinning provides no benefit over random replacement.

## Hypothesis 3: Residual Persistence Non-Viability
Materializing and persisting a dense residual matrix $R_\ell = W_\ell - U_\ell V_\ell$ is strictly worse than retaining $W_\ell$. Since $\text{dim}(R_\ell) = \text{dim}(W_\ell)$, storing $R_\ell$ incurs identical I/O as loading $W_\ell$ directly, but adds the compute overhead of $U(Vx) + Rx$. Therefore, storage must only hold $W_\ell$.

## Hypothesis 4: Cost Model Predictive Validity (Exp-2)
An analytical cost model combining first-byte NVMe access latency and streaming bus bandwidth ($T_{\text{pred}}$) predicts relative wall-clock ranking across execution modes (dense, svd-only, skip-pin, mmap, wari) within a calibrated tolerance. Disagreements highlight OS page-cache buffering effects.
