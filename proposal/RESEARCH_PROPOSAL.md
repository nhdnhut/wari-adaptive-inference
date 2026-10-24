# Research Proposal: WARI-L

## 1. Project Title
**Working-Set Adaptive Residual Inference: Memory-Budgeted CPU Decoding for Dense LLMs on Commodity PCs**

**Author:** Nguyen Huynh Dang Nhut  
**Affiliation:** Faculty of Computer Science — Program in Artificial Intelligence, University of Information Technology, VNU-HCM  

## 2. Core Research Question
Can a layer-granularity inference policy maintain useful dense language model quality on a CPU-only commodity PC under a hard peak-RAM budget by minimizing unique bytes transferred per accepted token?

## 3. Joint Optimization Objective
$$\mathcal{O} = \frac{\text{Quality}}{\text{PeakRAM} \times \text{Latency}}$$

Where:
- **Quality** is quantified by WikiText-2 perplexity, LAMBADA accuracy, and teacher greedy-token agreement.
- **PeakRAM** is strictly constrained by a hardware budget ($R_{\max}$).
- **Latency** is measured in wall-clock seconds per token (TPOT) and Time to First Token (TTFT).

## 4. Method Framing & Scientific Honesty
WARI-L is a hardware-aware inference POLICY that composes existing ideas. It is NOT a new neural architecture. If empirical ablations demonstrate that WARI-L fails to outperform naive skip-and-pin (B6), the project will be reframed as an empirical systems measurement study.

## 5. Architectural and Storage Constraints
1. **No Residual Blobs on Disk:** Disk stores only original weights $W_\ell$. Dense residual matrices $R = W - UV$ are never persisted because $\text{dim}(R) = \text{dim}(W)$ and loading $R$ would duplicate I/O.
2. **Layer Granularity:** Granularity is fixed at the sublayer level (attention and MLP projections) because sparse CPU GEMV loses to dense BLAS at low dynamic sparsity ($<87\%$) [cite: song2024powerinfer].
3. **Session-Adaptive Working Set:** Pinned RAM residency is governed by an exponential moving average (EMA) of token-level full-layer demand rather than corpus-static statistics.
