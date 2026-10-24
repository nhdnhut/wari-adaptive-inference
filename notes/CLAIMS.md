# Traceable Scientific and Empirical Claims

Every non-obvious claim is tagged with either `[cite: KEY]` (from `paper/references.bib`) or `[exp: PATH]` (from `results/raw/`). Any quantity not backed by an experimental run or verified literature is explicitly marked `TBD`.

---

## 1. Literature and Systems Background Claims
1. On commodity CPUs, single-batch autoregressive transformer decoding is memory-bandwidth bound rather than compute bound [cite: williams2009roofline].
2. Offloading transformer weights to NVMe flash can permit inference on memory-constrained systems, but requires minimizing volume transferred per token [cite: alizadeh2024llminflash].
3. Fine-grained neuron-level sparse CPU GEMV fails to outperform dense BLAS until dynamic activation sparsity exceeds approximately $87\%$, whereas SwiGLU architectures exhibit dynamic sparsity of only $40\%-50\%$ [cite: song2024powerinfer].
4. Activation-whitened SVD can approximate transformer projection matrices with significantly lower perplexity degradation than unweighted truncation [cite: wang2025svdllm].

---

## 2. Experimental Claims (Directly Derived from results/raw/)

### Experiment 0: Locality and Mechanism Go/No-Go
1. On the random-init MiniGPT mechanism laboratory, rank-16 truncation incurs a mean relative Frobenius error of $0.9038$ [exp: results/raw/exp0/20260916_180355Z_seed42.json].
2. Low-rank core matrices at rank-16 occupy $20.00\%$ ($0.2000$) of pageable dense weight bytes [exp: results/raw/exp0/20260916_180355Z_seed42.json].
3. For random Gaussian matrices, next-token KL divergence between dense and rank-16 always-core models is $7.0179$, and greedy argmax agreement is $0.0078$ [exp: results/raw/exp0/20260916_180355Z_seed42.json].
4. The windowed FULL-demand layer sets under oracle gating yield a mean Jaccard overlap of $0.0408$ on MiniGPT [exp: results/raw/exp0/20260916_180355Z_seed42.json].
5. Pre-registered Go/No-Go on MiniGPT resulted in `exp0_go_on_minigpt = false` [exp: results/raw/exp0/20260916_180355Z_seed42.json]. Evaluation on real trained small language models (TinyLlama-1.1B, Qwen2-0.5B, GPT-2) is marked TBD pending execution on a host meeting the $\ge 16\text{ GB}$ physical RAM requirement.

### Experiment 1: Quality Pilots
1. On MiniGPT, WARI-L student achieves a KL divergence of $4.9349$ and greedy token agreement of $0.0352$ relative to the dense teacher [exp: results/raw/exp1/20260916_180500Z_seed42.json].
2. WikiText-2 word-level perplexity for TinyLlama-1.1B under WARI-L is TBD.
3. LAMBADA zero-shot accuracy for TinyLlama-1.1B under WARI-L is TBD.

### Experiment 2: Systems and Cost Model Parity
1. Analytical I/O latency predictions ($T_{\text{pred}}$) track relative I/O byte volumes across modes [exp: results/raw/exp2/20260916_180627Z_seed0.json].
2. Measured wall-clock time reflects thread dispatch and runtime overhead in addition to modeled raw transfer times [exp: results/raw/exp2/20260916_180627Z_seed0.json].

### Experiment 3: Ablation Matrix
1. Pinned working set controller strictly enforces the peak memory cap across all evaluation seeds $0\dots 4$, never exceeding the allocated budget [exp: results/raw/exp3/20260916_180806Z_seed0.json].
2. In the MiniGPT mechanism test, B5 (SVD-only) transfers only DRAM core bytes ($2.46\text{ MB}$), while B7 (MMAP unpinned) incurs SSD reads for every layer [exp: results/raw/exp3/20260916_180806Z_seed0.json].
3. WARI-L with adaptive gating balances low-rank RAM streaming against selective dense layer paging [exp: results/raw/exp3/20260916_180806Z_seed0.json].
