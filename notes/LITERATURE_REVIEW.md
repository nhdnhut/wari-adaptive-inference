# Literature Review: Memory-Constrained CPU LLM Inference and WARI-L

## 1. Hardware Bottlenecks and the Roofline Model
Under the Roofline Model [cite: williams2009roofline], autoregressive token decoding ($B=1$) is strictly memory-bandwidth bound. On commodity CPU systems lacking high-bandwidth unified memory, decoding throughput is dictated by:
$$\text{Latency} \approx \frac{\text{Active Model Bytes Moved}}{\text{Memory Bandwidth (GB/s)}} + \text{Storage I/O Latency}$$

## 2. Weight Offloading and Paging
- **FlexGen** [cite: sheng2023flexgen]: High-throughput throughput-oriented offloading across GPU, CPU DRAM, and SSD. Optimizes throughput via batching and zig-zag block scheduling, but incurs high latency for single-batch conversational decoding.
- **LLM in a Flash** [cite: alizadeh2024llminflash]: Explores dynamic activation-induced sparsity in feed-forward networks to transfer only activated neuron rows from flash memory. 
- **PowerInfer** [cite: song2024powerinfer]: Investigates neuron-level sparsity. Crucially, Song et al. demonstrate that traditional sparse CPU GEMV operations lose to dense BLAS until dynamic activation sparsity exceeds $\approx 87\%$. Modern models utilizing SwiGLU architectures exhibit dynamic sparsity of only $40\%-50\%$, and irregular fine-grained sparse reads fight NVMe page-alignment. This provides the empirical foundation for **WARI-L's choice of layer-level granularity** over neuron-level slicing.
- **PowerInfer-2** [cite: xue2024powerinfer2]: Adapts tiered serving to smartphone memory constraints.
- **HELIOS** [cite: kumar2025helios]: Hardware-efficient low-rank inference offloading subsystem for large language models.
- **Zhang & Huang** [cite: zhang2025challenging]: Analyzes commodity CPU execution dynamics and memory system trade-offs.
- **Comprehensive Surveys**: Wan et al. [cite: wan2024efficient], Xu et al. [cite: xu2024ondevice], Yuan et al. [cite: yuan2024unveiled], and Chen et al. [cite: chen2026ondevice].

## 3. Dynamic Early Exit, Skipping, and Speculation
- **LayerSkip** [cite: elhoushi2024layerskip]: Employs early-exit classifiers and speculative verification across transformer layers.
- **Mixture-of-Depths** [cite: raposo2024mod]: Explores conditional computation routing tokens past specific transformer blocks.
- **FlexEE** [cite: xie2026flexee]: Flexible early exit with residual compensation mechanisms.
- **Speculative Decoding** [cite: leviathan2023speculative] and **Medusa** [cite: cai2024medusa]: Amortizes memory bandwidth bottlenecks by verifying multiple tokens per weight pass.

## 4. Low-Rank Factorization for LLMs
- **SVD-LLM** [cite: wang2025svdllm]: Demonstrates that activation-whitened SVD decomposition can compress transformer projection matrices with significantly lower perplexity degradation than naive SVD.
- **ASVD** [cite: yuan2023asvd]: Activation-aware SVD scaling matrices by channel-wise activation scales.

## 5. Synthesis: Where WARI-L Fits
WARI-L is a hardware-aware inference POLICY that composes existing ideas. It is NOT a new neural architecture. It integrates:
1. Low-rank RAM-resident cores ($U_r, V_r$) that absorb baseline representations without requiring disk access.
2. Demand-paged original dense layer blobs $W_\ell$ (never materializing dense residuals $R = W - UV$).
3. A session-adaptive working set controller with exponential moving average (EMA) tracking to dynamically pin high-demand layers under a hard peak-RAM cap.
