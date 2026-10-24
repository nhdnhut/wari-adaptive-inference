# Ablation Matrix Table (Template & Pilot Data)

| Policy / Baseline | Mode | Description | WikiText-2 PPL | Agree % | TPOT (ms) | Peak RSS (MB) | DRAM (MB) | SSD (MB) |
|---|---|---|---|---|---|---|---|---|
| **B4: Dense PyTorch** | `dense` | Standard dense forward | TBD* | 100.0% | 23.4 ± 1.2 | 84.2 | 12.3 | 0.0 |
| **B5: SVD-Only** | `svd_only` | Low-rank cores only ($r=32$) | TBD* | 1.1% | 14.8 ± 0.8 | 65.1 | 2.5 | 0.0 |
| **B6: Skip + Pin** | `skip_pin` | Pinned full, unpinned skip | TBD* | 4.2% | 18.2 ± 1.1 | 72.0 | 4.1 | 0.0 |
| **B7: MMAP All** | `mmap` | 0MB pin budget, demand page | TBD* | 100.0% | 45.6 ± 3.4 | 55.4 | 12.3 | 12.3 |
| **WARI-L (No-Skip)** | `wari` | Gate: Core or Full ($\tau_{\text{skip}}=0$) | TBD* | 3.5% | 28.1 ± 1.8 | 74.5 | 6.8 | 4.9 |
| **WARI-L (No-Core)** | `wari` | Gate: Skip or Full ($\tau_{\text{core}}=\tau_{\text{skip}}$) | TBD* | 4.0% | 21.3 ± 1.4 | 73.8 | 5.2 | 3.1 |
| **WARI-L (Full)** | `wari` | Adaptive Skip / Core / Full | TBD* | 3.8% | 24.5 ± 1.5 | 74.2 | 5.9 | 3.7 |

*\*Note: Language modeling perplexity numbers for real models (TinyLlama-1.1B) are marked TBD pending execution on 16GB+ hardware. Numbers shown reflect MiniGPT mechanism pilot runs across seeds 0–4 [exp: results/raw/exp3/].*
