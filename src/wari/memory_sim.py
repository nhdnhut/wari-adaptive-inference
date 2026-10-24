"""Memory hierarchy simulator and cost model for WARI-L."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class HardwareSpec:
    """Hardware parameters for CPU memory bus and NVMe SSD storage."""
    ssd_bandwidth_bytes_per_sec: float = 2500.0 * 1024 * 1024   # 2.5 GB/s sequential NVMe read
    ssd_latency_sec: float = 20.0e-6                             # 20 us first-byte latency
    dram_bandwidth_bytes_per_sec: float = 25.0 * 1024 * 1024 * 1024  # 25 GB/s DDR4/DDR5 stream


class MemorySimulator:
    """Instruments and models DRAM and SSD byte transfers per token step.

    Cost model:
      b_t_SSD = sum_{l: g=full AND l not pinned} bytes(W_l)
      T_hat includes SSD first-byte latency + bytes/B_ssd for misses,
      and DRAM streaming of cores / pinned W.
    """

    def __init__(self, spec: HardwareSpec):
        self.spec = spec
        self.reset()

    def reset(self):
        self.dram_bytes: int = 0
        self.ssd_bytes: int = 0
        self.ssd_misses: int = 0
        self.predicted_time_sec: float = 0.0
        self.step_history: List[Dict] = []

    def record_sublayer(
        self,
        layer_name: str,
        decision: str,
        is_pinned: bool,
        core_bytes: int,
        dense_bytes: int,
    ) -> float:
        """Record I/O for a single sublayer execution and return predicted delta time."""
        delta_dram = 0
        delta_ssd = 0
        delta_miss = 0
        delta_time = 0.0

        if decision == "skip":
            # Zero weight transfers
            delta_dram = 0
            delta_ssd = 0
            delta_time = 0.0
        elif decision == "core":
            # RAM-resident thin GEMMs read low-rank core matrices from DRAM
            delta_dram = core_bytes
            delta_ssd = 0
            delta_time = core_bytes / self.spec.dram_bandwidth_bytes_per_sec
        elif decision == "full":
            if is_pinned:
                # Pinned in RAM: streamed directly from DRAM
                delta_dram = dense_bytes
                delta_ssd = 0
                delta_time = dense_bytes / self.spec.dram_bandwidth_bytes_per_sec
            else:
                # Unpinned on disk: page fault incurs SSD latency + SSD transfer + DRAM load
                delta_dram = dense_bytes
                delta_ssd = dense_bytes
                delta_miss = 1
                delta_time = (
                    self.spec.ssd_latency_sec
                    + (dense_bytes / self.spec.ssd_bandwidth_bytes_per_sec)
                    + (dense_bytes / self.spec.dram_bandwidth_bytes_per_sec)
                )

        self.dram_bytes += delta_dram
        self.ssd_bytes += delta_ssd
        self.ssd_misses += delta_miss
        self.predicted_time_sec += delta_time

        return delta_time

    def summary(self) -> dict:
        """Summary metrics of simulated memory transfers."""
        return {
            "dram_bytes": self.dram_bytes,
            "dram_mb": round(self.dram_bytes / (1024 * 1024), 3),
            "ssd_bytes": self.ssd_bytes,
            "ssd_mb": round(self.ssd_bytes / (1024 * 1024), 3),
            "ssd_misses": self.ssd_misses,
            "predicted_time_sec": round(self.predicted_time_sec, 6),
            "predicted_time_ms": round(self.predicted_time_sec * 1000.0, 3),
        }
