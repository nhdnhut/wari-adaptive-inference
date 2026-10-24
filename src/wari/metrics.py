"""Hardware telemetry and experiment performance metrics."""

import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Set

import psutil
import torch


@dataclass
class HardwareSnapshot:
    """Snapshot of execution environment and hardware specifications."""
    os: str
    os_release: str
    architecture: str
    python_version: str
    torch_version: str
    cpu_count_logical: int
    cpu_count_physical: Optional[int]
    total_ram_bytes: int
    total_ram_gb: float
    num_threads_torch: int

    @classmethod
    def capture(cls) -> "HardwareSnapshot":
        vm = psutil.virtual_memory()
        return cls(
            os=platform.system(),
            os_release=platform.release(),
            architecture=platform.machine(),
            python_version=sys.version.split()[0],
            torch_version=torch.__version__,
            cpu_count_logical=os.cpu_count() or 1,
            cpu_count_physical=psutil.cpu_count(logical=False) or 1,
            total_ram_bytes=vm.total,
            total_ram_gb=round(vm.total / (1024**3), 2),
            num_threads_torch=torch.get_num_threads(),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def get_peak_rss_mb() -> float:
    """Current process peak Resident Set Size (RSS) in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    # On Windows, peak_wset or rss
    rss = getattr(mem_info, "peak_wset", mem_info.rss)
    return round(rss / (1024 * 1024), 2)


def compute_jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two sets of layer names."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    if union == 0:
        return 0.0
    return round(intersection / union, 4)


@dataclass
class PerformanceMetrics:
    """Execution telemetry for a benchmark or generation run."""
    wall_time_sec: float
    tokens_generated: int
    tok_per_sec: float
    ttft_sec: float
    tpot_sec: float
    peak_rss_mb: float
    dram_bytes: int
    ssd_bytes: int
    ssd_misses: int
    predicted_time_sec: float
    gate_histogram: Dict[str, int]
    pin_set_jaccard: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)
