"""WARI-L: Working-set Adaptive Residual Inference (Layer Granule).

A hardware-aware inference policy composing low-rank RAM resident cores,
residual pageable layers, and session-adaptive working set management.
"""

from .model import MiniGPT, MiniGPTConfig
from .decompose import factor_model, factor_linear, LayerCore
from .gates import WARIGate, OracleGate, ThresholdGate, GateDecision
from .controller import WorkingSetController
from .memory_sim import MemorySimulator, HardwareSpec
from .runtime import WARIRuntime, ExecutionMode
from .metrics import HardwareSnapshot, PerformanceMetrics, compute_jaccard

__all__ = [
    "MiniGPT",
    "MiniGPTConfig",
    "factor_model",
    "factor_linear",
    "LayerCore",
    "WARIGate",
    "OracleGate",
    "ThresholdGate",
    "GateDecision",
    "WorkingSetController",
    "MemorySimulator",
    "HardwareSpec",
    "WARIRuntime",
    "ExecutionMode",
    "HardwareSnapshot",
    "PerformanceMetrics",
    "compute_jaccard",
]
