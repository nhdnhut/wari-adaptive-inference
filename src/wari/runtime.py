"""Execution runtimes and sublayer dispatch for WARI-L."""

import time
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .controller import WorkingSetController
from .decompose import LayerCore
from .gates import GateDecision, OracleGate, ThresholdGate, WARIGate
from .memory_sim import HardwareSpec, MemorySimulator
from .metrics import PerformanceMetrics, compute_jaccard, get_peak_rss_mb


class ExecutionMode(str, Enum):
    DENSE = "dense"              # Baseline B4: standard dense execution
    SVD_ONLY = "svd_only"        # Baseline B5: always core, W never loaded
    SKIP_PIN = "skip_pin"        # Baseline B6: skip unpinned, full pinned (no core)
    MMAP = "mmap"                # Baseline B7: pin budget = 0, all unpinned full
    WARI = "wari"                # WARI-L: adaptive skip/core/full + session EMA
    ORACLE_WARI = "oracle_wari"  # WARI-L with oracle gate (upper bound)
    ALWAYS_FULL = "always_full"  # Baseline B8: always full with simulation


class WARISublayer(nn.Module):
    """Drop-in replacement for a pageable nn.Linear under WARI-L dispatch."""

    def __init__(
        self,
        name: str,
        core: LayerCore,
        dense_weight: torch.Tensor,
        dense_bias: Optional[torch.Tensor],
        runtime: "WARIRuntime",
    ):
        super().__init__()
        self.name = name
        self.core = core
        self.in_features = core.in_features
        self.out_features = core.out_features
        self.runtime = runtime

        # Dense weights are held as backing buffers (simulating disk/mmap backing)
        self.register_buffer("dense_weight", dense_weight)
        if dense_bias is not None:
            self.register_buffer("dense_bias", dense_bias)
        else:
            self.dense_bias = None

    def dense_forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dense_weight, self.dense_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.runtime.dispatch(self.name, x, self.core, self.dense_forward)


class WARIRuntime:
    """Orchestrates model execution under WARI-L policy or comparative baselines."""

    def __init__(
        self,
        mode: ExecutionMode,
        cores: Dict[str, LayerCore],
        dense_blobs: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]],
        gate: WARIGate,
        controller: WorkingSetController,
        memory_sim: MemorySimulator,
    ):
        self.mode = mode
        self.cores = cores
        self.dense_blobs = dense_blobs
        self.gate = gate
        self.controller = controller
        self.memory_sim = memory_sim

        self.gate_histogram = {"skip": 0, "core": 0, "full": 0}
        self.current_step_decisions: Dict[str, str] = {}
        self.window_pinned_history: List[set] = []

    def reset_telemetry(self):
        self.gate_histogram = {"skip": 0, "core": 0, "full": 0}
        self.memory_sim.reset()
        self.current_step_decisions = {}
        self.window_pinned_history = []

    def dispatch(
        self,
        layer_name: str,
        x: torch.Tensor,
        core: LayerCore,
        dense_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Route sublayer computation according to the active execution policy."""
        core_bytes = core.core_bytes
        dense_bytes = core.dense_bytes
        is_pinned = self.controller.is_pinned(layer_name)

        if self.mode == ExecutionMode.DENSE:
            self.memory_sim.record_sublayer(layer_name, "full", True, core_bytes, dense_bytes)
            self.gate_histogram["full"] += 1
            return dense_fn(x)

        elif self.mode == ExecutionMode.SVD_ONLY:
            self.memory_sim.record_sublayer(layer_name, "core", False, core_bytes, dense_bytes)
            self.gate_histogram["core"] += 1
            return core(x)

        elif self.mode == ExecutionMode.MMAP:
            # All layers unpinned, loaded on demand as full
            self.memory_sim.record_sublayer(layer_name, "full", False, core_bytes, dense_bytes)
            self.gate_histogram["full"] += 1
            return dense_fn(x)

        elif self.mode == ExecutionMode.SKIP_PIN:
            if is_pinned:
                self.memory_sim.record_sublayer(layer_name, "full", True, core_bytes, dense_bytes)
                self.gate_histogram["full"] += 1
                decision = "full"
                out = dense_fn(x)
            else:
                self.memory_sim.record_sublayer(layer_name, "skip", False, core_bytes, dense_bytes)
                self.gate_histogram["skip"] += 1
                decision = "skip"
                out = torch.zeros(x.shape[:-1] + (core.out_features,), dtype=x.dtype, device=x.device)
            self.current_step_decisions[layer_name] = decision
            return out

        elif self.mode in (ExecutionMode.WARI, ExecutionMode.ORACLE_WARI):
            decision, _ = self.gate.decide(layer_name, x, core, dense_fn)
            dec_str = decision.value
            self.gate_histogram[dec_str] += 1
            self.current_step_decisions[layer_name] = dec_str

            self.memory_sim.record_sublayer(layer_name, dec_str, is_pinned, core_bytes, dense_bytes)

            if decision == GateDecision.SKIP:
                return torch.zeros(x.shape[:-1] + (core.out_features,), dtype=x.dtype, device=x.device)
            elif decision == GateDecision.CORE:
                return core(x)
            else:  # FULL
                # Load W if unpinned; y = Wx + b (NEVER add UV)
                return dense_fn(x)

        elif self.mode == ExecutionMode.ALWAYS_FULL:
            self.memory_sim.record_sublayer(layer_name, "full", is_pinned, core_bytes, dense_bytes)
            self.gate_histogram["full"] += 1
            return dense_fn(x)

        else:
            raise ValueError(f"Unknown execution mode: {self.mode}")

    def on_token_end(self):
        """Called at the conclusion of each decoded token to update controller state."""
        if self.mode in (ExecutionMode.WARI, ExecutionMode.ORACLE_WARI, ExecutionMode.SKIP_PIN):
            self.controller.update_demand(self.current_step_decisions)
            self.window_pinned_history.append(set(self.controller.pinned_layers))
            self.current_step_decisions = {}

    def generate(
        self,
        model: nn.Module,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
    ) -> Tuple[torch.Tensor, PerformanceMetrics]:
        """Autoregressive generation loop with latency and I/O tracking."""
        self.reset_telemetry()
        model.eval()

        device = prompt_tokens.device
        curr_tokens = prompt_tokens.clone()

        t0 = time.perf_counter()
        ttft = 0.0

        with torch.no_grad():
            # Prefill phase
            logits, _ = model(curr_tokens)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            curr_tokens = torch.cat([curr_tokens, next_token], dim=-1)
            self.on_token_end()
            ttft = time.perf_counter() - t0

            # Decode phase
            decode_start = time.perf_counter()
            for _ in range(max_new_tokens - 1):
                # Only feed context up to block_size
                ctx = curr_tokens[:, -model.config.block_size:] if hasattr(model, "config") else curr_tokens
                logits, _ = model(ctx)
                if temperature == 0.0:
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                else:
                    probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                curr_tokens = torch.cat([curr_tokens, next_token], dim=-1)
                self.on_token_end()

        total_wall = time.perf_counter() - t0
        decode_wall = time.perf_counter() - decode_start if max_new_tokens > 1 else 0.0
        tpot = decode_wall / (max_new_tokens - 1) if max_new_tokens > 1 else ttft

        # Compute pin-set Jaccard across consecutive windows if available
        jaccard_val = None
        if len(self.window_pinned_history) >= 2:
            mid = len(self.window_pinned_history) // 2
            w1 = set().union(*self.window_pinned_history[:mid])
            w2 = set().union(*self.window_pinned_history[mid:])
            jaccard_val = compute_jaccard(w1, w2)

        mem_summary = self.memory_sim.summary()
        metrics = PerformanceMetrics(
            wall_time_sec=round(total_wall, 4),
            tokens_generated=max_new_tokens,
            tok_per_sec=round(max_new_tokens / (total_wall + 1e-12), 2),
            ttft_sec=round(ttft, 4),
            tpot_sec=round(tpot, 4),
            peak_rss_mb=get_peak_rss_mb(),
            dram_bytes=mem_summary["dram_bytes"],
            ssd_bytes=mem_summary["ssd_bytes"],
            ssd_misses=mem_summary["ssd_misses"],
            predicted_time_sec=mem_summary["predicted_time_sec"],
            gate_histogram=self.gate_histogram.copy(),
            pin_set_jaccard=jaccard_val,
        )

        return curr_tokens, metrics
