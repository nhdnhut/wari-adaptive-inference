"""Experiment 4: Sensitivity analysis over rank, pin budget R_max, tau thresholds, and thread counts."""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wari import (
    ExecutionMode,
    HardwareSnapshot,
    HardwareSpec,
    MemorySimulator,
    MiniGPT,
    MiniGPTConfig,
    ThresholdGate,
    WARIRuntime,
    WorkingSetController,
    factor_model,
)
from wari.hf_wrapper import HFModelAdapter


def run_exp4_sensitivity(
    seed: int = 42,
    tokens: int = 16,
    output_dir: str = "results/raw/exp4",
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    hw_snap = HardwareSnapshot.capture()

    cfg = MiniGPTConfig(vocab_size=1024, block_size=256, n_layer=8, n_head=8, n_embd=256, ffn_dim=1024)
    spec = HardwareSpec()

    torch.manual_seed(seed)
    base_model = MiniGPT(cfg)
    base_model.eval()

    prompt = torch.randint(0, cfg.vocab_size, (1, 8))

    # 1. Sensitivity over rank r
    rank_results = {}
    for r in [16, 32, 64]:
        cores, blobs = factor_model(base_model, rank=r)
        layer_sizes = {name: c.dense_bytes for name, c in cores.items()}
        controller = WorkingSetController(layer_sizes, pin_budget_bytes=2 * 1024 * 1024)
        sim = MemorySimulator(spec)
        gate = ThresholdGate(0.05, 0.25)
        runtime = WARIRuntime(ExecutionMode.WARI, cores, blobs, gate, controller, sim)

        m = MiniGPT(cfg)
        m.load_state_dict(base_model.state_dict())
        m = HFModelAdapter.apply_wari(m, runtime)
        _, perf = runtime.generate(m, prompt, max_new_tokens=tokens)
        rank_results[str(r)] = {
            "rank": r,
            "mean_frob": round(sum(c.frobenius_error for c in cores.values()) / len(cores), 4),
            "wall_time_sec": perf.wall_time_sec,
            "dram_mb": round(perf.dram_bytes / (1024 * 1024), 2),
            "ssd_mb": round(perf.ssd_bytes / (1024 * 1024), 2),
            "ssd_misses": perf.ssd_misses,
        }

    # 2. Sensitivity over pin budget R_max (MB)
    budget_results = {}
    cores32, blobs32 = factor_model(base_model, rank=32)
    layer_sizes32 = {name: c.dense_bytes for name, c in cores32.items()}
    for b_mb in [0.5, 1.0, 2.0, 4.0]:
        b_bytes = int(b_mb * 1024 * 1024)
        controller = WorkingSetController(layer_sizes32, pin_budget_bytes=b_bytes)
        sim = MemorySimulator(spec)
        gate = ThresholdGate(0.05, 0.25)
        runtime = WARIRuntime(ExecutionMode.WARI, cores32, blobs32, gate, controller, sim)

        m = MiniGPT(cfg)
        m.load_state_dict(base_model.state_dict())
        m = HFModelAdapter.apply_wari(m, runtime)
        _, perf = runtime.generate(m, prompt, max_new_tokens=tokens)
        budget_results[str(b_mb)] = {
            "budget_mb": b_mb,
            "pinned_bytes": controller.pinned_bytes,
            "ssd_misses": perf.ssd_misses,
            "ssd_mb": round(perf.ssd_bytes / (1024 * 1024), 2),
            "wall_time_sec": perf.wall_time_sec,
        }

    # 3. Sensitivity over tau thresholds
    tau_results = {}
    tau_pairs = [(0.01, 0.15), (0.05, 0.25), (0.10, 0.40)]
    for t_skip, t_core in tau_pairs:
        key = f"{t_skip}_{t_core}"
        controller = WorkingSetController(layer_sizes32, pin_budget_bytes=2 * 1024 * 1024)
        sim = MemorySimulator(spec)
        gate = ThresholdGate(t_skip, t_core)
        runtime = WARIRuntime(ExecutionMode.WARI, cores32, blobs32, gate, controller, sim)

        m = MiniGPT(cfg)
        m.load_state_dict(base_model.state_dict())
        m = HFModelAdapter.apply_wari(m, runtime)
        _, perf = runtime.generate(m, prompt, max_new_tokens=tokens)
        tau_results[key] = {
            "tau_skip": t_skip,
            "tau_core": t_core,
            "gate_histogram": perf.gate_histogram,
            "ssd_misses": perf.ssd_misses,
        }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    record = {
        "timestamp": timestamp,
        "experiment": "exp4_sensitivity",
        "model": "MiniGPT (random-init laboratory)",
        "disclaimer": "MiniGPT sensitivity pilot. Parameter sweeps for rank, budget, and thresholds.",
        "hardware": hw_snap.to_dict(),
        "seed": seed,
        "sensitivity": {
            "rank": rank_results,
            "budget_mb": budget_results,
            "tau": tau_results,
        },
    }

    out_file = Path(output_dir) / f"{timestamp}_seed{seed}.json"
    with open(out_file, "w") as f:
        json.dump(record, f, indent=2)

    print(f"[Exp-4 Done] Saved to {out_file}")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 4 Sensitivity Sweeps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="results/raw/exp4")
    args = parser.parse_args()
    run_exp4_sensitivity(seed=args.seed, tokens=args.tokens, output_dir=args.output_dir)
