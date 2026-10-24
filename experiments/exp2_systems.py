"""Experiment 2: Systems performance, I/O cost model validation, and latency tracking."""

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


def run_exp2_systems(
    seeds=(0, 1, 2, 3, 4),
    seq_len: int = 128,
    num_tokens: int = 32,
    output_dir: str = "results/raw/exp2",
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    hw_snap = HardwareSnapshot.capture()

    cfg = MiniGPTConfig(vocab_size=1024, block_size=256, n_layer=8, n_head=8, n_embd=256, ffn_dim=1024)

    all_records = []

    for seed in seeds:
        torch.manual_seed(seed)
        model = MiniGPT(cfg)
        model.eval()

        cores, blobs = factor_model(model, rank=32)
        layer_sizes = {name: c.dense_bytes for name, c in cores.items()}

        spec = HardwareSpec(
            ssd_bandwidth_bytes_per_sec=2500.0 * 1024 * 1024,
            ssd_latency_sec=20e-6,
            dram_bandwidth_bytes_per_sec=25.0 * 1024 * 1024 * 1024,
        )

        modes = [
            ExecutionMode.DENSE,
            ExecutionMode.SVD_ONLY,
            ExecutionMode.SKIP_PIN,
            ExecutionMode.MMAP,
            ExecutionMode.WARI,
        ]

        mode_results = {}

        for mode in modes:
            # 2MB pin budget for WARI and SKIP_PIN; 0MB for MMAP
            pin_budget = 2 * 1024 * 1024 if mode in (ExecutionMode.WARI, ExecutionMode.SKIP_PIN) else 0
            controller = WorkingSetController(layer_sizes, pin_budget_bytes=pin_budget, beta=0.8)
            sim = MemorySimulator(spec)
            gate = ThresholdGate(tau_skip=0.05, tau_core=0.25)

            runtime = WARIRuntime(
                mode=mode,
                cores=cores,
                dense_blobs=blobs,
                gate=gate,
                controller=controller,
                memory_sim=sim,
            )

            test_model = MiniGPT(cfg)
            test_model.load_state_dict(model.state_dict())
            test_model = HFModelAdapter.apply_wari(test_model, runtime)

            prompt = torch.randint(0, cfg.vocab_size, (1, 16))

            # Cold cache pass
            runtime.reset_telemetry()
            _, metrics_cold = runtime.generate(test_model, prompt, max_new_tokens=num_tokens)

            # Warm cache pass
            runtime.reset_telemetry()
            _, metrics_warm = runtime.generate(test_model, prompt, max_new_tokens=num_tokens)

            mode_results[mode.value] = {
                "cold": metrics_cold.to_dict(),
                "warm": metrics_warm.to_dict(),
                "disagreement_pct": round(
                    abs(metrics_cold.wall_time_sec - metrics_cold.predicted_time_sec)
                    / (metrics_cold.wall_time_sec + 1e-8)
                    * 100.0,
                    2,
                ),
            }

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        record = {
            "timestamp": timestamp,
            "experiment": "exp2_systems",
            "model": "MiniGPT (random-init laboratory)",
            "disclaimer": "MiniGPT mechanism pilot. Evaluates I/O cost model and runtime behavior.",
            "hardware": hw_snap.to_dict(),
            "seed": seed,
            "modes": mode_results,
        }

        out_file = Path(output_dir) / f"{timestamp}_seed{seed}.json"
        with open(out_file, "w") as f:
            json.dump(record, f, indent=2)

        print(f"[Exp-2 Done] Seed {seed} saved to {out_file}")
        all_records.append(record)

    return all_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 2 Systems Benchmarks")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="results/raw/exp2")
    args = parser.parse_args()
    run_exp2_systems(seeds=args.seeds, num_tokens=args.tokens, output_dir=args.output_dir)
