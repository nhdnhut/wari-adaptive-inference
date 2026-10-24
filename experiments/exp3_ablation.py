"""Experiment 3: Full ablation matrix across baselines and WARI-L variants."""

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


def run_exp3_ablation(
    seeds=(0, 1, 2, 3, 4),
    seq_len: int = 128,
    warmup_tokens: int = 4,
    eval_tokens: int = 16,
    output_dir: str = "results/raw/exp3",
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    hw_snap = HardwareSnapshot.capture()

    cfg = MiniGPTConfig(vocab_size=1024, block_size=256, n_layer=8, n_head=8, n_embd=256, ffn_dim=1024)

    all_records = []

    for seed in seeds:
        torch.manual_seed(seed)
        base_model = MiniGPT(cfg)
        base_model.eval()

        cores, blobs = factor_model(base_model, rank=32)
        layer_sizes = {name: c.dense_bytes for name, c in cores.items()}
        spec = HardwareSpec()

        variants = [
            ("B4_dense", ExecutionMode.DENSE, 0.05, 0.25, 0),
            ("B5_svd_only", ExecutionMode.SVD_ONLY, 0.05, 0.25, 0),
            ("B6_skip_pin", ExecutionMode.SKIP_PIN, 0.05, 0.25, 2 * 1024 * 1024),
            ("B7_mmap", ExecutionMode.MMAP, 0.05, 0.25, 0),
            ("WARI_no_skip", ExecutionMode.WARI, 0.0, 0.25, 2 * 1024 * 1024),
            ("WARI_no_core", ExecutionMode.WARI, 0.05, 0.05, 2 * 1024 * 1024),
            ("WARI_full", ExecutionMode.WARI, 0.05, 0.25, 2 * 1024 * 1024),
        ]

        variant_results = {}

        for var_name, mode, t_skip, t_core, pin_budget in variants:
            controller = WorkingSetController(layer_sizes, pin_budget_bytes=pin_budget, beta=0.8)
            sim = MemorySimulator(spec)
            gate = ThresholdGate(tau_skip=t_skip, tau_core=t_core)

            runtime = WARIRuntime(
                mode=mode,
                cores=cores,
                dense_blobs=blobs,
                gate=gate,
                controller=controller,
                memory_sim=sim,
            )

            test_model = MiniGPT(cfg)
            test_model.load_state_dict(base_model.state_dict())
            test_model = HFModelAdapter.apply_wari(test_model, runtime)

            prompt = torch.randint(0, cfg.vocab_size, (1, 8))

            # Warmup pass (discarded)
            runtime.reset_telemetry()
            _ = runtime.generate(test_model, prompt, max_new_tokens=warmup_tokens)

            # Measurement pass
            runtime.reset_telemetry()
            _, metrics = runtime.generate(test_model, prompt, max_new_tokens=eval_tokens)

            variant_results[var_name] = {
                "mode": mode.value,
                "tau_skip": t_skip,
                "tau_core": t_core,
                "pin_budget_bytes": pin_budget,
                "systems": metrics.to_dict(),
                "controller_snapshot": controller.snapshot(),
            }

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        record = {
            "timestamp": timestamp,
            "experiment": "exp3_ablation",
            "model": "MiniGPT (random-init laboratory)",
            "disclaimer": "MiniGPT mechanism pilot. Evaluates ablation matrix across baselines and WARI variants.",
            "hardware": hw_snap.to_dict(),
            "seed": seed,
            "variants": variant_results,
        }

        out_file = Path(output_dir) / f"{timestamp}_seed{seed}.json"
        with open(out_file, "w") as f:
            json.dump(record, f, indent=2)

        print(f"[Exp-3 Done] Seed {seed} saved to {out_file}")
        all_records.append(record)

    return all_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 3 Ablation Matrix")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="results/raw/exp3")
    args = parser.parse_args()
    run_exp3_ablation(seeds=args.seeds, eval_tokens=args.tokens, output_dir=args.output_dir)
