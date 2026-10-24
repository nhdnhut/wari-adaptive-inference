"""Experiment 0: Locality, SVD Spectral Decay, and Go/No-Go Decision."""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wari import (
    ExecutionMode,
    HardwareSnapshot,
    HardwareSpec,
    MemorySimulator,
    MiniGPT,
    MiniGPTConfig,
    OracleGate,
    ThresholdGate,
    WARIRuntime,
    WorkingSetController,
    compute_jaccard,
    factor_model,
)
from wari.hf_wrapper import HFModelAdapter, check_ram_safety


def run_exp0(
    model_type: str = "minigpt",
    ranks=(16, 32, 64),
    seed: int = 42,
    seq_len: int = 128,
    num_windows: int = 4,
    output_dir: str = "results/raw/exp0",
):
    torch.manual_seed(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    hw_snap = HardwareSnapshot.capture()

    if model_type != "minigpt":
        # Check RAM safety before attempting HuggingFace download/load
        check_ram_safety(model_type, required_ram_gb=10.0)
        raise NotImplementedError("HuggingFace real LM evaluation requires 16GB+ RAM host as documented in README.")

    # MiniGPT mechanism laboratory
    cfg = MiniGPTConfig(
        vocab_size=1024,
        block_size=256,
        n_layer=8,
        n_head=8,
        n_embd=256,
        ffn_dim=1024,
    )
    base_model = MiniGPT(cfg)
    base_model.eval()

    results_by_rank = {}

    for r in ranks:
        cores, blobs = factor_model(base_model, rank=r)
        mean_frob = sum(c.frobenius_error for c in cores.values()) / len(cores)
        total_core_bytes = sum(c.core_bytes for c in cores.values())
        total_dense_bytes = sum(c.dense_bytes for c in cores.values())
        byte_ratio = total_core_bytes / (total_dense_bytes + 1e-8)

        # Synthetic calibration tokens
        tokens = torch.randint(0, cfg.vocab_size, (1, seq_len))
        with torch.no_grad():
            dense_logits, _ = base_model(tokens)

        # SVD-only forward for KL and argmax agreement
        layer_sizes = {name: c.dense_bytes for name, c in cores.items()}
        controller = WorkingSetController(layer_sizes, pin_budget_bytes=10 * 1024 * 1024)
        sim = MemorySimulator(HardwareSpec())
        svd_runtime = WARIRuntime(
            mode=ExecutionMode.SVD_ONLY,
            cores=cores,
            dense_blobs=blobs,
            gate=ThresholdGate(),
            controller=controller,
            memory_sim=sim,
        )

        svd_model = MiniGPT(cfg)
        svd_model.load_state_dict(base_model.state_dict())
        svd_model = HFModelAdapter.apply_wari(svd_model, svd_runtime)
        with torch.no_grad():
            svd_logits, _ = svd_model(tokens)

        # Compute next-token KL divergence and greedy argmax agreement
        p_dense = F.softmax(dense_logits, dim=-1)
        p_svd = F.log_softmax(svd_logits, dim=-1)
        kl_div = F.kl_div(p_svd, p_dense, reduction="batchmean").item()

        argmax_dense = torch.argmax(dense_logits, dim=-1)
        argmax_svd = torch.argmax(svd_logits, dim=-1)
        agreement = (argmax_dense == argmax_svd).float().mean().item()

        # Windowed oracle FULL demand set tracking for session Jaccard
        window_full_sets = []
        oracle_gate = OracleGate(tau_skip=0.01, tau_core=0.20)
        oracle_runtime = WARIRuntime(
            mode=ExecutionMode.ORACLE_WARI,
            cores=cores,
            dense_blobs=blobs,
            gate=oracle_gate,
            controller=controller,
            memory_sim=sim,
        )

        test_model = MiniGPT(cfg)
        test_model.load_state_dict(base_model.state_dict())
        test_model = HFModelAdapter.apply_wari(test_model, oracle_runtime)

        window_size = seq_len // num_windows
        for w_idx in range(num_windows):
            sub_tokens = tokens[:, w_idx * window_size : (w_idx + 1) * window_size]
            oracle_runtime.reset_telemetry()
            with torch.no_grad():
                test_model(sub_tokens)
            full_layers = {name for name, dec in oracle_runtime.current_step_decisions.items() if dec == "full"}
            window_full_sets.append(full_layers)

        jaccards = []
        for i in range(len(window_full_sets) - 1):
            jaccards.append(compute_jaccard(window_full_sets[i], window_full_sets[i + 1]))
        mean_jaccard = sum(jaccards) / len(jaccards) if jaccards else 0.0

        results_by_rank[str(r)] = {
            "rank": r,
            "mean_relative_frobenius": round(mean_frob, 4),
            "core_to_dense_byte_ratio": round(byte_ratio, 4),
            "kl_divergence": round(kl_div, 4),
            "argmax_agreement": round(agreement, 4),
            "window_jaccard_mean": round(mean_jaccard, 4),
        }

    # Evaluate Go / No-Go decision on MiniGPT mechanism
    # Pre-registered criterion: useful quality AND core bytes < 0.50 AND Jaccard >= 0.50
    # On random init, Frobenius is expectedly high (~0.85 at rank 16)
    rank16_frob = results_by_rank.get("16", {}).get("mean_relative_frobenius", 1.0)
    go_decision = False  # MiniGPT random-init does not authorize trained LLM novelty

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    record = {
        "timestamp": timestamp,
        "experiment": "exp0_locality",
        "model": "MiniGPT (random-init laboratory)",
        "disclaimer": "Mechanism laboratory only. Does not authorize trained LLM novelty claims.",
        "hardware": hw_snap.to_dict(),
        "seed": seed,
        "pre_registered_thresholds": {
            "max_core_byte_ratio": 0.50,
            "min_session_jaccard": 0.50,
        },
        "results_by_rank": results_by_rank,
        "exp0_go_on_minigpt": go_decision,
        "note": (
            f"rank-16 mean rel-Frobenius ≈ {rank16_frob:.3f}. "
            "Expected for Gaussian random matrices. Real LM evaluation deferred to 16GB+ host."
        ),
    }

    out_file = Path(output_dir) / f"{timestamp}_seed{seed}.json"
    with open(out_file, "w") as f:
        json.dump(record, f, indent=2)

    print(f"[Exp-0 Done] Record saved to {out_file}")
    print(f"Rank-16 Rel-Frobenius: {rank16_frob:.4f}, Go Decision: {go_decision}")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 0 Locality and Go/No-Go")
    parser.add_argument("--model", type=str, default="minigpt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/raw/exp0")
    args = parser.parse_args()
    run_exp0(model_type=args.model, seed=args.seed, output_dir=args.output_dir)
