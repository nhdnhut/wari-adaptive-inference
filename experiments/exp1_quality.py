"""Experiment 1: Quality evaluation protocols (WikiText-2 PPL, LAMBADA, Teacher Agreement)."""

import argparse
import json
import os
import sys
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
    factor_model,
)
from wari.hf_wrapper import HFModelAdapter, check_ram_safety


def evaluate_minigpt_quality(
    prompts_file: str = "benchmarks/prompts.jsonl",
    seed: int = 42,
    output_dir: str = "results/raw/exp1",
):
    """Evaluate teacher agreement and KL on MiniGPT mechanism lab."""
    torch.manual_seed(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    hw_snap = HardwareSnapshot.capture()

    cfg = MiniGPTConfig(vocab_size=1024, block_size=256, n_layer=8, n_head=8, n_embd=256, ffn_dim=1024)
    teacher = MiniGPT(cfg)
    teacher.eval()

    cores, blobs = factor_model(teacher, rank=32)
    layer_sizes = {name: c.dense_bytes for name, c in cores.items()}
    controller = WorkingSetController(layer_sizes, pin_budget_bytes=2 * 1024 * 1024, beta=0.8)
    sim = MemorySimulator(HardwareSpec())
    gate = ThresholdGate(tau_skip=0.05, tau_core=0.25)

    runtime = WARIRuntime(
        mode=ExecutionMode.WARI,
        cores=cores,
        dense_blobs=blobs,
        gate=gate,
        controller=controller,
        memory_sim=sim,
    )

    student = MiniGPT(cfg)
    student.load_state_dict(teacher.state_dict())
    student = HFModelAdapter.apply_wari(student, runtime)

    # Synthetic prompt evaluation
    test_inputs = torch.randint(0, cfg.vocab_size, (8, 64))

    with torch.no_grad():
        logits_teacher, _ = teacher(test_inputs)
        logits_student, _ = student(test_inputs)

    prob_teacher = F.softmax(logits_teacher, dim=-1)
    log_prob_student = F.log_softmax(logits_student, dim=-1)
    kl = F.kl_div(log_prob_student, prob_teacher, reduction="batchmean").item()

    argmax_teacher = torch.argmax(logits_teacher, dim=-1)
    argmax_student = torch.argmax(logits_student, dim=-1)
    agreement = (argmax_teacher == argmax_student).float().mean().item()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    record = {
        "timestamp": timestamp,
        "experiment": "exp1_quality",
        "model": "MiniGPT (random-init laboratory)",
        "disclaimer": "MiniGPT pilot quality = KL and argmax agreement ONLY. Do not cite as LLM scores.",
        "hardware": hw_snap.to_dict(),
        "seed": seed,
        "quality": {
            "kl_divergence": round(kl, 4),
            "teacher_argmax_agreement": round(agreement, 4),
            "wikitext2_ppl": "TBD (Requires 16GB+ host for real LM)",
            "lambada_acc": "TBD (Requires 16GB+ host for real LM)",
        },
    }

    out_file = Path(output_dir) / f"{timestamp}_seed{seed}.json"
    with open(out_file, "w") as f:
        json.dump(record, f, indent=2)

    print(f"[Exp-1 MiniGPT Done] KL: {kl:.4f}, Agreement: {agreement:.4f}")
    return record


def evaluate_hf_quality(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Stub for real LM quality evaluation on WikiText-2 and LAMBADA."""
    # Strict RAM check to prevent thrashing
    check_ram_safety(model_name, required_ram_gb=12.0)
    raise NotImplementedError("Real LM evaluation is designated for a 16GB+ RAM host. See README.md.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Quality Evaluations")
    parser.add_argument("--model", type=str, default="minigpt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/raw/exp1")
    args = parser.parse_args()

    if args.model == "minigpt":
        evaluate_minigpt_quality(seed=args.seed, output_dir=args.output_dir)
    else:
        evaluate_hf_quality(args.model)
