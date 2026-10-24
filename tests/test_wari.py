"""Unit tests for WARI-L core modules and reconstruction correctness."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wari import (
    GateDecision,
    HardwareSpec,
    LayerCore,
    MemorySimulator,
    MiniGPT,
    MiniGPTConfig,
    OracleGate,
    ThresholdGate,
    WARIRuntime,
    WorkingSetController,
    factor_linear,
    factor_model,
)
from wari.hf_wrapper import HFModelAdapter
from wari.runtime import ExecutionMode, WARISublayer


def test_svd_reconstruction_shapes_and_full_rank():
    """Verify SVD decomposition shapes and exact reconstruction at full rank."""
    torch.manual_seed(42)
    in_dim, out_dim = 64, 128
    layer = nn.Linear(in_dim, out_dim, bias=True)

    # Full rank should yield near-zero reconstruction error
    full_rank = min(in_dim, out_dim)
    core, w_blob, b_blob = factor_linear("test_layer", layer, rank=full_rank)

    assert core.v_proj.weight.shape == (full_rank, in_dim)
    assert core.u_proj.weight.shape == (out_dim, full_rank)
    assert core.u_proj.bias.shape == (out_dim,)
    assert core.frobenius_error < 1e-4

    x = torch.randn(2, 5, in_dim)
    y_dense = layer(x)
    y_core = core(x)
    assert torch.allclose(y_dense, y_core, atol=1e-4)


def test_svd_truncated_rank_and_compression():
    """Verify low-rank core has correct byte sizes and predictable error."""
    torch.manual_seed(42)
    in_dim, out_dim, rank = 128, 256, 16
    layer = nn.Linear(in_dim, out_dim, bias=False)

    core, w_blob, b_blob = factor_linear("test_layer", layer, rank=rank)
    assert core.rank == rank
    # Bytes check: (128*16 + 16*256) * 4 = (2048 + 4096) * 4 = 24,576
    # Dense: 128*256 * 4 = 131,072
    assert core.core_bytes < core.dense_bytes
    assert core.core_bytes == (in_dim * rank + rank * out_dim) * 4
    assert core.dense_bytes == (in_dim * out_dim) * 4
    assert 0.0 < core.frobenius_error < 1.0


def test_storage_constraint_no_residual_matrix():
    """Storage constraint: verify that dense residual matrix R is never created or stored."""
    layer = nn.Linear(32, 64)
    core, w_blob, b_blob = factor_linear("test_layer", layer, rank=8)

    # Only core and dense W exist
    assert hasattr(core, "v_proj")
    assert hasattr(core, "u_proj")
    assert not hasattr(core, "residual_matrix")
    assert not hasattr(core, "R")
    assert w_blob.shape == (64, 32)


def test_gates():
    """Verify gate decision mechanics."""
    torch.manual_seed(42)
    layer = nn.Linear(32, 32)
    core, _, _ = factor_linear("test", layer, rank=8)

    gate = ThresholdGate(tau_skip=0.01, tau_core=0.99)
    x = torch.randn(1, 4, 32)

    decision, score = gate.decide("test", x, core)
    assert isinstance(decision, GateDecision)
    assert score >= 0.0

    oracle = OracleGate(tau_skip=0.001, tau_core=0.20)
    decision_o, score_o = oracle.decide("test", x, core, dense_fn=layer)
    assert isinstance(decision_o, GateDecision)
    assert score_o >= 0.0


def test_controller_budget_invariant():
    """Verify controller enforces hard peak-RAM budget cap on pinned layers."""
    layer_sizes = {
        "layer_0": 1000,
        "layer_1": 2000,
        "layer_2": 1500,
        "layer_3": 3000,
    }
    budget = 3500
    ctrl = WorkingSetController(layer_sizes, pin_budget_bytes=budget, beta=0.5)

    # Initial state: no demand, budget respected
    assert ctrl.pinned_bytes <= budget

    # Simulate heavy demand on layer_3 and layer_1
    ctrl.update_demand({"layer_3": "full", "layer_1": "full", "layer_0": "core", "layer_2": "skip"})

    assert ctrl.pinned_bytes <= budget
    assert "layer_3" in ctrl.pinned_layers or "layer_1" in ctrl.pinned_layers
    assert ctrl.pinned_bytes <= budget  # Strict invariant


def test_memory_simulator_accounting():
    """Verify analytical memory simulation formulas for DRAM and SSD transfers."""
    spec = HardwareSpec(
        ssd_bandwidth_bytes_per_sec=1000.0,
        ssd_latency_sec=0.01,
        dram_bandwidth_bytes_per_sec=10000.0,
    )
    sim = MemorySimulator(spec)

    # Core execution: reads only DRAM
    t1 = sim.record_sublayer("l0", "core", is_pinned=False, core_bytes=500, dense_bytes=2000)
    assert sim.dram_bytes == 500
    assert sim.ssd_bytes == 0
    assert sim.ssd_misses == 0
    assert t1 == pytest.approx(500 / 10000.0)

    # Unpinned full: reads SSD and DRAM
    t2 = sim.record_sublayer("l1", "full", is_pinned=False, core_bytes=500, dense_bytes=2000)
    assert sim.ssd_bytes == 2000
    assert sim.ssd_misses == 1
    assert sim.dram_bytes == 500 + 2000
    assert t2 == pytest.approx(0.01 + (2000 / 1000.0) + (2000 / 10000.0))


def test_minigpt_and_wari_runtime_end_to_end():
    """End-to-end integration test with MiniGPT under WARI-L runtime."""
    torch.manual_seed(42)
    cfg = MiniGPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=32, ffn_dim=64)
    model = MiniGPT(cfg)

    cores, blobs = factor_model(model, rank=8)
    assert len(cores) > 0

    layer_sizes = {name: c.dense_bytes for name, c in cores.items()}
    controller = WorkingSetController(layer_sizes, pin_budget_bytes=10000)
    sim = MemorySimulator(HardwareSpec())
    gate = ThresholdGate(tau_skip=0.05, tau_core=0.5)

    runtime = WARIRuntime(
        mode=ExecutionMode.WARI,
        cores=cores,
        dense_blobs=blobs,
        gate=gate,
        controller=controller,
        memory_sim=sim,
    )

    # Swap in WARISublayers
    model = HFModelAdapter.apply_wari(model, runtime)

    prompt = torch.randint(0, 64, (1, 8))
    tokens, metrics = runtime.generate(model, prompt, max_new_tokens=4)

    assert tokens.shape == (1, 12)
    assert metrics.tokens_generated == 4
    assert metrics.wall_time_sec > 0
    assert sum(metrics.gate_histogram.values()) > 0
