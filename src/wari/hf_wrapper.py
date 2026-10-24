"""HuggingFace model adapter for GPT-2, TinyLlama, and Qwen2 under WARI-L dispatch."""

import os
from typing import Dict, List, Optional, Tuple

import psutil
import torch
import torch.nn as nn

from .decompose import LayerCore, factor_linear
from .runtime import WARIRuntime, WARISublayer


def check_ram_safety(model_name: str, required_ram_gb: float = 10.0) -> bool:
    """Verify host machine has sufficient physical RAM before loading large weights."""
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    if total_ram_gb < required_ram_gb:
        raise RuntimeError(
            f"Host machine has {total_ram_gb:.2f} GB RAM, which is below the safe threshold of "
            f"{required_ram_gb:.1f} GB required to factor and evaluate {model_name}. "
            "To prevent out-of-memory thrashing or crash, please refer to the README guide: "
            "'How to run Exp-0 on a 16GB+ laptop'."
        )
    return True


class HFModelAdapter:
    """Wraps HuggingFace causal language models for WARI-L layer-level dispatch."""

    PAGEABLE_PATTERNS = {
        "gpt2": ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],
        "tinyllama": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
        "qwen2": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
    }

    @staticmethod
    def identify_pageable_layers(model: nn.Module) -> Dict[str, nn.Linear]:
        """Identify candidate pageable linear modules excluding the LM output head."""
        pageable = {}
        for name, module in model.named_modules():
            # Exclude final projection head
            if isinstance(module, nn.Linear) and not name.endswith("lm_head") and not name.endswith("wte"):
                pageable[name] = module
        return pageable

    @classmethod
    def apply_wari(
        cls,
        model: nn.Module,
        runtime: WARIRuntime,
    ) -> nn.Module:
        """Replace standard linear layers with WARISublayer modules."""
        for name, core in runtime.cores.items():
            w_dense, b_dense = runtime.dense_blobs[name]
            sublayer = WARISublayer(
                name=name,
                core=core,
                dense_weight=w_dense,
                dense_bias=b_dense,
                runtime=runtime,
            )

            # Navigate parent module and replace attribute
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                if p.isdigit():
                    parent = parent[int(p)]
                else:
                    parent = getattr(parent, p)

            attr_name = parts[-1]
            setattr(parent, attr_name, sublayer)

        return model
