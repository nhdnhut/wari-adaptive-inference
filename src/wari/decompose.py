"""SVD Core Factorization and Storage Layout for WARI-L."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class LayerCore(nn.Module):
    """RAM-resident low-rank core representing W \approx U_r V_r."""
    name: str
    in_features: int
    out_features: int
    rank: int
    v_proj: nn.Linear  # (in_features -> rank, bias=False)
    u_proj: nn.Linear  # (rank -> out_features, bias=original bias)
    frobenius_error: float
    core_bytes: int
    dense_bytes: int

    def __init__(
        self,
        name: str,
        in_features: int,
        out_features: int,
        rank: int,
        u_weight: torch.Tensor,
        v_weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        frobenius_error: float,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.name = name
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.frobenius_error = frobenius_error

        # V projection: x @ V^T
        self.v_proj = nn.Linear(in_features, rank, bias=False, dtype=dtype)
        self.v_proj.weight.data.copy_(v_weight)

        # U projection: (x @ V^T) @ U^T + b
        self.u_proj = nn.Linear(rank, out_features, bias=(bias is not None), dtype=dtype)
        self.u_proj.weight.data.copy_(u_weight)
        if bias is not None:
            self.u_proj.bias.data.copy_(bias)

        bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
        self.core_bytes = (in_features * rank + rank * out_features + (out_features if bias is not None else 0)) * bytes_per_elem
        self.dense_bytes = (in_features * out_features + (out_features if bias is not None else 0)) * bytes_per_elem

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using only RAM-resident thin GEMMs: y = U(V(x)) + b."""
        return self.u_proj(self.v_proj(x))


def factor_linear(
    name: str,
    layer: nn.Linear,
    rank: int,
    dtype: torch.dtype = torch.float32,
) -> Tuple[LayerCore, torch.Tensor, Optional[torch.Tensor]]:
    """Factor a single PyTorch Linear layer into RAM core (U, V) and pageable dense W.

    Disk storage constraint:
    W is preserved as a pageable tensor blob on disk / backing store.
    NEVER persist a dense residual R = W - UV.
    """
    W = layer.weight.data.to(torch.float32)  # (out_features, in_features)
    m, n = W.shape
    eff_rank = min(rank, m, n)

    # Truncated SVD: W = U S Vh
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)

    # Balance singular values symmetrically across U and V for numerical stability
    S_sqrt = torch.sqrt(S[:eff_rank])
    U_r = U[:, :eff_rank] * S_sqrt.unsqueeze(0)      # (m, r)
    V_r = S_sqrt.unsqueeze(1) * Vh[:eff_rank, :]     # (r, n)

    # Calculate relative Frobenius reconstruction error ||W - UV||_F / ||W||_F
    W_recon = U_r @ V_r
    frob_norm_orig = torch.linalg.norm(W, ord="fro").item()
    frob_norm_diff = torch.linalg.norm(W - W_recon, ord="fro").item()
    rel_frobenius = frob_norm_diff / (frob_norm_orig + 1e-12)

    bias = layer.bias.data.clone() if layer.bias is not None else None

    core = LayerCore(
        name=name,
        in_features=n,
        out_features=m,
        rank=eff_rank,
        u_weight=U_r.to(dtype),
        v_weight=V_r.to(dtype),
        bias=bias.to(dtype) if bias is not None else None,
        frobenius_error=rel_frobenius,
        dtype=dtype,
    )

    # Original pageable dense weight blob (to be stored in disk / simulated backing store)
    dense_weight = layer.weight.data.clone().to(dtype)
    dense_bias = layer.bias.data.clone().to(dtype) if layer.bias is not None else None

    return core, dense_weight, dense_bias


def factor_model(
    model: nn.Module,
    rank: int,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Dict[str, LayerCore], Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]]]:
    """Factor all pageable Linear layers in a model into RAM-resident cores and disk blobs."""
    cores: Dict[str, LayerCore] = {}
    disk_blobs: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not name.endswith("lm_head"):
            core, w_blob, b_blob = factor_linear(name, module, rank=rank, dtype=dtype)
            cores[name] = core
            disk_blobs[name] = (w_blob, b_blob)

    return cores, disk_blobs
