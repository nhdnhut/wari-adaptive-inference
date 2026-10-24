"""Gate mechanisms for WARI-L (skip, core, full)."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Optional, Tuple

import torch


class GateDecision(str, Enum):
    SKIP = "skip"  # Residual identity / zero-delta
    CORE = "core"  # Thin RAM-resident GEMMs: y = U(V(x)) + b
    FULL = "full"  # Load W if unpinned: y = Wx + b (do NOT add UV)


class WARIGate(ABC):
    """Abstract base class for sublayer execution gate."""

    @abstractmethod
    def decide(
        self,
        layer_name: str,
        x: torch.Tensor,
        core: torch.nn.Module,
        dense_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[GateDecision, float]:
        """Returns (decision, score)."""
        pass


class ThresholdGate(WARIGate):
    """MVP cheap scalar gate using relative core activation energy.

    Evaluates:
      scalar = ||V(x)||_2 / (||x||_2 + 1e-8)
    If scalar < tau_skip -> SKIP
    Else if scalar < tau_core -> CORE
    Else -> FULL
    """

    def __init__(self, tau_skip: float = 0.05, tau_core: float = 0.35):
        self.tau_skip = tau_skip
        self.tau_core = tau_core

    def decide(
        self,
        layer_name: str,
        x: torch.Tensor,
        core: torch.nn.Module,
        dense_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[GateDecision, float]:
        # Cheap projection onto low-rank V (part of the core calculation anyway)
        with torch.no_grad():
            vx = core.v_proj(x)
            norm_x = torch.norm(x, p=2, dim=-1).mean().item()
            norm_vx = torch.norm(vx, p=2, dim=-1).mean().item()
            score = norm_vx / (norm_x + 1e-8)

            if score < self.tau_skip:
                return GateDecision.SKIP, score
            elif score < self.tau_core:
                return GateDecision.CORE, score
            else:
                return GateDecision.FULL, score


class OracleGate(WARIGate):
    """Oracle gate (Exp-0 / empirical upper bound).

    Evaluates:
      rel_err = ||y_core - y_full||_2 / (||y_full||_2 + 1e-8)
    If rel_err < tau_core -> CORE
    Else -> FULL
    (Optional skip threshold if norm_full is negligible)
    """

    def __init__(self, tau_skip: float = 0.02, tau_core: float = 0.20):
        self.tau_skip = tau_skip
        self.tau_core = tau_core

    def decide(
        self,
        layer_name: str,
        x: torch.Tensor,
        core: torch.nn.Module,
        dense_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[GateDecision, float]:
        assert dense_fn is not None, "OracleGate requires dense_fn to evaluate ground truth."
        with torch.no_grad():
            y_full = dense_fn(x)
            norm_full = torch.norm(y_full, p=2, dim=-1).mean().item()

            if norm_full < 1e-6:
                return GateDecision.SKIP, 0.0

            y_core = core(x)
            diff = torch.norm(y_core - y_full, p=2, dim=-1).mean().item()
            score = diff / (norm_full + 1e-8)

            if score < self.tau_skip:
                return GateDecision.SKIP, score
            elif score < self.tau_core:
                return GateDecision.CORE, score
            else:
                return GateDecision.FULL, score
