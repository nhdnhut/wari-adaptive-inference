"""Session-adaptive Working-Set Controller for WARI-L."""

from typing import Dict, List, Set


class WorkingSetController:
    """Maintains EMA layer-demand scores and dynamically manages the pinned RAM budget.

    EMA update:
      e_l <- beta * e_l + (1 - beta) * 1[g_l == full]

    Budget constraint:
      Pins layers in descending order of e_l until pin_budget_bytes is exhausted.
      Guarantees that total pinned dense weight bytes <= pin_budget_bytes.
    """

    def __init__(
        self,
        layer_sizes: Dict[str, int],
        pin_budget_bytes: int,
        beta: float = 0.8,
    ):
        self.layer_sizes = layer_sizes  # {layer_name: bytes}
        self.pin_budget_bytes = pin_budget_bytes
        self.beta = beta

        # Initialize EMA demand scores to 0.0
        self.ema_scores: Dict[str, float] = {name: 0.0 for name in layer_sizes}
        self.pinned_layers: Set[str] = set()
        self.pinned_bytes: int = 0
        self._update_pinned_set()

    def update_demand(self, layer_decisions: Dict[str, str]):
        """Update EMA scores from observed token gate decisions and update pinned set."""
        for name in self.layer_sizes:
            is_full = 1.0 if layer_decisions.get(name) == "full" else 0.0
            self.ema_scores[name] = self.beta * self.ema_scores[name] + (1.0 - self.beta) * is_full

        self._update_pinned_set()

    def _update_pinned_set(self):
        """Greedily pin layers by descending EMA demand until budget is exhausted."""
        sorted_layers = sorted(
            self.ema_scores.keys(),
            key=lambda name: (self.ema_scores[name], -self.layer_sizes[name]),
            reverse=True,
        )

        new_pinned: Set[str] = set()
        accumulated_bytes = 0

        for name in sorted_layers:
            cost = self.layer_sizes[name]
            if accumulated_bytes + cost <= self.pin_budget_bytes:
                new_pinned.add(name)
                accumulated_bytes += cost

        self.pinned_layers = new_pinned
        self.pinned_bytes = accumulated_bytes

    def is_pinned(self, layer_name: str) -> bool:
        """Check whether a layer is currently resident in the pinned RAM budget."""
        return layer_name in self.pinned_layers

    def snapshot(self) -> dict:
        """Export controller state for telemetry and result logging."""
        return {
            "pin_budget_bytes": self.pin_budget_bytes,
            "pinned_bytes": self.pinned_bytes,
            "pinned_layers": sorted(list(self.pinned_layers)),
            "ema_scores": {k: round(v, 4) for k, v in self.ema_scores.items()},
        }
