"""ActKV Algorithm 2: confidence-driven adaptive budget allocation (+ an optional shrink extension)."""
from __future__ import annotations

import torch

from .config import BudgetConfig


def token_confidence(logits: torch.Tensor, k: int = 20) -> float:
    """T_i = -(1/k) * sum_{j<=k} log P_i(j) over the top-k probabilities.

    Peaked distribution -> tiny tail probabilities -> large T (high confidence).
    """
    logp = torch.log_softmax(logits.float(), dim=-1)
    top = torch.topk(logp, min(k, logp.shape[-1]), dim=-1).values
    return float(-top.mean())


def trace_confidence(tl: list[float], window: int, stride: int, bottom_pct: float):
    """Sliding-window min pooling, then keep the bottom-b% values.

    Returns (x, y): pooled-window indices and their values, ready for a line fit.
    """
    t = torch.tensor(tl, dtype=torch.float64)
    n = t.numel()
    if n == 0:
        return torch.zeros(0), torch.zeros(0)
    if n < window:
        window = n
    pooled = t.unfold(0, window, stride).amin(dim=1)          # R'_m, m = 0..floor((L-w)/s)
    thresh = torch.quantile(pooled, bottom_pct / 100.0)
    keep = pooled <= thresh
    x = torch.arange(pooled.numel(), dtype=torch.float64)[keep]
    return x, pooled[keep]


def fit_slope(x: torch.Tensor, y: torch.Tensor) -> float:
    """First-order least-squares slope. Fewer than two points -> 0 (no evidence)."""
    if x.numel() < 2:
        return 0.0
    xm, ym = x.mean(), y.mean()
    var = ((x - xm) ** 2).sum()
    if var == 0:
        return 0.0
    return float(((x - xm) * (y - ym)).sum() / var)


class FixedBudget:
    def __init__(self, budget: int):
        self.budget = budget

    @property
    def current(self) -> int:
        return self.budget

    def record(self, logits) -> None: ...
    def next_budget(self) -> int:
        return self.budget

    @property
    def history(self):
        return [self.budget]


class ConfidenceBudget:
    """Monitors confidence in parallel with decoding; adjusts the budget right before compression."""

    def __init__(self, cfg: BudgetConfig | None = None):
        self.cfg = cfg or BudgetConfig()
        self.budget = self.cfg.init_budget
        self.tl: list[float] = []
        self.history = [self.budget]
        self.slopes: list[float] = []
        self._rising = 0

    @property
    def current(self) -> int:
        return self.budget

    def record(self, logits: torch.Tensor) -> None:
        self.tl.append(token_confidence(logits, self.cfg.top_k))

    def slope(self) -> float:
        c = self.cfg
        x, y = trace_confidence(self.tl, c.window, c.stride, c.bottom_pct)
        return fit_slope(x, y)

    def next_budget(self) -> int:
        """Called when compression is triggered (Algorithm 2, steps 3-4)."""
        c = self.cfg
        a = self.slope()
        self.slopes.append(a)
        if a < 0:
            self.budget = min(int(self.budget * c.scale_up), c.upper)
            self._rising = 0
        elif c.bidirectional and a > c.shrink_slope:
            self._rising += 1
            if self._rising >= c.shrink_patience:
                self.budget = max(int(self.budget / c.scale_down), c.lower)
                self._rising = 0
        else:
            self._rising = 0
        self.history.append(self.budget)
        return self.budget
