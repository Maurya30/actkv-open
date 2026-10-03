"""Hyperparameters. Defaults match Section 8.1 of the ActKV paper (arXiv:2609.31395)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvictionConfig:
    # Algorithm 1
    decay: float = 0.5          # lambda, keep-score decay per compression step
    top_p: float = 0.90         # hit threshold: entries in the top-p attention mass of the action region
    # How per-KV-head scores are reduced to one score per token (one keep mask per layer).
    # The paper compacts whole slots (all heads together), so a per-token score is required;
    # it does not say how heads are combined. "mean" is our default, "max" is available.
    head_reduce: str = "mean"


@dataclass
class BudgetConfig:
    # Algorithm 2
    top_k: int = 20             # top-k tokens used for token-level confidence
    window: int = 64            # sliding window w for min pooling
    stride: int = 8             # stride s
    bottom_pct: float = 10.0    # keep bottom b% of pooled values
    init_budget: int = 512      # starting budget (entries)
    scale_up: float = 1.75
    upper: int = 30_000
    # Extension (not in the paper): allow the budget to shrink again when confidence keeps rising.
    bidirectional: bool = False
    shrink_patience: int = 3    # consecutive positive-slope compressions before shrinking
    shrink_slope: float = 0.0   # slope must exceed this to count as "rising"
    scale_down: float = 1.75
    lower: int = 256


@dataclass
class ActKVConfig:
    eviction: EvictionConfig = field(default_factory=EvictionConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    # "adaptive" -> Algorithm 2. Otherwise a fixed budget is used (entries, excluding the protected prefix).
    adaptive: bool = True
    fixed_budget: int | None = None
    # The system prompt + task description are never evicted (Section 8.1).
    protect_prefix: bool = True
