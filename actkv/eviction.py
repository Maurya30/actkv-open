"""Eviction policies, backend-agnostic.

Every policy works on per-layer, per-token scores (1-D tensors over the current cache length)
and returns, at compression time, the indices to keep for each layer. All layers keep the same
number of entries so a contiguous backend can keep one sequence length.

Implemented:
  * ActionLRFU      - ActKV Algorithm 1 (action-region attention + attention-aware LRFU)
  * SnapKV          - observation-window attention over the most recent generated tokens
  * StreamingLLM    - protected prefix (attention sinks) + most recent tokens
  * FullKV          - never evicts
"""
from __future__ import annotations

import torch

from .config import EvictionConfig


def reduce_heads(attn: torch.Tensor, n_kv_heads: int, how: str = "mean") -> torch.Tensor:
    """[Hq, S] attention probs of one query token -> [S] per-token score.

    Max over each GQA group (Algorithm 3, step 4), then reduce over KV heads.
    Also accepts [Hkv, S] (already group-reduced) when Hq == n_kv_heads.
    """
    hq, s = attn.shape
    g = hq // n_kv_heads
    per_kv = attn.view(n_kv_heads, g, s).amax(dim=1)
    if how == "mean":
        return per_kv.mean(0)
    if how == "max":
        return per_kv.amax(0)
    raise ValueError(how)


def top_p_mask(scores: torch.Tensor, p: float) -> torch.Tensor:
    """Bool mask of the smallest set of entries holding >= p of the total score mass (nucleus)."""
    if scores.numel() == 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    total = scores.sum()
    if total <= 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    vals, order = torch.sort(scores, descending=True)
    csum = torch.cumsum(vals, 0)
    # include the entry that crosses the threshold
    n_keep = int(torch.searchsorted(csum, p * total).item()) + 1
    n_keep = min(n_keep, scores.numel())
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask[order[:n_keep]] = True
    return mask & (scores > 0)


def _keep_indices(protected: int, length: int, chosen_rel: torch.Tensor, device) -> torch.Tensor:
    prot = torch.arange(protected, device=device)
    return torch.cat([prot, chosen_rel + protected]).sort().values


class Policy:
    """Base class. Backends call these hooks; see actkv/hf/session.py for the call order."""

    name = "base"
    ora_only = True       # compress only at observation-reasoning-action boundaries
    needs_attention = True
    head_reduce = "mean"

    def reset(self, n_layers: int, device="cpu") -> None:
        self.n_layers = n_layers
        self.device = device

    def begin_action(self) -> None: ...
    def observe(self, scores: list[torch.Tensor], in_action: bool) -> None: ...

    def select(self, length: int, protected: int, budget: int) -> torch.Tensor:
        """Return LongTensor [n_layers, protected + budget] of indices to keep (sorted)."""
        raise NotImplementedError

    def apply_keep(self, keep: torch.Tensor) -> None:
        """Called after the backend compacted the cache with `keep`, to compact policy state."""


class FullKV(Policy):
    name = "full"
    needs_attention = False

    def select(self, length, protected, budget):
        return None


class StreamingLLM(Policy):
    name = "streaming"
    ora_only = False
    needs_attention = False

    def select(self, length, protected, budget):
        if length - protected <= budget:
            return None
        recent = torch.arange(length - budget, length, device=self.device)
        keep = torch.cat([torch.arange(protected, device=self.device), recent])
        return keep.unsqueeze(0).expand(self.n_layers, -1).contiguous()


class SnapKV(Policy):
    """Scores = attention from the last `window` generated query tokens, avg-pooled (kernel 7)."""

    name = "snapkv"
    ora_only = False

    def __init__(self, window: int = 32, pool: int = 7, head_reduce: str = "mean"):
        self.window, self.pool, self.head_reduce = window, pool, head_reduce

    def reset(self, n_layers, device="cpu"):
        super().reset(n_layers, device)
        self.rows: list[list[torch.Tensor]] = []   # ring of [layer][S_i]

    def observe(self, scores, in_action):
        self.rows.append([s.float() for s in scores])
        if len(self.rows) > self.window:
            self.rows.pop(0)

    def select(self, length, protected, budget):
        if length - protected <= budget:
            return None
        w = min(len(self.rows), budget)
        keeps = []
        for layer in range(self.n_layers):
            acc = torch.zeros(length, device=self.device)
            for row in self.rows[-w:] if w else []:
                r = row[layer]
                acc[: r.numel()] += r[:length]
            if self.pool > 1:
                acc = torch.nn.functional.avg_pool1d(
                    acc.view(1, 1, -1), self.pool, stride=1, padding=self.pool // 2
                ).view(-1)[:length]
            # the observation window itself is always kept
            win_start = length - w
            cand = acc[protected:win_start]
            k = budget - w
            chosen = torch.topk(cand, k).indices if k > 0 else cand.new_zeros(0, dtype=torch.long)
            window_idx = torch.arange(win_start - protected, length - protected, device=self.device)
            keeps.append(_keep_indices(protected, length, torch.cat([chosen, window_idx]), self.device))
        return torch.stack(keeps)

    def apply_keep(self, keep):
        # past rows index the old layout; remap them so later windows stay aligned
        for row in self.rows:
            for layer in range(self.n_layers):
                r = row[layer]
                idx = keep[layer][keep[layer] < r.numel()]
                row[layer] = r[idx]


class ActionLRFU(Policy):
    """ActKV Algorithm 1.

    Per layer and per cached token we keep:
      act[l, i]  running mean, over the action tokens of the current round, of the attention the
                 action query paid to entry i (max over GQA group, reduced over KV heads)
      ks[l, i]   keep score, updated once per compression step:
                 ks = decay * ks + act * hit,  hit = entry is in the top-p mass of act
    Compression keeps the protected prefix plus the top-`budget` keep scores.
    """

    name = "actkv"

    def __init__(self, cfg: EvictionConfig | None = None):
        self.cfg = cfg or EvictionConfig()
        self.head_reduce = self.cfg.head_reduce

    def reset(self, n_layers, device="cpu"):
        super().reset(n_layers, device)
        self.ks = torch.zeros(n_layers, 0, device=device)
        self.act = torch.zeros(n_layers, 0, device=device)
        self.n_act = 0

    @staticmethod
    def _grow(t: torch.Tensor, length: int) -> torch.Tensor:
        if t.shape[1] >= length:
            return t
        pad = t.new_zeros(t.shape[0], length - t.shape[1])
        return torch.cat([t, pad], 1)

    def begin_action(self):
        # accumulated state is reset on entry to each action region (Algorithm 3 note)
        self.act.zero_()
        self.n_act = 0

    def observe(self, scores, in_action):
        if not in_action:
            return
        length = max(s.numel() for s in scores)
        self.act = self._grow(self.act, length)
        cur = torch.zeros_like(self.act)
        for layer, s in enumerate(scores):
            cur[layer, : s.numel()] = s.float()
        self.n_act += 1
        self.act += (cur - self.act) / self.n_act   # incremental mean

    def update_scores(self, length: int, protected: int) -> None:
        """Algorithm 1 steps 2-4 (also exposed for tests)."""
        self.ks = self._grow(self.ks, length)[:, :length]     # Init(L_ora, 0) + Cat
        act = self._grow(self.act, length)[:, :length]
        for layer in range(self.n_layers):
            a = act[layer, protected:]
            hit = top_p_mask(a, self.cfg.top_p) if self.n_act else torch.zeros_like(a, dtype=torch.bool)
            ks = self.ks[layer, protected:]
            ks.mul_(self.cfg.decay)
            ks[hit] += a[hit]

    def select(self, length, protected, budget):
        self.update_scores(length, protected)
        if length - protected <= budget:
            return None
        keeps = []
        for layer in range(self.n_layers):
            chosen = torch.topk(self.ks[layer, protected:length], budget).indices
            keeps.append(_keep_indices(protected, length, chosen, self.device))
        return torch.stack(keeps)

    def apply_keep(self, keep):
        self.ks = torch.gather(self.ks, 1, keep)
        self.act = torch.gather(self._grow(self.act, int(keep.max()) + 1), 1, keep)


def make_policy(name: str, cfg: EvictionConfig | None = None, **kw) -> Policy:
    name = name.lower()
    if name in ("full", "fullkv"):
        return FullKV()
    if name in ("actkv", "action", "lrfu"):
        return ActionLRFU(cfg)
    if name == "snapkv":
        return SnapKV(**kw)
    if name in ("streaming", "streamingllm"):
        return StreamingLLM()
    raise ValueError(f"unknown policy {name}")
