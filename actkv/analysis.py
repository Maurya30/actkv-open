"""Action access-pattern study (Section 5.1 / Fig. 6 of the paper).

For every KV entry, the action access vector AAV = [a_1..a_n] marks whether the entry was in the
top-p attention mass of each action region that could attend to it. Entries are classified:
    cold-start   : starts with one or more zeros
    intermittent : has zeros between its first and last one
    continuous   : no zero between first and last one
(all-zero vectors are discarded). The paper finds cold-start is negligible and continuous is the
most common, while intermittent entries carry more attention per access, which motivates LRFU.
Run this on FullKV traces of small models to check whether that still holds.
"""
from __future__ import annotations

from collections import defaultdict

import torch

from .eviction import ActionLRFU, top_p_mask


class AAVRecorder(ActionLRFU):
    """Never evicts; records the per-round hit mask and scores of each action region."""

    name = "aav-recorder"

    def reset(self, n_layers, device="cpu"):
        super().reset(n_layers, device)
        self.rounds: list[tuple[torch.Tensor, torch.Tensor]] = []   # (hit [L,S], score [L,S]) per round

    def select(self, length, protected, budget):
        if self.n_act == 0:
            return None
        act = self._grow(self.act, length)[:, :length].clone()
        hit = torch.zeros_like(act, dtype=torch.bool)
        for layer in range(self.n_layers):
            hit[layer, protected:] = top_p_mask(act[layer, protected:], self.cfg.top_p)
        self.rounds.append((hit.cpu(), act.cpu()))
        self.protected_len = protected
        return None


def classify(rounds, protected: int = 0) -> dict:
    """rounds: list of (hit [L,S_t], score [L,S_t]); S_t grows by round. Returns counts + scores."""
    if not rounds:
        return {}
    n_layers = rounds[0][0].shape[0]
    final_len = rounds[-1][0].shape[1]
    counts = defaultdict(int)
    scores = defaultdict(float)
    for layer in range(n_layers):
        for i in range(protected, final_len):
            vec, sc = [], []
            for hit, score in rounds:
                if i < hit.shape[1]:          # region could attend to entry i
                    vec.append(bool(hit[layer, i]))
                    sc.append(float(score[layer, i]))
            if not any(vec):
                continue
            first = vec.index(True)
            last = len(vec) - 1 - vec[::-1].index(True)
            if first > 0:
                cat = "cold-start"
            elif not all(vec[first:last + 1]):
                cat = "intermittent"
            else:
                cat = "continuous"
            counts[cat] += 1
            hits = [s for v, s in zip(vec, sc) if v]
            scores[cat] += sum(hits) / len(hits)          # per-access attention score
    total_c = sum(counts.values()) or 1
    total_s = sum(scores.values()) or 1.0
    return {c: {"count_ratio": counts[c] / total_c, "score_ratio": scores[c] / total_s,
                "count": counts[c]}
            for c in ("cold-start", "continuous", "intermittent")}
