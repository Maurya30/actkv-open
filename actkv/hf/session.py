"""ActKV on Hugging Face transformers (reference backend for accuracy experiments).

The cache is a regular contiguous DynamicCache. After compression the cache is shorter than the
number of tokens the model has seen, so we track two counters:

    seen      logical position, used for RoPE `position_ids` of new tokens
    phys      physical cache length, used for `cache_position` (mask construction / cache writes)

Cached keys already carry their RoPE rotation, so evicting or reordering them is safe as long as
new tokens keep their true logical positions.

Attention scores for eviction come from `output_attentions=True` on single-token decode steps
(needs `attn_implementation="eager"`). That is the slow-but-simple path; the paged backend uses
the LSE-recovery Triton kernel instead (actkv/kernels/lse_attn.py).

Supported: decoder-only models whose cache layers are plain full-attention `DynamicLayer`s
(Qwen2/2.5/3 dense, Llama, Mistral...). Sliding-window layers (GPT-OSS) are not handled yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import DynamicCache

from ..action_region import ActionRegionDetector
from ..budget import ConfidenceBudget, FixedBudget
from ..eviction import Policy, reduce_heads


@dataclass
class SamplingParams:
    temperature: float = 0.6     # Qwen3 thinking-mode recommendation
    top_p: float = 0.95
    top_k: int = 20
    greedy: bool = False


def sample(logits: torch.Tensor, sp: SamplingParams, gen: torch.Generator | None = None) -> int:
    if sp.greedy or sp.temperature <= 0:
        return int(torch.argmax(logits))
    logits = logits.float() / sp.temperature
    if sp.top_k > 0:
        kth = torch.topk(logits, min(sp.top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, -1)
    if sp.top_p < 1.0:
        sp_, idx = torch.sort(probs, descending=True)
        cut = torch.cumsum(sp_, 0) - sp_ > sp.top_p
        sp_[cut] = 0
        probs = torch.zeros_like(probs).scatter(0, idx, sp_)
    return int(torch.multinomial(probs / probs.sum(), 1, generator=gen))


@dataclass
class SessionStats:
    peak_entries: int = 0           # max physical cache length (per layer) reached
    trace_len: int = 0              # total tokens processed = sum of ORA lengths
    ora_lens: list[int] = field(default_factory=list)
    budgets: list[int] = field(default_factory=list)
    compressions: int = 0


class ActKVSession:
    def __init__(self, model, tokenizer, policy: Policy, budget=None,
                 detector: ActionRegionDetector | None = None,
                 sampling: SamplingParams | None = None, mid_round_slack: int | None = None,
                 seed: int | None = None):
        """
        policy           eviction policy (ActionLRFU, SnapKV, StreamingLLM, FullKV)
        budget           ConfidenceBudget | FixedBudget | int | None (None -> never compress)
        mid_round_slack  for policies with ora_only=False: compress mid-round once the cache exceeds
                         budget + slack (the paper uses the mean ORA length for baselines)
        """
        self.model, self.tok, self.policy = model, tokenizer, policy
        self.budget = FixedBudget(budget) if isinstance(budget, int) else budget
        self.detector = detector or ActionRegionDetector()
        self.sp = sampling or SamplingParams()
        self.slack = mid_round_slack
        self.gen = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
        cfg = model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        self.device = next(model.parameters()).device
        if policy.needs_attention and getattr(cfg, "_attn_implementation", "eager") != "eager":
            raise ValueError("this policy needs attention probs: load the model with attn_implementation='eager'")
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.cache = DynamicCache(config=self.model.config)
        self.seen = 0
        self.phys = 0
        self.protected = 0
        self.round_tokens = 0
        self.stats = SessionStats()
        self.policy.reset(self.n_layers, self.device)
        self.detector.reset()
        self.last_logits = None

    def _layers(self):
        layers = self.cache.layers
        for layer in layers:
            if type(layer).__name__ != "DynamicLayer":
                raise NotImplementedError(f"cache layer {type(layer).__name__} not supported")
        return layers

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def _forward(self, ids: torch.Tensor, want_attn: bool):
        n = ids.shape[-1]
        pos = torch.arange(self.seen, self.seen + n, device=self.device).unsqueeze(0)
        cpos = torch.arange(self.phys, self.phys + n, device=self.device)
        out = self.model(input_ids=ids.view(1, -1).to(self.device), past_key_values=self.cache,
                         position_ids=pos, cache_position=cpos, use_cache=True,
                         output_attentions=want_attn)
        self.seen += n
        self.phys += n
        self.round_tokens += n
        self.stats.trace_len += n
        self.stats.peak_entries = max(self.stats.peak_entries, self.phys)
        return out

    def prefill(self, text_or_ids, protected: bool = False) -> None:
        ids = text_or_ids
        if isinstance(text_or_ids, str):
            ids = self.tok(text_or_ids, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if ids.numel() == 0:
            return
        out = self._forward(ids, want_attn=False)
        self.last_logits = out.logits[0, -1]
        if protected:
            self.protected = self.phys

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, max_new_tokens: int = 2048, stop_strings: tuple[str, ...] = ()) -> str:
        """Decode until the action region closes, EOS, a stop string, or the token limit."""
        self.detector.reset()
        eos = self._eos_ids()
        text, ids = "", []
        self.ended_with_eos = False
        for _ in range(max_new_tokens):
            logits = self.last_logits
            if self.budget is not None:
                self.budget.record(logits)
            t = sample(logits, self.sp, self.gen)
            ids.append(t)
            if t in eos:
                self.ended_with_eos = True
                # feed EOS so the cache matches the text the chat template expects
                self.last_logits = self._forward(torch.tensor([t]), False).logits[0, -1]
                break
            piece = self.tok.decode([t], skip_special_tokens=False)
            text += piece
            in_action, entered = self.detector.feed(piece)
            if entered:
                self.policy.begin_action()
            want = self.policy.needs_attention
            out = self._forward(torch.tensor([t]), want)
            self.last_logits = out.logits[0, -1]
            if want:
                scores = [reduce_heads(a[0, :, -1, :].float(), self.n_kv, self.policy.head_reduce)
                          for a in out.attentions]
                self.policy.observe(scores, in_action)
            if not self.policy.ora_only and self.slack is not None and self.budget is not None:
                if self.phys - self.protected > self.budget.current + self.slack:
                    self._compress(self.budget.current)
            if self.detector.closed or any(s in text for s in stop_strings):
                break
        return text

    def _eos_ids(self) -> set[int]:
        ids = {self.tok.eos_token_id}
        gen_eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        ids.update(gen_eos if isinstance(gen_eos, (list, tuple)) else [gen_eos])
        ids.discard(None)
        return ids

    # ------------------------------------------------------------------ compression
    def end_round(self) -> None:
        """End of an observation-reasoning-action round: ActKV compresses here (Algorithm 1)."""
        self.stats.ora_lens.append(self.round_tokens)
        self.round_tokens = 0
        if self.budget is None:
            return
        b = self.budget.next_budget()
        self.stats.budgets.append(b)
        self._compress(b)

    def _compress(self, budget: int) -> None:
        keep = self.policy.select(self.phys, self.protected, budget)
        if keep is None:
            return
        keep = keep.to(self.device)
        for li, layer in enumerate(self._layers()):
            idx = keep[li]
            layer.keys = layer.keys.index_select(2, idx).contiguous()
            layer.values = layer.values.index_select(2, idx).contiguous()
        self.policy.apply_keep(keep)
        self.phys = keep.shape[1]
        self.stats.compressions += 1
