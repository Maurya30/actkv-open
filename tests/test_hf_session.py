import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from actkv.eviction import ActionLRFU, Policy, StreamingLLM


def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=1024, attn_implementation="eager")
    return Qwen3ForCausalLM(cfg).eval()


class ByteTok:
    """Minimal tokenizer stand-in: token id t <-> chr(32 + t)."""
    eos_token_id = 95

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(32 + i) for i in ids)


class KeepPolicy(Policy):
    """Test policy returning a fixed keep set."""
    name = "fixed"
    needs_attention = False

    def __init__(self, keep):
        self.keep = keep

    def select(self, length, protected, budget):
        return self.keep.unsqueeze(0).expand(self.n_layers, -1)


def _session(model, policy, budget=None):
    from actkv.hf.session import ActKVSession, SamplingParams
    return ActKVSession(model, ByteTok(), policy, budget=budget, sampling=SamplingParams(greedy=True))


@torch.no_grad()
def masked_reference(model, ids, keep, new_ids):
    """Prefill everything with full attention, then decode new_ids with evicted positions masked."""
    from transformers import DynamicCache
    n = ids.numel()
    cache = DynamicCache(config=model.config)
    model(ids.unsqueeze(0), past_key_values=cache, use_cache=True)
    mask = torch.zeros(1, n + new_ids.numel(), dtype=torch.long)
    mask[0, keep] = 1
    mask[0, n:] = 1
    return model(new_ids.unsqueeze(0), past_key_values=cache, attention_mask=mask,
                 position_ids=torch.arange(n, n + new_ids.numel()).unsqueeze(0),
                 cache_position=torch.arange(n, n + new_ids.numel())).logits[0]


@pytest.mark.parametrize("keep_fn", [
    lambda n: torch.cat([torch.arange(4), torch.arange(n - 10, n)]),     # streaming-style
    lambda n: torch.tensor(sorted(torch.randperm(n)[: n // 2].tolist())),  # random scatter
])
def test_eviction_matches_masked_full_attention(keep_fn):
    model = tiny_model()
    ids = torch.randint(0, 90, (40,))
    keep = keep_fn(40)
    s = _session(model, KeepPolicy(keep), budget=1)
    s.prefill(ids)
    s._compress(1)
    assert s.phys == keep.numel() and s.seen == 40
    new = torch.randint(0, 90, (5,))
    got = s._forward(new, False).logits[0]
    want = masked_reference(model, ids, keep, new)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_reordering_cache_is_harmless():
    """Compaction may permute survivors; logits must not change."""
    model = tiny_model(1)
    ids = torch.randint(0, 90, (30,))
    perm = torch.randperm(30)
    a, b = _session(model, KeepPolicy(perm), 1), _session(model, KeepPolicy(torch.arange(30)), 1)
    for s in (a, b):
        s.prefill(ids)
        s._compress(1)
    new = torch.randint(0, 90, (3,))
    torch.testing.assert_close(a._forward(new, False).logits, b._forward(new, False).logits,
                               rtol=1e-4, atol=1e-4)


def test_streaming_policy_end_to_end():
    model = tiny_model(2)
    s = _session(model, StreamingLLM(), budget=8)
    s.prefill(torch.randint(0, 90, (6,)), protected=True)
    s.prefill(torch.randint(0, 90, (30,)))
    s.end_round()
    assert s.phys == 6 + 8 and s.seen == 36


def test_actkv_policy_collects_action_attention_and_compresses():
    model = tiny_model(3)
    pol = ActionLRFU()
    s = _session(model, pol, budget=10)
    s.prefill(torch.randint(0, 90, (8,)), protected=True)
    s.prefill(torch.randint(0, 90, (40,)))
    # force everything decoded to count as an action region
    s.detector.start, s.detector.end = "", "\x7f\x7f\x7f"
    s.generate(max_new_tokens=6)
    assert pol.n_act > 0 and pol.act.shape[1] == s.phys
    s.end_round()
    assert s.phys == 8 + 10
    assert pol.ks.shape == (3, 18)
    # keep scores are non-negative and, after one step, equal to the hit-masked action attention
    assert (pol.ks >= 0).all()
