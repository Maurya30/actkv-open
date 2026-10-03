import math

import pytest
import torch

from conftest import DEVICE
from actkv.kernels.compaction import apply_copies, apply_copies_ref, plan_compaction
from actkv.kernels.lse_attn import action_scores, action_scores_ref, paged_decode_attention_ref
from actkv.paged import PagedKVCache


def _filled_cache(lens, bs=16, hkv=2, d=32, layers=1, seed=0, shuffle=True):
    g = torch.Generator().manual_seed(seed)
    nb = sum(math.ceil(n / bs) for n in lens) + 8
    cache = PagedKVCache(nb, bs, hkv, d, n_layers=layers, dtype=torch.float32, device=DEVICE)
    if shuffle:  # scatter physical blocks so block tables are non-trivial
        perm = torch.randperm(nb, generator=g).tolist()
        cache.free = perm
    for r, n in enumerate(lens):
        k = torch.randn(layers, n, hkv, d, generator=g).to(DEVICE)
        v = torch.randn(layers, n, hkv, d, generator=g).to(DEVICE)
        cache.append(r, k, v)
    return cache


@pytest.mark.parametrize("lens,hq,hkv,d,bs", [
    ([37], 4, 2, 32, 16),
    ([5, 64, 100], 8, 2, 64, 16),
    ([33, 17], 6, 3, 24, 8),      # non-power-of-two head_dim and group size
])
def test_action_scores_matches_reference(lens, hq, hkv, d, bs):
    cache = _filled_cache(lens, bs=bs, hkv=hkv, d=d)
    reqs = list(range(len(lens)))
    bt, seq = cache.block_table(reqs), cache.seq_tensor(reqs)
    width = bt.shape[1] * bs
    out_k = torch.zeros(len(reqs), hkv, width, device=DEVICE)
    out_r = out_k.clone()
    n_k = torch.zeros(len(reqs), dtype=torch.int32, device=DEVICE)
    n_r = n_k.clone()
    g = torch.Generator().manual_seed(1)
    for _ in range(3):  # three action tokens -> exercises the running mean
        q = torch.randn(len(reqs), hq, d, generator=g).to(DEVICE)
        _, lse = paged_decode_attention_ref(q, cache.kv[0], bt, seq)
        action_scores(q, cache.kv[0], bt, seq, lse, out_k, n_k)
        action_scores_ref(q, cache.kv[0], bt, seq, lse, out_r, n_r)
    torch.testing.assert_close(out_k, out_r, rtol=1e-4, atol=1e-6)
    assert torch.equal(n_k, n_r)


def test_recovered_scores_are_probabilities():
    cache = _filled_cache([50], hkv=1, d=16)
    bt, seq = cache.block_table([0]), cache.seq_tensor([0])
    q = torch.randn(1, 1, 16, device=DEVICE)       # Hq == Hkv -> no GQA max, rows sum to 1
    _, lse = paged_decode_attention_ref(q, cache.kv[0], bt, seq)
    out = torch.zeros(1, 1, bt.shape[1] * 16, device=DEVICE)
    action_scores(q, cache.kv[0], bt, seq, lse, out, torch.zeros(1, dtype=torch.int32, device=DEVICE))
    assert abs(float(out[0, 0, :50].sum()) - 1.0) < 1e-4
    assert float(out[0, 0, 50:].abs().sum()) == 0.0


@pytest.mark.parametrize("seq_used,budget,bs", [(100, 40, 16), (17, 10, 16), (64, 64, 16), (200, 3, 8)])
def test_plan_is_conflict_free(seq_used, budget, bs):
    g = torch.Generator().manual_seed(seq_used)
    keep = torch.zeros(seq_used, dtype=torch.bool)
    keep[torch.randperm(seq_used, generator=g)[:budget]] = True
    table = torch.randperm(math.ceil(seq_used / bs) + 4, generator=g)
    plan = plan_compaction(keep, seq_used, table, bs)
    src, dst = set(plan.src_slots.tolist()), set(plan.dst_slots.tolist())
    assert not (src & dst)
    assert len(src) == len(plan.src_slots) and len(dst) == len(plan.dst_slots)
    assert sorted(plan.perm.tolist()) == sorted(torch.nonzero(keep).flatten().tolist())
    assert plan.drop_blocks == math.ceil(seq_used / bs) - math.ceil(budget / bs)


def test_copy_kernel_matches_reference():
    flat = torch.randn(4, 300, 96, device=DEVICE)
    src = torch.tensor([3, 50, 77, 299], device=DEVICE)
    dst = torch.tensor([10, 11, 200, 0], device=DEVICE)
    a, b = flat.clone(), flat.clone()
    apply_copies(a, src, dst)
    apply_copies_ref(b, src, dst)
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize("n,budget", [(150, 60), (40, 39), (33, 16)])
def test_compaction_preserves_kept_entries_and_frees_blocks(n, budget):
    cache = _filled_cache([n], layers=2)
    k_before = [cache.gather(0, layer)[0].clone() for layer in range(2)]
    used_before = cache.used_blocks
    keep = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    keep[torch.randperm(n)[:budget].to(DEVICE)] = True
    perm = cache.compact(0, keep)
    assert cache.seq_used[0] == budget
    for layer in range(2):
        k_after = cache.gather(0, layer)[0]
        torch.testing.assert_close(k_after, k_before[layer][perm])
    freed = used_before - cache.used_blocks
    assert freed == math.ceil(n / 16) - math.ceil(budget / 16)


def test_attention_is_invariant_to_compaction_permutation():
    """Compaction reorders survivors; decode attention over them must not change."""
    cache = _filled_cache([90], hkv=2, d=32)
    keep = torch.zeros(90, dtype=torch.bool, device=DEVICE)
    keep[torch.randperm(90)[:41].to(DEVICE)] = True
    k, v = cache.gather(0)
    k_kept, v_kept = k[keep], v[keep]
    q = torch.randn(1, 4, 32, device=DEVICE)

    ref = _filled_cache([1], hkv=2, d=32)
    ref.release(0)
    ref.append(0, k_kept.unsqueeze(0), v_kept.unsqueeze(0))
    want, _ = ref.decode_attention([0], q)

    cache.compact(0, keep)
    got, _ = cache.decode_attention([0], q)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
