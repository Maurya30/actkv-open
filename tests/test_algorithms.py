import torch

from actkv.budget import ConfidenceBudget, fit_slope, token_confidence, trace_confidence
from actkv.config import BudgetConfig, EvictionConfig
from actkv.eviction import ActionLRFU, SnapKV, StreamingLLM, reduce_heads, top_p_mask


def test_reduce_heads_max_over_group_then_mean():
    attn = torch.tensor([[0.1, 0.9], [0.5, 0.2],    # kv head 0 group
                         [0.3, 0.3], [0.0, 0.6]])   # kv head 1 group
    s = reduce_heads(attn, n_kv_heads=2, how="mean")
    torch.testing.assert_close(s, torch.tensor([(0.5 + 0.3) / 2, (0.9 + 0.6) / 2]))


def test_top_p_mask_is_nucleus():
    s = torch.tensor([0.5, 0.05, 0.3, 0.15, 0.0])
    m = top_p_mask(s, 0.9)          # 0.5 + 0.3 = 0.8 < 0.9 -> also needs 0.15
    assert m.tolist() == [True, False, True, True, False]


def test_lrfu_update_matches_algorithm_1():
    pol = ActionLRFU(EvictionConfig(decay=0.5, top_p=0.9))
    pol.reset(1)
    # round 1: entries 0..3, action attention concentrated on entry 2
    pol.begin_action()
    pol.observe([torch.tensor([0.05, 0.05, 0.8, 0.1])], in_action=True)
    pol.update_scores(4, protected=0)
    torch.testing.assert_close(pol.ks[0], torch.tensor([0.0, 0.0, 0.8, 0.1]))
    # round 2: two new entries; entry 2 misses, entry 4 hits
    pol.begin_action()
    pol.observe([torch.tensor([0.0, 0.0, 0.0, 0.05, 0.95, 0.0])], in_action=True)
    pol.update_scores(6, protected=0)
    torch.testing.assert_close(pol.ks[0], torch.tensor([0.0, 0.0, 0.4, 0.05, 0.95, 0.0]))


def test_lrfu_keeps_intermittent_entry_over_fresh_miss():
    """An entry that was hot a round ago survives one quiet round (the point of LRFU)."""
    pol = ActionLRFU(EvictionConfig(decay=0.5))
    pol.reset(1)
    pol.begin_action()
    pol.observe([torch.tensor([0.9, 0.1, 0.0])], True)
    pol.select(3, 0, budget=3)
    pol.begin_action()
    pol.observe([torch.tensor([0.0, 0.2, 0.0, 0.8, 0.0])], True)
    keep = pol.select(5, 0, budget=2)
    assert keep[0].tolist() == [0, 3]


def test_action_mean_ignores_non_action_tokens():
    pol = ActionLRFU()
    pol.reset(1)
    pol.begin_action()
    pol.observe([torch.tensor([1.0, 0.0])], in_action=False)
    pol.observe([torch.tensor([0.0, 1.0])], in_action=True)
    pol.observe([torch.tensor([0.0, 0.5, 0.5])], in_action=True)
    torch.testing.assert_close(pol.act[0], torch.tensor([0.0, 0.75, 0.25]))


def test_protected_prefix_always_kept():
    for pol in (ActionLRFU(), StreamingLLM(), SnapKV(window=2)):
        pol.reset(2)
        pol.begin_action()
        for _ in range(3):
            pol.observe([torch.rand(20), torch.rand(20)], True)
        keep = pol.select(20, protected=5, budget=6)
        assert keep.shape == (2, 11)
        assert (keep[:, :5] == torch.arange(5)).all()


def test_token_confidence_higher_when_peaked():
    peaked = torch.tensor([10.0] + [0.0] * 99)
    flat = torch.zeros(100)
    assert token_confidence(peaked, 20) > token_confidence(flat, 20)


def test_trace_confidence_min_pool_and_bottom_pct():
    tl = list(range(100, 0, -1))           # strictly decreasing confidence
    x, y = trace_confidence(tl, window=10, stride=5, bottom_pct=50)
    assert x.numel() > 1 and fit_slope(x, y) < 0


def test_budget_grows_on_decline_and_respects_cap():
    b = ConfidenceBudget(BudgetConfig(init_budget=512, scale_up=1.75, upper=1000, window=4, stride=2))
    b.tl = [float(v) for v in range(50, 0, -1)]
    assert b.next_budget() == 896
    assert b.next_budget() == 1000


def test_budget_holds_when_confidence_flat_or_rising():
    b = ConfidenceBudget(BudgetConfig(init_budget=512, window=4, stride=2))
    b.tl = [float(v) for v in range(50)]
    assert b.next_budget() == 512


def test_bidirectional_extension_shrinks_after_patience():
    cfg = BudgetConfig(init_budget=1000, window=4, stride=2, bidirectional=True,
                       shrink_patience=2, scale_down=2.0, lower=300)
    b = ConfidenceBudget(cfg)
    b.tl = [float(v) for v in range(50)]
    assert b.next_budget() == 1000
    assert b.next_budget() == 500
    assert b.next_budget() == 500
    assert b.next_budget() == 300


def test_aav_classification():
    from actkv.analysis import classify
    t, f = True, False
    # one layer; entry 0 continuous, entry 1 intermittent, entry 2 cold-start, entry 3 never hit
    hits = [[t, t, f, f], [t, f, t, f], [f, t, f, f]]
    rounds = [(torch.tensor([h]), torch.tensor([[0.5 if x else 0.0 for x in h]])) for h in hits]
    out = classify(rounds)
    assert out["continuous"]["count"] == 1
    assert out["intermittent"]["count"] == 1
    assert out["cold-start"]["count"] == 1
