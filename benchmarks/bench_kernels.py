"""Per-compression-operation benchmark (the setup of Table 1 in the paper).

One compression step for r requests: score one action query against the cache, turn the scores
into a keep mask (top-`budget` tokens) and compact the cache so the survivors fill the fewest blocks.

  actkv   LSE-recovery scoring directly on the paged blocks (cal_attn) + conflict-free in-place
          compaction (evict -> compact)
  gather  gather K into a contiguous buffer and score it there with the same LSE, then copy the
          survivors (K and V) into a contiguous workspace and write them back into the request's
          first blocks (paged R-KV style)

Both paths start from identical caches, get the same query and the same LSE (a paged attention
kernel returns the LSE as a by-product, so it is computed outside the timer), score in fp32, turn
scores into keep masks with the same function and end in the same state. The benchmark checks
that the two paths keep the same tokens and give the same decode attention output afterwards.

Defaults: block_size=16, kv_heads=8, head_dim=128, q_heads=32, r in {1,4,16,64},
compress 1K->0.5K, 2K->1.5K, 6K->4K. Meaningful timings need a CUDA GPU with triton.

    python benchmarks/bench_kernels.py --device cuda
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch

if not torch.cuda.is_available():  # must be set before triton decorates the kernels
    os.environ.setdefault("TRITON_INTERPRET", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from actkv.paged import PagedKVCache, keep_mask_from_scores  # noqa: E402

CASES = [(1024, 512), (2048, 1536), (6144, 4096)]
REQS = [1, 4, 16, 64]


def _sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def build(r, n, args, dev, dtype, seed=0):
    """r requests of n tokens on scattered physical blocks; the same seed gives the same cache."""
    bs = args.block_size
    nb = r * math.ceil(n / bs) + 4
    cache = PagedKVCache(nb, bs, args.kv_heads, args.head_dim, dtype=dtype, device=dev)
    cache.free = torch.randperm(nb, generator=torch.Generator().manual_seed(seed)).tolist()
    g = torch.Generator(device=dev).manual_seed(seed)
    for i in range(r):
        k = torch.randn(1, n, args.kv_heads, args.head_dim, generator=g, device=dev, dtype=dtype)
        cache.append(i, k, torch.randn(k.shape, generator=g, device=dev, dtype=dtype))
    return cache


def actkv_step(cache, reqs, q, lse, budget):
    """Score on the paged blocks, then compact in place. Returns the [R, Hkv, slots] scores."""
    bt = cache.block_table(reqs)
    out = torch.zeros(len(reqs), cache.n_kv_heads, bt.shape[1] * cache.block_size, device=q.device)
    nseen = torch.zeros(len(reqs), dtype=torch.int32, device=q.device)
    cache.cal_attn(reqs, q, lse, out, nseen)
    for i in reqs:
        cache.evict(i, out[i], budget)
    return out


def gather_step(cache, reqs, q, lse, budget, slots):
    """Gather K, score it, copy the survivors out and back into the first blocks. Returns the scores."""
    hkv, d, bs = cache.n_kv_heads, cache.head_dim, cache.block_size
    g = q.shape[1] // hkv
    k_flat = cache.kv[0, 0].view(-1, hkv, d)                       # layer 0's K plane
    kv_flat = cache.kv.view(cache.n_layers * 2, -1, hkv * d)       # every layer's K and V
    scores = []
    for i in reqs:
        k = k_flat[slots[i]].float()                               # contiguous copy of K
        logits = torch.einsum("hgd,shd->hgs", q[i].float().view(hkv, g, d), k) * d ** -0.5
        p = torch.exp(logits - lse[i].view(hkv, g, 1)).amax(1)     # [Hkv, S], max over the GQA group
        scores.append(p)
        keep = keep_mask_from_scores(p, p.shape[1], budget)
        buf = kv_flat[:, slots[i][keep]]                           # survivors -> workspace
        table = cache.tables[i]
        n_keep = math.ceil(budget / bs)
        pos = torch.arange(budget, device=q.device)
        dst = torch.tensor(table[:n_keep], device=q.device)[pos // bs] * bs + pos % bs
        kv_flat[:, dst] = buf                                      # workspace -> first blocks
        cache.free.extend(reversed(table[n_keep:]))
        cache.tables[i] = table[:n_keep]
        cache.seq_used[i] = budget
    return scores


def run_case(r, n, budget, args, dev, dtype):
    hq = args.q_heads
    reqs = list(range(r))
    q = torch.randn(r, hq, args.head_dim, generator=torch.Generator(device=dev).manual_seed(1),
                    device=dev, dtype=dtype)
    probe = build(r, n, args, dev, dtype)
    _, lse = probe.decode_attention(reqs, q)          # what the paged attention kernel would return
    slots = [probe.slot_map(i) for i in reqs]         # gather's slot maps (on-device in a real backend)

    # correctness: same keep masks, same attention afterwards
    a, b = build(r, n, args, dev, dtype), build(r, n, args, dev, dtype)
    sa, sb = actkv_step(a, reqs, q, lse, budget), gather_step(b, reqs, q, lse, budget, slots)
    masks_match = all(torch.equal(keep_mask_from_scores(sa[i], n, budget), keep_mask_from_scores(sb[i], n, budget))
                      for i in reqs)
    q2 = torch.randn_like(q)
    outputs_match = torch.allclose(a.decode_attention(reqs, q2)[0], b.decode_attention(reqs, q2)[0],
                                   rtol=1e-3, atol=1e-3)

    def timed(step):
        """Median time over --iters fresh caches, and the most extra memory one step allocated."""
        times, extra = [], []
        for _ in range(args.iters):
            cache = build(r, n, args, dev, dtype)      # compaction mutates the cache
            _sync(dev)
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
            t = time.perf_counter()
            step(cache)
            _sync(dev)
            times.append((time.perf_counter() - t) * 1e3)
            if dev == "cuda":
                extra.append(torch.cuda.max_memory_allocated() - base)
        return sorted(times)[len(times) // 2], (max(extra) / 2**20 if extra else None)

    actkv_ms, actkv_mb = timed(lambda c: actkv_step(c, reqs, q, lse, budget))
    gather_ms, gather_mb = timed(lambda c: gather_step(c, reqs, q, lse, budget, slots))
    return {"n": n, "budget": budget, "r": r, "actkv_ms": actkv_ms, "gather_ms": gather_ms,
            "actkv_extra_mb": actkv_mb, "gather_extra_mb": gather_mb,
            "memory": ("peak bytes allocated during the step above the cache (torch.cuda.max_memory_allocated)"
                       if dev == "cuda" else f"not measured: no peak-memory counter for device {dev}"),
            "masks_match": masks_match, "outputs_match": outputs_match}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--q-heads", type=int, default=32)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--reqs", type=int, nargs="*", default=REQS)
    ap.add_argument("--small", action="store_true", help="tiny sizes for a CPU smoke test")
    args = ap.parse_args(argv)
    dev = args.device
    dtype = torch.float16 if dev == "cuda" else torch.float32
    cases = [(128, 64)] if args.small else CASES
    reqs = [1, 2] if args.small else args.reqs

    print(f"device={dev} block={args.block_size} kv_heads={args.kv_heads} head_dim={args.head_dim}")
    print(f"{'case':>14} {'r':>4} {'actkv ms':>10} {'gather ms':>10} {'speedup':>8} "
          f"{'actkv MB':>9} {'gather MB':>10} {'check':>9}")
    rows = []
    for n, budget in cases:
        for r in reqs:
            row = run_case(r, n, budget, args, dev, dtype)
            rows.append(row)
            check = "ok" if row["masks_match"] and row["outputs_match"] else "MISMATCH"
            mb = [f"{v:.1f}" if v is not None else "n/a" for v in (row["actkv_extra_mb"], row["gather_extra_mb"])]
            print(f"{f'{n}->{budget}':>14} {r:>4} {row['actkv_ms']:>10.3f} {row['gather_ms']:>10.3f} "
                  f"{row['gather_ms'] / row['actkv_ms']:>7.2f}x {mb[0]:>9} {mb[1]:>10} {check:>9}")
    return rows


if __name__ == "__main__":
    main()
