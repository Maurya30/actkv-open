"""Per-compression-operation benchmark, mirroring Table 1 of the paper.

Compares one compression step (score the action query + compact) under:
  actkv   : LSE-recovery scoring on paged blocks (Triton) + conflict-free in-place compaction
  gather  : gather paged K into a contiguous buffer, recompute softmax, gather survivors into a
            new contiguous workspace (paged R-KV style)

Defaults: block_size=16, kv_heads=8, head_dim=128, q_heads=32, r in {1,4,16,64},
compress 1K->0.5K, 2K->1.5K, 6K->4K. Meant for a CUDA GPU; --device cpu works but is slow
and only useful as a smoke test.

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

from actkv.kernels.compaction import gather_compaction  # noqa: E402
from actkv.paged import PagedKVCache  # noqa: E402

CASES = [(1024, 512), (2048, 1536), (6144, 4096)]
REQS = [1, 4, 16, 64]


def _sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def _timeit(fn, dev, iters):
    for _ in range(2):
        fn()
    _sync(dev)
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(dev)
    return (time.perf_counter() - t) / iters * 1e3


def build(r, n, args, dev, dtype):
    bs = args.block_size
    nb = r * math.ceil(n / bs) + 4
    cache = PagedKVCache(nb, bs, args.kv_heads, args.head_dim, dtype=dtype, device=dev)
    cache.free = torch.randperm(nb).tolist()
    for i in range(r):
        k = torch.randn(1, n, args.kv_heads, args.head_dim, device=dev, dtype=dtype)
        cache.append(i, k, torch.randn_like(k))
    return cache


def run_case(r, n, budget, args, dev, dtype):
    hq, d = args.q_heads, args.head_dim
    reqs = list(range(r))
    q = torch.randn(r, hq, d, device=dev, dtype=dtype)
    keep = torch.zeros(r, n, dtype=torch.bool, device=dev)
    for i in range(r):
        keep[i, torch.randperm(n, device=dev)[:budget]] = True

    # ---- ActKV: paged scoring + in-place compaction (fresh cache per iteration, compaction mutates)
    def actkv_step():
        cache = build(r, n, args, dev, dtype)
        # LSE comes for free from the paged attention backend; here we compute it outside the timer
        _, lse = cache.decode_attention(reqs, q)
        bt = cache.block_table(reqs)
        out = torch.zeros(r, args.kv_heads, bt.shape[1] * args.block_size, device=dev)
        nseen = torch.zeros(r, dtype=torch.int32, device=dev)
        _sync(dev)
        t = time.perf_counter()
        cache.cal_attn(reqs, q, lse, out, nseen)
        for i in reqs:
            cache.compact(i, keep[i])
        _sync(dev)
        return (time.perf_counter() - t) * 1e3

    # ---- Gather baseline
    def gather_step():
        cache = build(r, n, args, dev, dtype)
        g = hq // args.kv_heads
        _sync(dev)
        t = time.perf_counter()
        for i in reqs:
            k, _ = cache.gather(i)                               # contiguous copy of K
            kk = k.float().repeat_interleave(g, 1)
            logits = torch.einsum("hd,shd->hs", q[i].float(), kk) * d ** -0.5
            p = torch.softmax(logits, -1).view(args.kv_heads, g, n).amax(1)
            slots = cache.slot_map(i)
            flat = cache.kv.view(2, -1, args.kv_heads * d)
            buf = gather_compaction(flat, slots[keep[i]])       # extra workspace
            del p, buf
        _sync(dev)
        return (time.perf_counter() - t) * 1e3

    a = sorted(actkv_step() for _ in range(args.iters))[args.iters // 2]
    b = sorted(gather_step() for _ in range(args.iters))[args.iters // 2]
    extra_mb = r * budget * args.kv_heads * d * 2 * torch.finfo(dtype).bits / 8 / 2**20
    return a, b, extra_mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--q-heads", type=int, default=32)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--reqs", type=int, nargs="*", default=REQS)
    ap.add_argument("--small", action="store_true", help="tiny sizes for a CPU smoke test")
    args = ap.parse_args()
    dev = args.device
    dtype = torch.float16 if dev == "cuda" else torch.float32
    cases = [(128, 64)] if args.small else CASES
    reqs = [1, 2] if args.small else args.reqs

    print(f"device={dev} block={args.block_size} kv_heads={args.kv_heads} head_dim={args.head_dim}")
    print(f"{'case':>14} {'r':>4} {'actkv ms':>10} {'gather ms':>10} {'speedup':>8} {'gather extra MB':>16}")
    for n, budget in cases:
        for r in reqs:
            a, b, mb = run_case(r, n, budget, args, dev, dtype)
            print(f"{f'{n}->{budget}':>14} {r:>4} {a:>10.3f} {b:>10.3f} {b / a:>7.2f}x {mb:>16.1f}")


if __name__ == "__main__":
    main()
