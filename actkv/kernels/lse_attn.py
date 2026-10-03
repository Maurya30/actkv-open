"""Recovery-based attention calculation on a paged KV cache (ActKV Algorithm 3).

Paged attention backends return the softmax log-sum-exp (LSE) per (query, head) but not the
attention matrix. Since A_ij = exp(s_ij - LSE_i), scores can be recovered by recomputing only
q·k inside each paged block, with no gather into contiguous memory.

Layout (vLLM-style), one layer:
    kv          [2, num_blocks, block_size, n_kv_heads, head_dim]
    block_table [R, max_blocks] int32, logical block -> physical block
    seq_used    [R] int32, valid tokens per request
Output, updated in place:
    out         [R, n_kv_heads, max_blocks * block_size] float32, running mean over action tokens
    n_seen      [R] int32, action tokens already folded into `out` for each request
"""
from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False


# ----------------------------------------------------------------------------- reference (torch)

def gather_kv(kv: torch.Tensor, block_table: torch.Tensor, seq_used: int, row: int):
    """Contiguous K, V [S, H, D] for one request (what a gather-based approach materializes)."""
    bs = kv.shape[2]
    n_blocks = math.ceil(seq_used / bs)
    blocks = block_table[row, :n_blocks].long()
    k = kv[0, blocks].reshape(-1, kv.shape[3], kv.shape[4])[:seq_used]
    v = kv[1, blocks].reshape(-1, kv.shape[3], kv.shape[4])[:seq_used]
    return k, v


def paged_decode_attention_ref(q, kv, block_table, seq_used, scale=None):
    """Plain decode attention over a paged cache. Returns (out [R,Hq,D], lse [R,Hq]).

    Stands in for the paged FlashAttention backend whose LSE by-product Algorithm 3 consumes.
    """
    r_n, hq, d = q.shape
    hkv = kv.shape[3]
    g = hq // hkv
    scale = scale if scale is not None else d ** -0.5
    out = torch.empty_like(q, dtype=torch.float32)
    lse = torch.empty(r_n, hq, dtype=torch.float32, device=q.device)
    for r in range(r_n):
        k, v = gather_kv(kv, block_table, int(seq_used[r]), r)
        k = k.float().repeat_interleave(g, dim=1)        # [S, Hq, D]
        v = v.float().repeat_interleave(g, dim=1)
        logits = torch.einsum("hd,shd->hs", q[r].float(), k) * scale
        lse[r] = torch.logsumexp(logits, -1)
        out[r] = torch.einsum("hs,shd->hd", torch.softmax(logits, -1), v)
    return out, lse


def action_scores_ref(q, kv, block_table, seq_used, lse, out, n_seen, scale=None):
    r_n, hq, d = q.shape
    hkv = kv.shape[3]
    g = hq // hkv
    scale = scale if scale is not None else d ** -0.5
    for r in range(r_n):
        s = int(seq_used[r])
        k, _ = gather_kv(kv, block_table, s, r)
        k = k.float().repeat_interleave(g, dim=1)
        logits = torch.einsum("hd,shd->hs", q[r].float(), k) * scale
        p = torch.exp(logits - lse[r].unsqueeze(-1))     # [Hq, S]
        p = p.view(hkv, g, s).amax(1)                     # max over GQA group
        n = float(n_seen[r])
        out[r, :, :s] += (p - out[r, :, :s]) / (n + 1.0)
    n_seen += 1
    return out


# ----------------------------------------------------------------------------- triton kernel

if HAS_TRITON:

    @triton.jit
    def _action_scores_kernel(
        Q, KV, BT, SEQ, LSE, OUT, NSEEN,
        scale,
        stride_q_r, stride_q_h,
        stride_kv_kv, stride_kv_b, stride_kv_s, stride_kv_h,
        stride_bt_r,
        stride_lse_r,
        stride_o_r, stride_o_h,
        HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr, GROUP: tl.constexpr,
    ):
        # Step 1: one program per (request, logical block, kv head)
        r = tl.program_id(0)
        b = tl.program_id(1)
        h = tl.program_id(2)

        seq = tl.load(SEQ + r)
        if b * BLOCK_SIZE >= seq:
            return

        # Step 2: load the paged K block through the block table
        phys = tl.load(BT + r * stride_bt_r + b).to(tl.int64)
        offs_s = tl.arange(0, BLOCK_S)
        offs_d = tl.arange(0, BLOCK_D)
        tok = b * BLOCK_SIZE + offs_s
        s_mask = (offs_s < BLOCK_SIZE) & (tok < seq)
        d_mask = offs_d < HEAD_DIM
        k_ptr = (KV + phys * stride_kv_b + offs_s[:, None] * stride_kv_s
                 + h * stride_kv_h + offs_d[None, :])
        k = tl.load(k_ptr, mask=s_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        # Steps 3-4: recover probs for every query head in the GQA group, max-reduce
        best = tl.zeros([BLOCK_S], dtype=tl.float32)
        for gi in tl.static_range(GROUP):
            qh = h * GROUP + gi
            q = tl.load(Q + r * stride_q_r + qh * stride_q_h + offs_d, mask=d_mask, other=0.0)
            logits = tl.sum(k * q.to(tl.float32)[None, :], axis=1) * scale
            lse = tl.load(LSE + r * stride_lse_r + qh)
            best = tl.maximum(best, tl.exp(logits - lse))

        # Step 5: running mean over the action tokens of this region
        n = tl.load(NSEEN + r).to(tl.float32)
        o_ptr = OUT + r * stride_o_r + h * stride_o_h + tok
        old = tl.load(o_ptr, mask=s_mask, other=0.0)
        tl.store(o_ptr, old + (best - old) / (n + 1.0), mask=s_mask)


def action_scores(q, kv, block_table, seq_used, lse, out, n_seen, scale=None):
    """Fold one action query token per request into `out` (Algorithm 3). In-place; returns `out`.

    q [R, Hq, D], kv [2, NB, BS, Hkv, D], block_table [R, MB] int32, seq_used [R] int32,
    lse [R, Hq] fp32, out [R, Hkv, MB*BS] fp32, n_seen [R] int32 (incremented here).
    """
    if not HAS_TRITON:
        return action_scores_ref(q, kv, block_table, seq_used, lse, out, n_seen, scale)
    r_n, hq, d = q.shape
    _, _, bs, hkv, _ = kv.shape
    assert hq % hkv == 0
    scale = scale if scale is not None else d ** -0.5
    assert kv.stride(-1) == 1 and q.stride(-1) == 1
    grid = (r_n, block_table.shape[1], hkv)
    _action_scores_kernel[grid](
        q, kv, block_table, seq_used, lse, out, n_seen,
        scale,
        q.stride(0), q.stride(1),
        kv.stride(0), kv.stride(1), kv.stride(2), kv.stride(3),
        block_table.stride(0),
        lse.stride(0),
        out.stride(0), out.stride(1),
        HEAD_DIM=d, BLOCK_SIZE=bs,
        BLOCK_D=triton.next_power_of_2(d), BLOCK_S=triton.next_power_of_2(bs), GROUP=hq // hkv,
    )
    n_seen += 1
    return out
