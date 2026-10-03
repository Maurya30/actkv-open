"""In-place KV cache compaction on a paged layout (ActKV Algorithm 4).

Token eviction leaves holes scattered over pages. Instead of gathering the survivors into a new
buffer (extra memory) or shifting them in place (read/write conflicts), we plan conflict-free
copies first:

    X     = ceil(B / block_size)              blocks needed after compaction
    start = (Y - X) * block_size              Y = blocks used before compaction
    end   = start + B
    dst   = evicted slots inside [start, end)
    src   = kept slots outside [start, end)

src and dst are disjoint, so every copy can run in parallel in one pass with no scratch buffer.
Afterwards the request's first Y - X logical blocks are free and go back to the allocator.

Note: survivors are permuted (a moved entry lands wherever a hole was). Attention over cached
keys is order-invariant once RoPE has been applied, so this is safe; any per-token metadata
(e.g. ActKV keep scores) must be permuted with `perm`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False


@dataclass
class CompactionPlan:
    src_slots: torch.Tensor   # physical slots to read
    dst_slots: torch.Tensor   # physical slots to write
    perm: torch.Tensor        # [B] new logical position -> old logical position
    drop_blocks: int          # leading logical blocks that become free (Y - X)


def plan_compaction(keep_mask: torch.Tensor, seq_used: int, block_table_row: torch.Tensor,
                    block_size: int) -> CompactionPlan:
    """Algorithm 4 steps 1-5 for one request. keep_mask: [>= seq_used] bool, True = retain."""
    keep = keep_mask[:seq_used].clone()
    budget = int(keep.sum())
    y = math.ceil(seq_used / block_size)
    x = math.ceil(budget / block_size)
    start = (y - x) * block_size
    end = start + budget
    span = y * block_size
    full = torch.zeros(span, dtype=torch.bool, device=keep.device)
    full[:seq_used] = keep

    dst_logical = torch.nonzero(~full[start:end]).flatten() + start   # holes in the tail region
    outside = full.clone()
    outside[start:end] = False
    src_logical = torch.nonzero(outside).flatten()
    assert src_logical.numel() == dst_logical.numel(), "kept count must equal budget"

    def to_phys(logical):
        blocks = block_table_row[(logical // block_size)].long()
        return blocks * block_size + logical % block_size

    perm = torch.arange(start, end, device=keep.device)
    perm[dst_logical - start] = src_logical
    return CompactionPlan(to_phys(src_logical), to_phys(dst_logical), perm, y - x)


def apply_copies_ref(kv_flat: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> None:
    """kv_flat [N, slots, H*D]; disjoint src/dst, so a single vectorized copy is safe."""
    kv_flat[:, dst] = kv_flat[:, src]


if HAS_TRITON:

    @triton.jit
    def _copy_slots_kernel(KV, SRC, DST, n_planes, stride_plane, stride_slot,
                           ROW: tl.constexpr, BLOCK: tl.constexpr):
        # Step 6: one program per (copy pair, chunk of the H*D row); loops over K/V (and layers)
        p = tl.program_id(0)
        c = tl.program_id(1)
        src = tl.load(SRC + p).to(tl.int64)
        dst = tl.load(DST + p).to(tl.int64)
        offs = c * BLOCK + tl.arange(0, BLOCK)
        m = offs < ROW
        for plane in range(n_planes):
            base = KV + plane * stride_plane
            val = tl.load(base + src * stride_slot + offs, mask=m)
            tl.store(base + dst * stride_slot + offs, val, mask=m)


def apply_copies(kv_flat: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> None:
    if src.numel() == 0:
        return
    if not HAS_TRITON:
        return apply_copies_ref(kv_flat, src, dst)
    planes, _, row = kv_flat.shape
    assert kv_flat.stride(2) == 1
    block = min(1024, triton.next_power_of_2(row))
    grid = (src.numel(), triton.cdiv(row, block))
    _copy_slots_kernel[grid](kv_flat, src.contiguous(), dst.contiguous(), planes,
                             kv_flat.stride(0), kv_flat.stride(1), ROW=row, BLOCK=block)


def gather_compaction(kv_flat: torch.Tensor, keep_slots: torch.Tensor) -> torch.Tensor:
    """Gather-based baseline (paged R-KV style): survivors go into a fresh contiguous buffer.

    Needs an extra [planes, B, H*D] workspace; used only for benchmarking against Algorithm 4.
    """
    return kv_flat[:, keep_slots].clone()
