"""A minimal vLLM-style paged KV cache, used to run and test the page-aware primitives.

    kv[layer] : [2, num_blocks, block_size, n_kv_heads, head_dim]

This is a standalone model of block-based memory (allocator + block tables + slot maps), not a
vLLM integration. It is what the kernels, the benchmark, and the tests operate on, and it keeps
the same tensor layout vLLM uses so the kernels can be dropped into a real backend.
"""
from __future__ import annotations

import math

import torch

from .kernels.compaction import apply_copies, plan_compaction
from .kernels.lse_attn import action_scores, paged_decode_attention_ref


class OutOfBlocks(RuntimeError):
    pass


class PagedKVCache:
    def __init__(self, num_blocks: int, block_size: int, n_kv_heads: int, head_dim: int,
                 n_layers: int = 1, dtype=torch.float16, device="cpu"):
        self.block_size, self.n_kv_heads, self.head_dim = block_size, n_kv_heads, head_dim
        self.n_layers, self.num_blocks = n_layers, num_blocks
        self.kv = torch.zeros(n_layers, 2, num_blocks, block_size, n_kv_heads, head_dim,
                              dtype=dtype, device=device)
        self.free: list[int] = list(range(num_blocks - 1, -1, -1))
        self.tables: dict[int, list[int]] = {}
        self.seq_used: dict[int, int] = {}
        self.device = device

    # ---------------------------------------------------------------- allocation
    @property
    def used_blocks(self) -> int:
        return self.num_blocks - len(self.free)

    def _ensure(self, req: int, n_tokens: int) -> None:
        table = self.tables.setdefault(req, [])
        self.seq_used.setdefault(req, 0)
        need = math.ceil(n_tokens / self.block_size) - len(table)
        if need > len(self.free):
            raise OutOfBlocks(f"need {need} blocks, {len(self.free)} free")
        for _ in range(need):
            table.append(self.free.pop())

    def append(self, req: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """k, v: [n_layers, n, Hkv, D] new tokens for request `req`."""
        n = k.shape[1]
        start = self.seq_used.get(req, 0)
        self._ensure(req, start + n)
        slots = self.slot_map(req, start + n)[start:]
        flat = self.kv.view(self.n_layers, 2, -1, self.n_kv_heads, self.head_dim)
        flat[:, 0, slots] = k.to(self.kv.dtype)
        flat[:, 1, slots] = v.to(self.kv.dtype)
        self.seq_used[req] = start + n

    def release(self, req: int) -> None:
        self.free.extend(reversed(self.tables.pop(req, [])))
        self.seq_used.pop(req, None)

    # ---------------------------------------------------------------- views
    def slot_map(self, req: int, n: int | None = None) -> torch.Tensor:
        n = self.seq_used[req] if n is None else n
        table = torch.tensor(self.tables[req], dtype=torch.long, device=self.device)
        pos = torch.arange(n, device=self.device)
        return table[pos // self.block_size] * self.block_size + pos % self.block_size

    def block_table(self, reqs: list[int]) -> torch.Tensor:
        mb = max(len(self.tables[r]) for r in reqs)
        bt = torch.zeros(len(reqs), max(mb, 1), dtype=torch.int32, device=self.device)
        for i, r in enumerate(reqs):
            bt[i, : len(self.tables[r])] = torch.tensor(self.tables[r], dtype=torch.int32)
        return bt

    def seq_tensor(self, reqs: list[int]) -> torch.Tensor:
        return torch.tensor([self.seq_used[r] for r in reqs], dtype=torch.int32, device=self.device)

    def gather(self, req: int, layer: int = 0):
        slots = self.slot_map(req)
        flat = self.kv[layer].view(2, -1, self.n_kv_heads, self.head_dim)
        return flat[0, slots], flat[1, slots]

    # ---------------------------------------------------------------- primitives
    def decode_attention(self, reqs, q, layer=0, scale=None):
        """Reference paged decode attention -> (out, lse); stands in for the paged FA backend."""
        return paged_decode_attention_ref(q, self.kv[layer], self.block_table(reqs),
                                          self.seq_tensor(reqs), scale)

    def cal_attn(self, reqs, q, lse, out, n_seen, layer=0, scale=None):
        """Primitive 1: AttnScore <- CalAttn(target_q, KCache) via LSE recovery (Algorithm 3)."""
        return action_scores(q, self.kv[layer], self.block_table(reqs), self.seq_tensor(reqs),
                             lse, out, n_seen, scale)

    def compact(self, req: int, keep_mask: torch.Tensor) -> torch.Tensor:
        """Primitive 3: KVCache <- Compaction(KVCache, KeepMask) (Algorithm 4), all layers at once.

        Returns `perm` (new logical position -> old logical position) so callers can permute
        per-token metadata. Freed blocks go straight back to the allocator.
        """
        table = torch.tensor(self.tables[req], dtype=torch.long, device=self.device)
        plan = plan_compaction(keep_mask, self.seq_used[req], table, self.block_size)
        flat = self.kv.view(self.n_layers * 2, -1, self.n_kv_heads * self.head_dim)
        apply_copies(flat, plan.src_slots, plan.dst_slots)
        freed = self.tables[req][: plan.drop_blocks]
        self.tables[req] = self.tables[req][plan.drop_blocks:]
        self.free.extend(reversed(freed))
        self.seq_used[req] = plan.perm.numel()
        return plan.perm
