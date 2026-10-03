# actkv-open

An unofficial, open-source implementation of **ActKV: Efficient LLM Agents through Action-Guided KV Cache Management** (Wang et al., [arXiv:2609.31395](https://arxiv.org/abs/2609.31395)). The paper ships no code. This repo implements all three components, with Triton kernels, a Hugging Face backend for agent experiments, baselines, and an eval harness.

## The idea in one paragraph

Agents run observation, reasoning, action (ORA) loops, so the KV cache grows every round. Only action tokens change the environment, and the paper finds that the KV entries actions attend to are (1) different from the ones observation and reasoning tokens attend to and (2) stable across rounds. ActKV therefore scores entries by **action-region attention**, keeps them with an attention-aware **LRFU** policy, grows the budget when the model's **confidence trends down**, and makes token-level eviction real on paged memory with two kernels: **LSE-recovery scoring** and **conflict-free in-place compaction**.

## What's here

| Paper | Code | Status |
|---|---|---|
| Alg. 1, action-oriented LRFU eviction | `actkv/eviction.py::ActionLRFU` | unit-tested against hand-worked examples |
| Alg. 2, confidence-driven budget | `actkv/budget.py::ConfidenceBudget` | unit-tested |
| Alg. 3, LSE-recovery attention kernel | `actkv/kernels/lse_attn.py` (Triton + torch ref) | matches reference (Triton interpreter) |
| Alg. 4, in-place compaction kernel | `actkv/kernels/compaction.py` (Triton + torch ref) | conflict-freedom, data integrity and block reclamation tested |
| Paged KV cache (vLLM layout) | `actkv/paged.py` | tested |
| Baselines: FullKV, SnapKV, StreamingLLM | `actkv/eviction.py` | tested |
| ReAct agent + HF backend | `actkv/hf/session.py`, `actkv/agent/` | eviction proven equivalent to masked full attention |
| Fig. 6 access-pattern study | `actkv/analysis.py`, `scripts/access_patterns.py` | tested |
| Table 1 kernel benchmark | `benchmarks/bench_kernels.py` | runs; real numbers need a GPU |
| **Extension:** bidirectional budget | `BudgetConfig(bidirectional=True)` | tested, not in the paper |

R-KV is not included yet.

## Install

```bash
pip install -e ".[triton,dev]"            # + ".[alfworld]" for ALFWorld
pytest                                    # 36 tests; on CPU the Triton kernels run in interpreter mode
```

## Run

```bash
# 1. FullKV reference (also gives per-task trace lengths for ratio budgets)
python scripts/run_eval.py --model Qwen/Qwen3-8B --env vault --method full --n-tasks 50 --runs 3 \
    --out results/full.jsonl

# 2. ActKV with the adaptive budget
python scripts/run_eval.py --model Qwen/Qwen3-8B --env vault --method actkv --adaptive \
    --out results/actkv_adaptive.jsonl

# 3. Fixed per-trace budget ratio (Sec 8.2 protocol), any method
for m in actkv snapkv streaming; do for r in 0.05 0.1 0.2 0.3; do
  python scripts/run_eval.py --model Qwen/Qwen3-8B --env vault --method $m --budget-ratio $r \
      --full-results results/full.jsonl --out results/${m}_${r}.jsonl
done; done

# 4. Table + accuracy-vs-memory plot
python scripts/summarize.py --full results/full.jsonl results/*.jsonl --plot results/pareto.png

# 5. Do action-critical entries stay stable on small models? (Fig. 6)
python scripts/access_patterns.py --model Qwen/Qwen3-8B --only-success --out results/aav_8b.json

# 6. Kernel speed (Table 1 setup), on a GPU
python benchmarks/bench_kernels.py --device cuda
```

**Environments.** `vault` is an offline toy task built for this repo: long noisy room listings, and a fact found early that is needed several rounds later. It's good for fast iteration. `alfworld` wraps the ALFWorld text env (`pip install alfworld && alfworld-download`, then point `--alfworld-config` at ALFWorld's `base_config.yaml`). The ALFWorld adapter follows the standard API but has not been run in CI yet.

## How the HF backend stays correct

After eviction the cache is shorter than the number of tokens seen, so the session tracks two counters: the logical position (for RoPE `position_ids`) and the physical cache length (for `cache_position`). Cached keys already carry their rotation, so dropping or reordering them is safe. `tests/test_hf_session.py` checks that logits after eviction exactly match a full cache with the evicted positions masked, and that reordering the cache changes nothing. Compaction depends on that second property.

Scores come from `output_attentions` on single-token decode steps, which requires `attn_implementation="eager"`. That's simple and model-agnostic but slow. The fast path is the paged kernels.

## Choices where the paper is ambiguous

- **Head aggregation.** Compaction moves whole slots (all heads), so each token needs one score per layer. Scores are max over the GQA group (as in Alg. 3) and then **mean over KV heads** (`head_reduce="max"` is available).
- **Per-layer keep sets, equal size.** Each layer picks its own top-B entries, but every layer keeps B, so a contiguous backend keeps a single sequence length.
- **Top-p hit set.** Implemented as a nucleus: the smallest set of entries holding at least 90% of the round's action attention.
- **Budget excludes the protected prefix.** The system prompt and task are never evicted and don't count toward B.
- **Confidence trend window.** The slope is fit over the whole trace so far (min-pool w=64, s=8, bottom 10%).
- **Baselines** compress mid-round once the cache exceeds B + L̄ (mean FullKV ORA length), as in Sec. 8.2. ActKV compresses only at ORA boundaries.

Defaults otherwise follow Sec. 8.1: λ=0.5, p=0.9, k=20, initial budget 512, ×1.75 scale-up, 30K cap.

## Known limitations

- Not integrated into vLLM. `actkv/paged.py` reproduces vLLM's KV layout and block tables so the kernels can be dropped in, but the scheduler and block-manager hooks are future work.
- Sliding-window layers (GPT-OSS) aren't supported by the HF backend.
- Results in the paper used 20B to 235B models on 8x RTX PRO 6000. This repo targets single-GPU 4B to 8B runs, and checking whether the method holds at that scale is part of the point.

## Citation

```bibtex
@article{wang2026actkv,
  title   = {ActKV: Efficient LLM Agents through Action-Guided KV Cache Management},
  author  = {Wang, Zihan and Tang, Cheng and Gong, Lei and Wang, Chao and Lou, Wenqi and Wang, Teng and Zhou, Xuehai},
  journal = {arXiv preprint arXiv:2609.31395},
  year    = {2026}
}
```

Unofficial implementation, not affiliated with the authors. MIT licensed.
