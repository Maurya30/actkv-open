"""Run agent episodes under a KV policy and write one JSON line per (task, run).

Examples
  # FullKV reference (also gives per-task trace lengths for --budget-ratio runs)
  python scripts/run_eval.py --model Qwen/Qwen3-8B --env vault --method full --out results/full.jsonl

  # ActKV, adaptive budget (Algorithm 2)
  python scripts/run_eval.py --model Qwen/Qwen3-8B --env vault --method actkv --adaptive --out results/actkv_adapt.jsonl

  # Any method at a fixed per-trace budget ratio r (budget = r * FullKV trace length, as in Sec 8.2)
  python scripts/run_eval.py --model Qwen/Qwen3-8B --env vault --method snapkv --budget-ratio 0.1 \
      --full-results results/full.jsonl --out results/snapkv_10.jsonl

  # ALFWorld (after `pip install alfworld && alfworld-download`)
  python scripts/run_eval.py --env alfworld --alfworld-config configs/alfworld.yaml ...
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from actkv.action_region import ActionRegionDetector  # noqa: E402
from actkv.agent import QwenFormat, make_env, run_episode  # noqa: E402
from actkv.budget import ConfidenceBudget, FixedBudget  # noqa: E402
from actkv.config import BudgetConfig, EvictionConfig  # noqa: E402
from actkv.eviction import make_policy  # noqa: E402
from actkv.hf import ActKVSession, SamplingParams  # noqa: E402


def load_model(name, dtype, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype),
                                                 attn_implementation="eager").to(device).eval()
    return model, tok


def full_lengths(path):
    lens = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            lens.setdefault(r["task"], []).append(r["trace_len"])
    return {k: int(statistics.median(v)) for k, v in lens.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--env", default="vault", choices=["vault", "alfworld"])
    ap.add_argument("--alfworld-config", default="configs/alfworld.yaml")
    ap.add_argument("--method", default="actkv", choices=["full", "actkv", "snapkv", "streaming"])
    ap.add_argument("--adaptive", action="store_true", help="ActKV Algorithm 2 budget")
    ap.add_argument("--bidirectional", action="store_true", help="extension: allow the budget to shrink")
    ap.add_argument("--budget", type=int, help="fixed budget in entries (excl. protected prefix)")
    ap.add_argument("--budget-ratio", type=float, help="per-trace budget = ratio * FullKV trace length")
    ap.add_argument("--full-results", help="FullKV jsonl, needed with --budget-ratio")
    ap.add_argument("--slack", type=int, default=None,
                    help="baselines: compress mid-round past budget+slack (default: mean FullKV ORA length)")
    ap.add_argument("--decay", type=float, default=0.5)
    ap.add_argument("--top-p-hit", type=float, default=0.9)
    ap.add_argument("--head-reduce", default="mean", choices=["mean", "max"])
    ap.add_argument("--n-tasks", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    model, tok = load_model(args.model, args.dtype, args.device)
    fmt = QwenFormat(tok, enable_thinking=not args.no_thinking)
    env = (make_env("alfworld", config_path=args.alfworld_config, n_tasks=args.n_tasks)
           if args.env == "alfworld" else make_env("vault", n_tasks=args.start + args.n_tasks))

    per_task = full_lengths(args.full_results) if args.budget_ratio else {}
    slack = args.slack
    if slack is None and args.full_results:
        ora = []
        with open(args.full_results) as f:
            for line in f:
                ora += json.loads(line)["ora_lens"]
        slack = int(statistics.mean(ora)) if ora else None

    ecfg = EvictionConfig(decay=args.decay, top_p=args.top_p_hit, head_reduce=args.head_reduce)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out):  # resume
        with open(args.out) as f:
            done = {(json.loads(line)["task"], json.loads(line)["run"]) for line in f}

    with open(args.out, "a") as out:
        for task in range(args.start, args.start + args.n_tasks):
            for run in range(args.runs):
                if (task, run) in done:
                    continue
                if args.method == "full":
                    budget = None
                elif args.adaptive:
                    budget = ConfidenceBudget(BudgetConfig(bidirectional=args.bidirectional))
                elif args.budget_ratio:
                    if task not in per_task:
                        print(f"skip task {task}: no FullKV length")
                        continue
                    budget = FixedBudget(max(1, int(args.budget_ratio * per_task[task])))
                elif args.budget:
                    budget = FixedBudget(args.budget)
                else:
                    ap.error("choose --adaptive, --budget or --budget-ratio (or --method full)")
                policy = make_policy(args.method, ecfg) if args.method == "actkv" else make_policy(args.method)
                session = ActKVSession(model, tok, policy, budget=budget,
                                       detector=ActionRegionDetector.for_template(fmt.template),
                                       sampling=SamplingParams(greedy=args.greedy),
                                       mid_round_slack=slack, seed=1000 * task + run)
                res = run_episode(session, env, fmt, task, args.max_new_tokens, args.verbose)
                res.update(run=run, method=args.method, adaptive=args.adaptive,
                           budget_ratio=args.budget_ratio, fixed_budget=args.budget, model=args.model)
                out.write(json.dumps(res) + "\n")
                out.flush()
                print(f"task {task} run {run}: success={res['success']} rounds={res['rounds']} "
                      f"peak={res['peak_entries']} trace={res['trace_len']} {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
