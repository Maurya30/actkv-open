"""Run agent episodes under a KV policy and write one JSON line per (task, run).

Examples
  # FullKV reference (also gives per-task trace lengths for --budget-ratio runs)
  python scripts/run_eval.py --model Qwen/Qwen3-8B --env vault --method full --out results/full.jsonl

  # ActKV, adaptive budget (Algorithm 2)
  python scripts/run_eval.py --model Qwen/Qwen3-8B --env vault --method actkv --adaptive --out results/actkv_adapt.jsonl

  # Any method at a fixed per-trace budget ratio r (budget = r * FullKV trace length, as in Sec 8.2)
  python scripts/run_eval.py --model Qwen/Qwen3-8B --env vault --method snapkv --budget-ratio 0.1 \
      --full-results results/full.jsonl --out results/snapkv_10.jsonl

  # ALFWorld (after `pip install alfworld && alfworld-download`); task i = i-th game of the split
  python scripts/run_eval.py --env alfworld --alfworld-split eval_out_of_distribution ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from actkv.action_region import ActionRegionDetector  # noqa: E402
from actkv.agent import QwenFormat, make_env, run_episode  # noqa: E402
from actkv.agent.envs import ALFWORLD_SPLITS  # noqa: E402
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


def read_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def full_lengths(path):
    lens = {}
    for r in read_rows(path):
        if r.get("method", "full") == "full":
            lens.setdefault(r["task"], []).append(r["trace_len"])
    return {k: int(statistics.median(v)) for k, v in lens.items()}


def run_config(args) -> dict:
    """Every setting that can change an episode's outcome; resume and summaries key on it."""
    cfg = {"model": args.model, "dtype": args.dtype, "env": args.env, "method": args.method,
           "adaptive": args.adaptive, "bidirectional": args.bidirectional, "budget": args.budget,
           "budget_ratio": args.budget_ratio, "full_results": args.full_results,
           "slack": args.slack if args.slack is not None else ("fullkv-mean" if args.full_results else None),
           "decay": args.decay, "top_p_hit": args.top_p_hit, "head_reduce": args.head_reduce,
           "max_new_tokens": args.max_new_tokens, "greedy": args.greedy, "no_thinking": args.no_thinking}
    if args.method != "actkv":            # ActKV's knobs don't affect the other methods
        cfg.update(decay=None, top_p_hit=None, head_reduce=None)
    if args.method in ("full", "actkv"):  # only the baselines compress mid-round
        cfg.update(slack=None)
    if args.env == "alfworld":
        cfg.update(alfworld_split=args.alfworld_split, alfworld_config=args.alfworld_config)
    return cfg


def config_id(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def check_args(ap, args):
    """Reject bad flag combinations before the (slow) model load."""
    modes = [flag for flag, on in (("--adaptive", args.adaptive), ("--budget", args.budget is not None),
                                   ("--budget-ratio", args.budget_ratio is not None)) if on]
    if args.method == "full":
        if modes or args.bidirectional:
            ap.error(f"--method full never evicts; drop {', '.join(modes or ['--bidirectional'])}")
    elif len(modes) != 1:
        ap.error(f"choose exactly one of --adaptive, --budget, --budget-ratio (got {', '.join(modes) or 'none'})")
    if args.bidirectional and not args.adaptive:
        ap.error("--bidirectional only applies to --adaptive")
    if args.budget is not None and args.budget <= 0:
        ap.error("--budget must be positive")
    if args.budget_ratio is not None:
        if not 0 < args.budget_ratio <= 1:
            ap.error("--budget-ratio must be in (0, 1]")
        if not args.full_results:
            ap.error("--budget-ratio needs --full-results (the FullKV jsonl)")
    if args.full_results and not os.path.isfile(args.full_results):
        ap.error(f"--full-results {args.full_results} does not exist")
    if args.n_tasks <= 0 or args.runs <= 0:
        ap.error("--n-tasks and --runs must be positive")
    if args.alfworld_config and not os.path.isfile(args.alfworld_config):
        ap.error(f"--alfworld-config {args.alfworld_config} does not exist "
                 "(alfworld's configs/base_config.yaml lives in its GitHub repo, not the pip package)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--env", default="vault", choices=["vault", "alfworld"])
    ap.add_argument("--alfworld-config", default=None,
                    help="alfworld base_config.yaml (default: built-in, data under $ALFWORLD_DATA)")
    ap.add_argument("--alfworld-split", default="eval_out_of_distribution", choices=sorted(ALFWORLD_SPLITS))
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
    check_args(ap, args)

    env = (make_env("alfworld", config_path=args.alfworld_config, split=args.alfworld_split)
           if args.env == "alfworld" else make_env("vault", n_tasks=args.start + args.n_tasks))
    if args.start + args.n_tasks > len(env):
        ap.error(f"tasks {args.start}..{args.start + args.n_tasks - 1} exceed the {len(env)} available")
    model, tok = load_model(args.model, args.dtype, args.device)
    fmt = QwenFormat(tok, enable_thinking=not args.no_thinking)
    # Qwen3's recommended sampling: T=0.6/top_p=0.95 with thinking, T=0.7/top_p=0.8 without (top_k=20)
    sampling = (SamplingParams(temperature=0.7, top_p=0.8, top_k=20, greedy=args.greedy) if args.no_thinking
                else SamplingParams(greedy=args.greedy))
    if args.greedy and not args.no_thinking:
        print("warning: Qwen3 advises against greedy decoding in thinking mode (endless repetitions)")

    per_task = full_lengths(args.full_results) if args.budget_ratio is not None else {}
    slack = args.slack
    if slack is None and args.full_results:
        ora = []
        with open(args.full_results) as f:
            for line in f:
                ora += json.loads(line)["ora_lens"]
        slack = int(statistics.mean(ora)) if ora else None

    ecfg = EvictionConfig(decay=args.decay, top_p=args.top_p_hit, head_reduce=args.head_reduce)
    cfg = run_config(args)
    cid = config_id(cfg)
    done = set()
    if os.path.exists(args.out):  # resume: only rows of this exact config count as done
        done = {(r.get("config_id"), r["task"], r["run"]) for r in read_rows(args.out)}

    for task in range(args.start, args.start + args.n_tasks):
        for run in range(args.runs):
            if (cid, task, run) in done:
                continue
            if args.method == "full":
                budget = None
            elif args.adaptive:
                budget = ConfidenceBudget(BudgetConfig(bidirectional=args.bidirectional))
            elif args.budget_ratio is not None:
                if task not in per_task:
                    print(f"skip task {task}: no FullKV length")
                    continue
                budget = FixedBudget(max(1, int(args.budget_ratio * per_task[task])))
            else:
                budget = FixedBudget(args.budget)
            policy = make_policy(args.method, ecfg) if args.method == "actkv" else make_policy(args.method)
            session = ActKVSession(model, tok, policy, budget=budget,
                                   detector=ActionRegionDetector.for_template(fmt.template),
                                   sampling=sampling,
                                   mid_round_slack=slack, seed=1000 * task + run)
            res = run_episode(session, env, fmt, task, args.max_new_tokens, args.verbose)
            res.update(run=run, method=args.method, model=args.model, config=cfg, config_id=cid,
                       game=getattr(env, "game_id", None))
            # the file is created by the first finished episode, so failed runs leave nothing behind
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "a") as out:
                out.write(json.dumps(res) + "\n")
            print(f"task {task} run {run}: success={res['success']} rounds={res['rounds']} "
                  f"peak={res['peak_entries']} trace={res['trace_len']} {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
