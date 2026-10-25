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

  # Many configs in one process (model loaded once), task-major, stopping before a time budget
  python scripts/run_eval.py --sweep configs/vault_sweep.json --out-dir results/vault --hours 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time

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
        if r.get("method", "full") == "full":   # evictable tokens: budgets exclude the protected prompt
            lens.setdefault(r["task"], []).append(r["trace_len"] - r.get("protected", 0))
    return {k: int(statistics.median(v)) for k, v in lens.items()}


def full_ora_means(path):
    """Mean FullKV ORA length per task: the baselines' mid-round slack (L-bar) for that task."""
    ora = {}
    for r in read_rows(path):
        if r.get("method", "full") == "full":
            ora.setdefault(r["task"], []).extend(r["ora_lens"])
    return {k: int(statistics.mean(v)) for k, v in ora.items() if v}


def run_config(args) -> dict:
    """Every setting that can change an episode's outcome; resume and summaries key on it."""
    cfg = {"model": args.model, "dtype": args.dtype, "env": args.env, "method": args.method,
           "adaptive": args.adaptive, "bidirectional": args.bidirectional, "budget": args.budget,
           "budget_ratio": args.budget_ratio, "full_results": args.full_results,
           "slack": args.slack if args.slack is not None else ("fullkv-task-mean" if args.full_results else None),
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


def check_args(ap, args, produced=()):
    """Reject bad flag combinations before the (slow) model load. `produced`: files this run writes."""
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
    if args.full_results and args.full_results not in produced and not os.path.isfile(args.full_results):
        ap.error(f"--full-results {args.full_results} does not exist")
    if args.n_tasks <= 0 or args.runs <= 0:
        ap.error("--n-tasks and --runs must be positive")
    if args.alfworld_config and not os.path.isfile(args.alfworld_config):
        ap.error(f"--alfworld-config {args.alfworld_config} does not exist "
                 "(alfworld's configs/base_config.yaml lives in its GitHub repo, not the pip package)")


SWEEP_KEYS = {"method", "adaptive", "bidirectional", "budget", "budget_ratio", "full_results", "slack",
              "decay", "top_p_hit", "head_reduce"}


def load_sweep(ap, args):
    """--sweep FILE: a JSON list of {"name": ..., <per-config flags>}. The other flags are shared and
    come from the command line; relative full_results paths are resolved against --out-dir."""
    with open(args.sweep) as f:
        entries = json.load(f)
    jobs = []
    for e in entries:
        e = dict(e)
        name = e.pop("name", None)
        if not name or set(e) - SWEEP_KEYS:
            ap.error(f"sweep entry {name!r}: needs a 'name'; unknown keys {sorted(set(e) - SWEEP_KEYS)}")
        if e.get("full_results") and not os.path.isabs(e["full_results"]):
            e["full_results"] = os.path.join(args.out_dir, e["full_results"])
        c = argparse.Namespace(**{**vars(args), **{k: ap.get_default(k) for k in SWEEP_KEYS}, **e})
        jobs.append((os.path.join(args.out_dir, f"{name}.jsonl"), c))
    return jobs


def hms(seconds):
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"


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
    ap.add_argument("--budget-ratio", type=float,
                    help="per-trace budget = ratio * FullKV trace length (excluding the protected prompt)")
    ap.add_argument("--full-results", help="FullKV jsonl, needed with --budget-ratio")
    ap.add_argument("--slack", type=int, default=None,
                    help="baselines: compress mid-round past budget+slack (default: that task's mean FullKV "
                         "ORA length, if --full-results is given)")
    ap.add_argument("--decay", type=float, default=0.5)
    ap.add_argument("--top-p-hit", type=float, default=0.9)
    ap.add_argument("--head-reduce", default="mean", choices=["mean", "max"])
    ap.add_argument("--n-tasks", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--out", help="results jsonl for a single config")
    ap.add_argument("--sweep", help="JSON list of configs to run task-major in one process (with --out-dir)")
    ap.add_argument("--out-dir", help="with --sweep: one <name>.jsonl per config")
    ap.add_argument("--hours", type=float, help="don't start a task that would end past this budget "
                                                "(counts episode time already in the output files)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if bool(args.out) == bool(args.sweep) or bool(args.sweep) != bool(args.out_dir):
        ap.error("give either --out, or --sweep with --out-dir")
    jobs = load_sweep(ap, args) if args.sweep else [(args.out, args)]
    for _, c in jobs:
        check_args(ap, c, produced={out for out, _ in jobs})

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
    gpu = torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") else None

    used = 0.0        # episode seconds already in the output files (resume)
    states = []
    for out, c in jobs:
        cfg = run_config(c)
        cid = config_id(cfg)
        rows = [r for r in (read_rows(out) if os.path.exists(out) else []) if r.get("config_id") == cid]
        used += sum(r.get("seconds", 0.0) for r in rows)
        states.append((out, c, cfg, cid, {(r["task"], r["run"]) for r in rows}))

    def run_one(out, c, cfg, cid, done, task, run):
        if (task, run) in done:      # resume: only rows of this exact config count as done
            return None
        if c.method == "full":
            budget = None
        elif c.adaptive:
            budget = ConfidenceBudget(BudgetConfig(bidirectional=c.bidirectional))
        elif c.budget_ratio is not None:
            base = full_lengths(c.full_results).get(task) if os.path.exists(c.full_results) else None
            if base is None:
                print(f"skip {os.path.basename(out)} task {task}: no FullKV length")
                return None
            budget = FixedBudget(max(1, int(c.budget_ratio * base)))
        else:
            budget = FixedBudget(c.budget)
        slack = c.slack
        if slack is None and c.full_results and c.method in ("snapkv", "streaming"):
            slack = full_ora_means(c.full_results).get(task) if os.path.exists(c.full_results) else None
            if slack is None:
                print(f"skip {os.path.basename(out)} task {task}: no FullKV ORA lengths")
                return None
        ecfg = EvictionConfig(decay=c.decay, top_p=c.top_p_hit, head_reduce=c.head_reduce)
        policy = make_policy(c.method, ecfg) if c.method == "actkv" else make_policy(c.method)
        session = ActKVSession(model, tok, policy, budget=budget,
                               detector=ActionRegionDetector.for_template(fmt.template),
                               sampling=sampling, mid_round_slack=slack, seed=1000 * task + run)
        res = run_episode(session, env, fmt, task, c.max_new_tokens, c.verbose)
        res.update(run=run, method=c.method, model=c.model, config=cfg, config_id=cid,
                   game=getattr(env, "game_id", None), device=c.device, gpu=gpu)
        # the file is created by the first finished episode, so failed runs leave nothing behind
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "a") as f:
            f.write(json.dumps(res) + "\n")
        done.add((task, run))
        print(f"{os.path.basename(out)} task {task} run {run}: success={res['success']} rounds={res['rounds']} "
              f"peak={res['peak_entries']} trace={res['trace_len']} {res['seconds']:.0f}s")
        return res

    # task-major: every config gets task t before any config gets task t+1, so whenever the run
    # stops (time budget, preemption) all configs have run the same tasks. Within a task, configs run
    # in sweep order with all their runs, so ratio configs see every FullKV run of the task.
    tasks = range(args.start, args.start + args.n_tasks)

    def complete(t):
        return all((t, run) in done for *_, done in states for run in range(args.runs))

    spent, t0 = 0.0, time.time()
    for k, task in enumerate(tasks):
        started = any((task, run) in done for *_, done in states for run in range(args.runs))
        if args.hours and not started:   # a task left half-done by a preempted run is finished first
            n_done = sum(map(complete, tasks))
            per_task = (used + spent) / n_done if n_done else 0.0
            if used + spent + per_task > args.hours * 3600:
                print(f"stopping before task {task}: another task (~{hms(per_task)}) would pass the "
                      f"{args.hours:g}h budget ({hms(used + spent)} used)")
                break
        ran = False
        for state in states:
            for run in range(args.runs):
                res = run_one(*state, task, run)
                if res is not None:
                    spent += res["seconds"]
                    ran = True
        if ran and len(jobs) > 1:
            n_done = sum(map(complete, tasks))
            per_task = (used + spent) / max(n_done, 1)
            print(f"[{k + 1}/{len(tasks)} tasks] {hms(used + spent)} used, {hms(per_task)} per task, "
                  f"ETA for all tasks {hms((len(tasks) - n_done) * per_task)} (wall {hms(time.time() - t0)})")


if __name__ == "__main__":
    main()
