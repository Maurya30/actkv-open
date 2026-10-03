"""Reproduce the Fig. 6 access-pattern study on any (small) model, using FullKV traces.

    python scripts/access_patterns.py --model Qwen/Qwen3-8B --env vault --n-tasks 20 --out results/aav_8b.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from actkv.action_region import ActionRegionDetector  # noqa: E402
from actkv.agent import QwenFormat, make_env, run_episode  # noqa: E402
from actkv.analysis import AAVRecorder, classify  # noqa: E402
from actkv.budget import FixedBudget  # noqa: E402
from actkv.hf import ActKVSession, SamplingParams  # noqa: E402
from run_eval import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--env", default="vault", choices=["vault", "alfworld"])
    ap.add_argument("--alfworld-config", default="configs/alfworld.yaml")
    ap.add_argument("--n-tasks", type=int, default=20)
    ap.add_argument("--only-success", action="store_true", help="use FullKV-correct traces only (as the paper)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, tok = load_model(args.model, args.dtype, args.device)
    fmt = QwenFormat(tok)
    env = (make_env("alfworld", config_path=args.alfworld_config, n_tasks=args.n_tasks)
           if args.env == "alfworld" else make_env("vault", n_tasks=args.n_tasks))
    agg = defaultdict(lambda: defaultdict(float))
    used = 0
    for task in range(args.n_tasks):
        pol = AAVRecorder()
        s = ActKVSession(model, tok, pol, budget=FixedBudget(10**9),
                         detector=ActionRegionDetector.for_template(fmt.template),
                         sampling=SamplingParams(), seed=task)
        res = run_episode(s, env, fmt, task)
        if args.only_success and not res["success"]:
            continue
        stats = classify(pol.rounds, s.protected)
        if not stats:
            continue
        used += 1
        for cat, d in stats.items():
            agg[cat]["count_ratio"] += d["count_ratio"]
            agg[cat]["score_ratio"] += d["score_ratio"]
        print(task, {c: round(d["count_ratio"], 3) for c, d in stats.items()})
    summary = {c: {k: v / max(used, 1) for k, v in d.items()} for c, d in agg.items()}
    summary["traces"] = used
    summary["model"] = args.model
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
