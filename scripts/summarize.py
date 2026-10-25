"""Summarize result files relative to FullKV and plot accuracy vs peak memory (Fig. 9 style).

    python scripts/summarize.py --full results/full.jsonl results/*.jsonl --plot results/pareto.png \
        --json results/summary.json

Accuracy is averaged per task first (so tasks count equally), and its 95% CI comes from a bootstrap
over tasks: tasks differ far more than repeated runs of one task. Because every config runs the same
tasks, the difference from FullKV is computed per task and bootstrapped as a paired difference.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict

BOOTSTRAP = {"unit": "task", "resamples": 2000, "level": 0.95, "seed": 0}


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def base_label(cfg):
    if cfg["method"] == "full":
        return "FullKV"
    if cfg.get("adaptive"):
        tag = "adaptive, bidirectional" if cfg.get("bidirectional") else "adaptive"
    elif cfg.get("budget_ratio") is not None:
        tag = f"r={cfg['budget_ratio']}"
    else:
        tag = f"B={cfg.get('budget')}"
    return f"{cfg['method']} ({tag})"


def labels(configs):
    """{config_id: config} -> {config_id: label}. Configs sharing a base label get the fields that differ."""
    base = {cid: base_label(c) for cid, c in configs.items()}
    out = {}
    for cid, cfg in configs.items():
        group = [c for k, c in configs.items() if base[k] == base[cid]]
        diff = sorted(k for k in set().union(*group) if len({json.dumps(c.get(k)) for c in group}) > 1)
        out[cid] = base[cid] + (" [" + ", ".join(f"{k}={cfg.get(k)}" for k in diff) + "]" if diff else "")
    return out


def bootstrap_ci(values, resamples=BOOTSTRAP["resamples"], level=BOOTSTRAP["level"], seed=BOOTSTRAP["seed"]):
    """Percentile CI of the mean, resampling the values (one per task) with replacement."""
    if len(values) < 2:
        return None, None
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(resamples))
    return means[int((1 - level) / 2 * resamples)], means[int((1 + level) / 2 * resamples) - 1]


def task_means(rows, key="success"):
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(float(r[key]))
    return {t: statistics.fmean(v) for t, v in by_task.items()}


def summarize(full_rows, rows):
    full_peak = defaultdict(list)
    full_trace = defaultdict(list)
    for r in full_rows:
        full_peak[r["task"]].append(r["peak_entries"])
        full_trace[r["task"]].append(r["trace_len"])
    full_acc_t, acc_t = task_means(full_rows), task_means(rows)
    full_acc = statistics.fmean(full_acc_t.values())
    acc = statistics.fmean(acc_t.values())
    common = sorted(set(acc_t) & set(full_acc_t))
    deltas = [acc_t[t] - full_acc_t[t] for t in common]
    rel_peak, rel_trace = [], []
    for r in rows:
        if r["task"] in full_peak:
            rel_peak.append(r["peak_entries"] / statistics.median(full_peak[r["task"]]))
            rel_trace.append(r["trace_len"] / statistics.median(full_trace[r["task"]]))
    return {
        "n_tasks": len(acc_t), "n_episodes": len(rows), "acc": acc, "acc_ci": bootstrap_ci(list(acc_t.values())),
        "n_common_tasks": len(common), "delta_vs_full": statistics.fmean(deltas) if deltas else float("nan"),
        "delta_ci": bootstrap_ci(deltas),
        "acc_vs_full": acc / full_acc if full_acc else float("nan"),
        "peak_vs_full": statistics.mean(rel_peak) if rel_peak else float("nan"),
        "trace_vs_full": statistics.mean(rel_trace) if rel_trace else float("nan"),
        "devices": sorted({r["device"] for r in rows if r.get("device")}),
    }


def fmt_ci(x, ci, signed=False):
    if math.isnan(x):
        return "n/a"
    v = f"{x:+.3f}" if signed else f"{x:.3f}"
    return v if ci[0] is None else f"{v} [{ci[0]:.2f}, {ci[1]:.2f}]"


def pct(x):
    return "n/a" if math.isnan(x) else f"{x:.1%}"


def plot(table, path):
    """Accuracy vs peak KV memory. Accuracy is absolute (a ratio to FullKV is undefined when FullKV
    never succeeds); marker area grows with trace length relative to FullKV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, s in table:
        if math.isnan(s["peak_vs_full"]):
            print(f"not plotted: {name} (no tasks in common with FullKV)")
            continue
        size = min(max(40 + 200 * (s["trace_vs_full"] - 0.5), 10), 400)
        if s["acc_ci"][0] is not None:
            ax.errorbar(100 * s["peak_vs_full"], 100 * s["acc"], fmt="none", ecolor="0.6", capsize=2,
                        yerr=[[100 * (s["acc"] - s["acc_ci"][0])], [100 * (s["acc_ci"][1] - s["acc"])]])
        ax.scatter(100 * s["peak_vs_full"], 100 * s["acc"], s=size)
        ax.annotate(name, (100 * s["peak_vs_full"], 100 * s["acc"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Peak KV memory (% of FullKV)")
    ax.set_ylabel("Accuracy (%), 95% CI")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--plot")
    ap.add_argument("--json", help="write the table (with sample sizes and CIs) as JSON")
    args = ap.parse_args()
    full_rows = load(args.full)
    if len({r.get("config_id") for r in full_rows}) > 1:
        ap.error(f"{args.full} holds FullKV runs with different settings; summarize one setting at a time")
    groups, configs = {}, {}          # rows grouped by run config, in order of first appearance
    for path in dict.fromkeys([args.full, *args.files]):
        for r in load(path):
            cfg = r.get("config") or {"method": r["method"]}
            cid = r.get("config_id") or json.dumps(cfg, sort_keys=True)
            configs.setdefault(cid, cfg)
            groups.setdefault(cid, []).append(r)
    names = labels(configs)
    table = []
    print(f"{'config':<28} {'tasks':>5} {'acc [95% CI]':>22} {'vs FullKV [95% CI]':>23} "
          f"{'peak/full':>10} {'trace/full':>11}")
    for cid, rows in groups.items():
        s = summarize(full_rows, rows)
        name = names[cid]
        table.append((name, s))
        print(f"{name:<28} {s['n_tasks']:>5} {fmt_ci(s['acc'], s['acc_ci']):>22} "
              f"{fmt_ci(s['delta_vs_full'], s['delta_ci'], signed=True):>23} "
              f"{pct(s['peak_vs_full']):>10} {pct(s['trace_vs_full']):>11}")
    if args.plot:
        plot(table, args.plot)
    if args.json:
        def clean(v):   # JSON has no NaN
            return None if isinstance(v, float) and math.isnan(v) else v
        out = {"full": args.full, "bootstrap": BOOTSTRAP, "configs": [
            {"name": name, "config_id": cid, "config": configs[cid],
             **{k: (list(v) if isinstance(v, tuple) else clean(v)) for k, v in s.items()}}
            for cid, (name, s) in zip(groups, table)]}
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
