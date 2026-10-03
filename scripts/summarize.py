"""Summarize result files relative to FullKV and plot accuracy vs peak memory (Fig. 9 style).

    python scripts/summarize.py --full results/full.jsonl results/*.jsonl --plot results/pareto.png
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def label(rows, path):
    r = rows[0]
    if r["method"] == "full":
        return "FullKV"
    tag = "adaptive" if r.get("adaptive") else (f"r={r['budget_ratio']}" if r.get("budget_ratio")
                                                 else f"B={r.get('fixed_budget')}")
    return f"{r['method']} ({tag})"


def summarize(full_rows, rows):
    full_peak = defaultdict(list)
    full_trace = defaultdict(list)
    for r in full_rows:
        full_peak[r["task"]].append(r["peak_entries"])
        full_trace[r["task"]].append(r["trace_len"])
    full_acc = statistics.mean(r["success"] for r in full_rows)
    acc = statistics.mean(r["success"] for r in rows)
    rel_peak, rel_trace = [], []
    for r in rows:
        if r["task"] in full_peak:
            rel_peak.append(r["peak_entries"] / statistics.median(full_peak[r["task"]]))
            rel_trace.append(r["trace_len"] / statistics.median(full_trace[r["task"]]))
    return {
        "n": len(rows), "acc": acc,
        "acc_vs_full": acc / full_acc if full_acc else float("nan"),
        "peak_vs_full": statistics.mean(rel_peak) if rel_peak else float("nan"),
        "trace_vs_full": statistics.mean(rel_trace) if rel_trace else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--plot")
    args = ap.parse_args()
    full_rows = load(args.full)
    table = []
    print(f"{'config':<28} {'n':>5} {'acc':>7} {'acc/full':>9} {'peak/full':>10} {'trace/full':>11}")
    for path in [args.full] + [p for p in args.files if p != args.full]:
        rows = load(path)
        if not rows:
            continue
        s = summarize(full_rows, rows)
        name = label(rows, path)
        table.append((name, s))
        print(f"{name:<28} {s['n']:>5} {s['acc']:>7.3f} {s['acc_vs_full']:>8.1%} "
              f"{s['peak_vs_full']:>9.1%} {s['trace_vs_full']:>10.1%}")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, s in table:
            ax.scatter(100 * s["peak_vs_full"], 100 * s["acc_vs_full"], s=40 + 200 * (s["trace_vs_full"] - 0.5))
            ax.annotate(name, (100 * s["peak_vs_full"], 100 * s["acc_vs_full"]), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("Peak KV memory (% of FullKV)")
        ax.set_ylabel("Accuracy (% of FullKV)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
