"""Summarize result files relative to FullKV and plot accuracy vs peak memory (Fig. 9 style).

    python scripts/summarize.py --full results/full.jsonl results/*.jsonl --plot results/pareto.png
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict


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
        ax.scatter(100 * s["peak_vs_full"], 100 * s["acc"], s=size)
        ax.annotate(name, (100 * s["peak_vs_full"], 100 * s["acc"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Peak KV memory (% of FullKV)")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--plot")
    args = ap.parse_args()
    full_rows = load(args.full)
    groups, configs = {}, {}          # rows grouped by run config, in order of first appearance
    for path in dict.fromkeys([args.full, *args.files]):
        for r in load(path):
            cfg = r.get("config") or {"method": r["method"]}
            cid = r.get("config_id") or json.dumps(cfg, sort_keys=True)
            configs.setdefault(cid, cfg)
            groups.setdefault(cid, []).append(r)
    names = labels(configs)
    table = []
    print(f"{'config':<28} {'n':>5} {'acc':>7} {'acc/full':>9} {'peak/full':>10} {'trace/full':>11}")
    for cid, rows in groups.items():
        s = summarize(full_rows, rows)
        name = names[cid]
        table.append((name, s))
        print(f"{name:<28} {s['n']:>5} {s['acc']:>7.3f} {pct(s['acc_vs_full']):>9} "
              f"{pct(s['peak_vs_full']):>10} {pct(s['trace_vs_full']):>11}")
    if args.plot:
        plot(table, args.plot)


if __name__ == "__main__":
    main()
