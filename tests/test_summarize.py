import json
import sys

from conftest import load_script


def row(task, success, config, peak=100, trace=200):
    cid = json.dumps(config, sort_keys=True)
    return {"task": task, "run": 0, "success": success, "peak_entries": peak, "trace_len": trace,
            "method": config["method"], "config": config, "config_id": cid}


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


FULL = {"method": "full", "budget": None, "slack": None}


def test_configs_that_differ_only_in_slack_get_distinct_labels(tmp_path, monkeypatch, capsys):
    summarize = load_script("scripts/summarize.py")
    full = write(tmp_path / "full.jsonl", [row(0, True, FULL)])
    a = write(tmp_path / "a.jsonl", [row(0, True, {"method": "snapkv", "budget": 64, "slack": 32})])
    b = write(tmp_path / "b.jsonl", [row(0, False, {"method": "snapkv", "budget": 64, "slack": 64})])
    monkeypatch.setattr(sys, "argv", ["summarize.py", "--full", full, full, a, b])
    summarize.main()
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("snapkv")]
    assert len(lines) == 2
    assert "slack=32" in lines[0] and "slack=64" in lines[1]


def _plot_run(tmp_path, monkeypatch, full_rows, other_rows):
    """Run summarize --plot and return the (x, y) points that were scattered."""
    import math

    import matplotlib.axes
    summarize = load_script("scripts/summarize.py")
    points = []
    real = matplotlib.axes.Axes.scatter
    monkeypatch.setattr(matplotlib.axes.Axes, "scatter",
                        lambda self, x, y, **kw: points.append((x, y)) or real(self, x, y, **kw))
    full = write(tmp_path / "full.jsonl", full_rows)
    other = write(tmp_path / "other.jsonl", other_rows)
    png = tmp_path / "pareto.png"
    monkeypatch.setattr(sys, "argv", ["summarize.py", "--full", full, other, "--plot", str(png)])
    summarize.main()
    assert png.exists()
    assert all(math.isfinite(x) and math.isfinite(y) for x, y in points)
    return points


def test_plot_survives_short_traces(tmp_path, monkeypatch):
    cfg = {"method": "streaming", "budget": 64}
    pts = _plot_run(tmp_path, monkeypatch, [row(0, True, FULL, peak=100, trace=400)],
                    [row(0, True, cfg, peak=40, trace=100)])      # trace/full = 0.25
    assert len(pts) == 2


def test_plot_skips_configs_with_no_tasks_in_common(tmp_path, monkeypatch, capsys):
    cfg = {"method": "streaming", "budget": 64}
    pts = _plot_run(tmp_path, monkeypatch, [row(0, True, FULL)], [row(7, True, cfg)])
    assert len(pts) == 1
    assert "not plotted" in capsys.readouterr().out


def test_plot_works_when_fullkv_never_succeeds(tmp_path, monkeypatch):
    cfg = {"method": "streaming", "budget": 64}
    pts = _plot_run(tmp_path, monkeypatch, [row(0, False, FULL)], [row(0, True, cfg)])
    assert len(pts) == 2


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    summarize = load_script("scripts/summarize.py")
    values = [1.0] * 30 + [0.0] * 10
    lo, hi = summarize.bootstrap_ci(values)
    assert lo < 0.75 < hi and (lo, hi) == summarize.bootstrap_ci(values)
    assert summarize.bootstrap_ci([1.0]) == (None, None)     # one task: no interval


def test_paired_delta_vs_fullkv_with_sample_sizes(tmp_path, monkeypatch):
    import json
    summarize = load_script("scripts/summarize.py")
    cfg = {"method": "streaming", "budget": 64}
    full = write(tmp_path / "full.jsonl", [row(t, True, FULL) for t in range(10)])
    other = write(tmp_path / "s.jsonl", [row(t, t != 0, cfg) for t in range(10)])
    out = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", ["summarize.py", "--full", full, other, "--json", str(out)])
    summarize.main()
    res = {c["name"]: c for c in json.loads(out.read_text())["configs"]}
    s = res["streaming (B=64)"]
    assert (s["n_tasks"], s["n_episodes"], s["n_common_tasks"]) == (10, 10, 10)
    assert abs(s["acc"] - 0.9) < 1e-9 and abs(s["delta_vs_full"] + 0.1) < 1e-9
    assert s["delta_ci"][0] < -0.05 and s["delta_ci"][1] <= 0.0
    f = res["FullKV"]
    assert f["delta_vs_full"] == 0.0 and f["delta_ci"] == [0.0, 0.0] and f["acc_ci"] == [1.0, 1.0]


def test_a_fullkv_file_with_several_settings_is_rejected(tmp_path, monkeypatch):
    import pytest
    summarize = load_script("scripts/summarize.py")
    full = write(tmp_path / "full.jsonl", [row(0, True, {**FULL, "max_new_tokens": 1}),
                                           row(0, True, {**FULL, "max_new_tokens": 3})])
    monkeypatch.setattr(sys, "argv", ["summarize.py", "--full", full, full])
    with pytest.raises(SystemExit) as e:
        summarize.main()
    assert e.value.code == 2
