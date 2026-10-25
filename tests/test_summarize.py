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
