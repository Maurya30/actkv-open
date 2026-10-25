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
