import json
import sys

import pytest

from conftest import load_script
from test_agent_e2e import CharTok, tiny

BAD_ARGS = [
    ["--method", "actkv"],                                     # no budget mode
    ["--method", "actkv", "--adaptive", "--budget", "64"],     # two budget modes
    ["--method", "full", "--budget", "64"],                    # FullKV never evicts
    ["--method", "actkv", "--budget", "64", "--bidirectional"],  # bidirectional needs --adaptive
    ["--method", "actkv", "--budget", "0"],
    ["--method", "snapkv", "--budget-ratio", "0.1"],           # ratio needs FullKV results
    ["--method", "snapkv", "--budget-ratio", "0.1", "--full-results", "MISSING"],
    ["--method", "snapkv", "--budget-ratio", "1.5", "--full-results", "FULL"],
]


def _full_file(tmp_path, task=0):
    p = tmp_path / "full.jsonl"
    p.write_text(json.dumps({"task": task, "run": 0, "method": "full", "trace_len": 1000,
                             "protected": 200, "ora_lens": [100, 100]}) + "\n")
    return p


@pytest.mark.parametrize("argv", BAD_ARGS)
def test_bad_args_fail_before_the_model_loads(argv, tmp_path, monkeypatch):
    run_eval = load_script("scripts/run_eval.py")
    monkeypatch.setattr(run_eval, "load_model", lambda *a, **k: pytest.fail("model was loaded"))
    full = _full_file(tmp_path)
    argv = [str(full) if a == "FULL" else str(tmp_path / "nope.jsonl") if a == "MISSING" else a
            for a in argv]
    out = tmp_path / "res" / "out.jsonl"
    monkeypatch.setattr(sys, "argv", ["run_eval.py", *argv, "--out", str(out)])
    with pytest.raises(SystemExit) as e:
        run_eval.main()
    assert e.value.code == 2
    assert not out.exists()


def test_no_output_file_when_no_episode_runs(tmp_path, monkeypatch):
    run_eval = load_script("scripts/run_eval.py")
    monkeypatch.setattr(run_eval, "load_model", lambda *a, **k: (tiny(), CharTok()))
    full = _full_file(tmp_path, task=99)          # no FullKV length for task 0 -> task is skipped
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--method", "snapkv", "--budget-ratio", "0.1",
                                      "--full-results", str(full), "--n-tasks", "1", "--device", "cpu",
                                      "--out", str(out)])
    run_eval.main()
    assert not out.exists()
