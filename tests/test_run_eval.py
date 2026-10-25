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


def _stub_episode(session, env, fmt, task, *a, **k):
    return {"task": task, "success": False, "rounds": 1, "invalid": 0, "peak_entries": 1,
            "trace_len": 1, "protected": 0, "compressions": 0, "budgets": [], "ora_lens": [1],
            "seconds": 0.0, "transcript": []}


def test_resume_keys_on_the_full_config(tmp_path, monkeypatch):
    """A run that differs only in --slack must not be skipped as 'already done'."""
    run_eval = load_script("scripts/run_eval.py")
    monkeypatch.setattr(run_eval, "load_model", lambda *a, **k: (tiny(), CharTok()))
    monkeypatch.setattr(run_eval, "run_episode", _stub_episode)
    out = tmp_path / "out.jsonl"
    for slack in ("32", "64", "64"):          # the repeated config is the only one to skip
        monkeypatch.setattr(sys, "argv", ["run_eval.py", "--method", "snapkv", "--budget", "64",
                                          "--slack", slack, "--n-tasks", "1", "--runs", "1",
                                          "--device", "cpu", "--out", str(out)])
        run_eval.main()
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["config"]["slack"] for r in rows] == [32, 64]
    assert rows[0]["config_id"] != rows[1]["config_id"]


def test_no_thinking_uses_qwen_non_thinking_sampling(tmp_path, monkeypatch):
    run_eval = load_script("scripts/run_eval.py")
    monkeypatch.setattr(run_eval, "load_model", lambda *a, **k: (tiny(), CharTok()))
    monkeypatch.setattr(run_eval, "run_episode", _stub_episode)
    seen = []
    real = run_eval.ActKVSession
    monkeypatch.setattr(run_eval, "ActKVSession", lambda *a, **k: seen.append(k["sampling"]) or real(*a, **k))
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--method", "full", "--no-thinking", "--n-tasks", "1",
                                      "--runs", "1", "--device", "cpu", "--out", str(tmp_path / "o.jsonl")])
    run_eval.main()
    sp = seen[0]
    assert (sp.temperature, sp.top_p, sp.top_k) == (0.7, 0.8, 20)


def test_ratio_budget_base_excludes_the_protected_prompt(tmp_path):
    """Budgets exclude the protected prefix, so the ratio base must too."""
    run_eval = load_script("scripts/run_eval.py")
    assert run_eval.full_lengths(_full_file(tmp_path)) == {0: 800}     # trace_len 1000 - protected 200


def _write(path, obj):
    path.write_text(json.dumps(obj))
    return str(path)


def test_sweep_runs_task_major_and_stops_before_the_time_budget(tmp_path, monkeypatch):
    run_eval = load_script("scripts/run_eval.py")
    monkeypatch.setattr(run_eval, "load_model", lambda *a, **k: (tiny(), CharTok()))
    budgets = []

    def episode(session, env, fmt, task, *a, **k):     # every episode "takes" 10 s
        budgets.append(session.budget.current if session.budget else None)
        return {**_stub_episode(session, env, fmt, task), "trace_len": 300, "protected": 100, "seconds": 10.0}

    monkeypatch.setattr(run_eval, "run_episode", episode)
    sweep = _write(tmp_path / "sweep.json", [
        {"name": "full", "method": "full"},
        {"name": "stream_r0.5", "method": "streaming", "budget_ratio": 0.5, "full_results": "full.jsonl"}])
    out = tmp_path / "res"
    argv = ["run_eval.py", "--sweep", sweep, "--out-dir", str(out), "--n-tasks", "5", "--runs", "1",
            "--device", "cpu", "--hours"]

    def tasks(name):
        return [json.loads(line)["task"] for line in (out / f"{name}.jsonl").read_text().splitlines()]

    monkeypatch.setattr(sys, "argv", argv + [str(45 / 3600)])   # 20 s per task -> a third would end at 60 s
    run_eval.main()
    assert tasks("full") == tasks("stream_r0.5") == [0, 1]
    assert budgets == [None, 100, None, 100]                     # 0.5 * (300 - 100) from that task's FullKV
    monkeypatch.setattr(sys, "argv", argv + [str(85 / 3600)])   # resume: 40 s used, room for two more tasks
    run_eval.main()
    assert tasks("full") == tasks("stream_r0.5") == [0, 1, 2, 3]


def test_baseline_slack_is_that_tasks_mean_fullkv_round_length(tmp_path, monkeypatch):
    run_eval = load_script("scripts/run_eval.py")
    monkeypatch.setattr(run_eval, "load_model", lambda *a, **k: (tiny(), CharTok()))
    slacks = []
    monkeypatch.setattr(run_eval, "run_episode",
                        lambda session, *a, **k: slacks.append(session.slack) or _stub_episode(session, *a, **k))
    full = tmp_path / "full.jsonl"
    full.write_text("".join(json.dumps({"task": t, "run": 0, "method": "full", "trace_len": 9, "ora_lens": o}) + "\n"
                            for t, o in [(0, [10, 30]), (1, [100])]))
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--method", "snapkv", "--budget", "64", "--full-results",
                                      str(full), "--n-tasks", "2", "--runs", "1", "--device", "cpu",
                                      "--out", str(tmp_path / "o.jsonl")])
    run_eval.main()
    assert slacks == [20, 100]


def test_resumed_sweep_finishes_the_partial_task_then_respects_the_budget(tmp_path, monkeypatch):
    run_eval = load_script("scripts/run_eval.py")
    monkeypatch.setattr(run_eval, "load_model", lambda *a, **k: (tiny(), CharTok()))
    calls = []

    def episode(session, env, fmt, task, *a, **k):        # every episode "takes" 100 s
        calls.append(task)
        if crash_after and len(calls) > crash_after:
            raise KeyboardInterrupt("pod preempted")
        return {**_stub_episode(session, env, fmt, task), "seconds": 100.0}

    monkeypatch.setattr(run_eval, "run_episode", episode)
    sweep = _write(tmp_path / "sweep.json", [{"name": "full", "method": "full"},
                                             {"name": "stream", "method": "streaming", "budget": 64}])
    out = tmp_path / "res"

    def run(hours):
        monkeypatch.setattr(sys, "argv", ["run_eval.py", "--sweep", sweep, "--out-dir", str(out), "--n-tasks", "5",
                                          "--runs", "1", "--device", "cpu", "--hours", str(hours / 3600)])
        run_eval.main()

    def tasks(name):
        return [json.loads(line)["task"] for line in (out / f"{name}.jsonl").read_text().splitlines()]

    crash_after = 3                       # preempted after task 0 (both configs) and task 1 (full only)
    with pytest.raises(KeyboardInterrupt):
        run(520)
    crash_after = None
    run(520)                              # finish task 1, then 400 s used + 200 s/task > 520 s: stop
    assert tasks("full") == tasks("stream") == [0, 1]
    run(180)                              # budget already spent: no new task starts
    assert tasks("full") == tasks("stream") == [0, 1]


def test_sweep_ratio_budget_uses_all_fullkv_runs_of_the_task(tmp_path, monkeypatch):
    """With --runs 2, every run of a ratio config gets ratio * median over both FullKV runs."""
    run_eval = load_script("scripts/run_eval.py")
    monkeypatch.setattr(run_eval, "load_model", lambda *a, **k: (tiny(), CharTok()))
    budgets, full_trace = [], iter([300, 500])

    def episode(session, env, fmt, task, *a, **k):
        res = _stub_episode(session, env, fmt, task)
        if session.budget is None:
            return {**res, "trace_len": next(full_trace), "protected": 100}
        budgets.append(session.budget.current)
        return res

    monkeypatch.setattr(run_eval, "run_episode", episode)
    sweep = _write(tmp_path / "sweep.json", [
        {"name": "full", "method": "full"},
        {"name": "stream_r0.5", "method": "streaming", "budget_ratio": 0.5, "full_results": "full.jsonl"}])
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--sweep", sweep, "--out-dir", str(tmp_path / "res"),
                                      "--n-tasks", "1", "--runs", "2", "--device", "cpu"])
    run_eval.main()
    assert budgets == [150, 150]          # 0.5 * median(300 - 100, 500 - 100)


def test_ratio_budget_uses_the_fullkv_run_with_the_same_settings(tmp_path, monkeypatch):
    """A results file can hold FullKV runs of several settings; budgets must come from the matching one."""
    run_eval = load_script("scripts/run_eval.py")
    monkeypatch.setattr(run_eval, "load_model", lambda *a, **k: (tiny(), CharTok()))
    budgets = []

    def episode(session, env, fmt, task, max_new_tokens, *a):
        if session.budget is not None:
            budgets.append(session.budget.current)
        return {**_stub_episode(session, env, fmt, task), "trace_len": 1000 * max_new_tokens + 100, "protected": 100}

    monkeypatch.setattr(run_eval, "run_episode", episode)
    full = str(tmp_path / "full.jsonl")
    common = ["--n-tasks", "1", "--runs", "1", "--device", "cpu"]
    for argv in (["--method", "full", "--max-new-tokens", "1", "--out", full],
                 ["--method", "full", "--max-new-tokens", "3", "--out", full],
                 ["--method", "streaming", "--budget-ratio", "0.2", "--full-results", full,
                  "--max-new-tokens", "3", "--out", str(tmp_path / "s.jsonl")]):
        monkeypatch.setattr(sys, "argv", ["run_eval.py", *argv, *common])
        run_eval.main()
    assert budgets == [600]               # 0.2 * (3100 - 100), not the median over both FullKV settings


def test_config_id_does_not_depend_on_how_the_fullkv_path_is_spelled():
    import argparse
    run_eval = load_script("scripts/run_eval.py")
    base = dict(model="m", dtype="bfloat16", env="vault", method="streaming", adaptive=False, bidirectional=False,
                budget=None, budget_ratio=0.1, slack=None, decay=0.5, top_p_hit=0.9, head_reduce="mean",
                max_new_tokens=8, greedy=False, no_thinking=True)
    ids = {run_eval.config_id(run_eval.run_config(argparse.Namespace(**base, full_results=p)))
           for p in ("results/vault/full.jsonl", "./results/vault/full.jsonl", "results//vault/full.jsonl")}
    assert len(ids) == 1
