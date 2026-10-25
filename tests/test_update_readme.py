import json

from conftest import load_script

README = "# x\n\nintro\n\n<!-- results:begin -->\nNo GPU results yet.\n<!-- results:end -->\n\n## after\n"
ENV = {"device": "cuda", "gpu": "NVIDIA A100-SXM4-80GB", "torch": "2.9", "cuda": "12.8", "triton": "3.5",
       "transformers": "4.57", "git_sha": "abcdef1234", "date": "2026-10-26T00:00:00+00:00"}
CONFIG = {"name": "actkv (adaptive)", "n_tasks": 23, "acc": 0.61, "acc_ci": [0.4, 0.8], "delta_vs_full": -0.04,
          "delta_ci": [-0.2, 0.1], "peak_vs_full": 0.3, "trace_vs_full": 1.0, "devices": ["cuda"],
          "config": {"model": "Qwen/Qwen3-8B", "no_thinking": True, "max_new_tokens": 1024}}


def fixture(tmp_path, device="cuda"):
    res = tmp_path / "results"
    (res / "vault").mkdir(parents=True)
    (res / "env.json").write_text(json.dumps({**ENV, "device": device}))
    (res / "vault" / "summary.json").write_text(json.dumps({"configs": [{**CONFIG, "devices": [device]}]}))
    readme = tmp_path / "README.md"
    readme.write_text(README)
    return res, readme


def test_section_is_generated_from_gpu_results(tmp_path):
    update = load_script("scripts/update_readme.py")
    res, readme = fixture(tmp_path)
    assert update.main(["--results", str(res), "--readme", str(readme)]) == 0
    text = readme.read_text()
    assert "| actkv (adaptive) | 23 | 0.610 [0.40, 0.80] | -0.040 [-0.20, 0.10] | 30% | 100% |" in text
    assert "NVIDIA A100-SXM4-80GB" in text and "No GPU results yet" not in text
    assert text.startswith("# x\n\nintro\n\n<!-- results:begin -->") and text.endswith("<!-- results:end -->\n\n## after\n")


def test_cpu_results_are_refused(tmp_path, capsys):
    update = load_script("scripts/update_readme.py")
    res, readme = fixture(tmp_path, device="cpu")
    assert update.main(["--results", str(res), "--readme", str(readme)]) == 1
    assert readme.read_text() == README
    assert "refusing" in capsys.readouterr().err


def test_check_does_not_write(tmp_path):
    update = load_script("scripts/update_readme.py")
    res, readme = fixture(tmp_path)
    assert update.main(["--results", str(res), "--readme", str(readme), "--check"]) == 0
    assert readme.read_text() == README
