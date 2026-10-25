import json

from conftest import load_script

README = "# x\n\nintro\n\n<!-- results:begin -->\nNo GPU results yet.\n<!-- results:end -->\n\n## after\n"
ENV = {"device": "cuda", "gpu": "NVIDIA A100-SXM4-80GB", "torch": "2.9", "cuda": "12.8", "triton": "3.5",
       "transformers": "4.57", "git_sha": "abcdef1234", "date": "2026-10-26T00:00:00+00:00"}
CONFIG = {"name": "actkv (adaptive)", "n_tasks": 23, "n_episodes": 23, "acc": 0.61, "acc_ci": [0.4, 0.8], "delta_vs_full": -0.04,
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
    assert "| actkv (adaptive) | 23 | 23 | 0.610 [0.40, 0.80] | -0.040 [-0.20, 0.10] | 30% | 100% |" in text
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


def test_episode_counts_come_from_the_data(tmp_path):
    update = load_script("scripts/update_readme.py")
    res, readme = fixture(tmp_path)
    summary = json.loads((res / "vault" / "summary.json").read_text())
    summary["configs"][0]["n_episodes"] = 46                     # two runs per task
    (res / "vault" / "summary.json").write_text(json.dumps(summary))
    assert update.main(["--results", str(res), "--readme", str(readme)]) == 0
    text = readme.read_text()
    assert "| actkv (adaptive) | 23 | 46 |" in text and "one run each" not in text


BENCH = {"kernel_path": "triton", "note": "compiled Triton kernels", "device": "cuda", "gpu": "NVIDIA A100-SXM4-80GB",
         "dtype": "float16", "torch": "2.9", "triton": "3.5", "n_layers": 1, "block_size": 16, "kv_heads": 8,
         "q_heads": 32, "head_dim": 128, "iters": 5, "warmup": 1,
         "rows": [{"n": 1024, "budget": 512, "r": 4, "actkv_ms": 2.0, "gather_ms": 3.0, "actkv_extra_mb": 1.5,
                   "gather_extra_mb": 9.0, "masks_match": True, "outputs_match": True}]}


def test_benchmark_table_is_rendered_from_a_compiled_triton_run(tmp_path):
    update = load_script("scripts/update_readme.py")
    res, readme = fixture(tmp_path)
    (res / "bench_kernels.json").write_text(json.dumps(BENCH))
    assert update.main(["--results", str(res), "--readme", str(readme)]) == 0
    assert "| 1024 → 512 | 4 | 2.000 | 3.000 | 1.50x | 1.5 | 9.0 | yes |" in readme.read_text()


def test_a_benchmark_that_did_not_run_compiled_triton_is_refused(tmp_path):
    update = load_script("scripts/update_readme.py")
    res, readme = fixture(tmp_path)                       # env.json and episodes are CUDA
    (res / "bench_kernels.json").write_text(json.dumps({**BENCH, "kernel_path": "torch-fallback"}))
    assert update.main(["--results", str(res), "--readme", str(readme)]) == 1
    assert readme.read_text() == README


def test_a_config_with_any_non_cuda_episode_is_refused(tmp_path):
    update = load_script("scripts/update_readme.py")
    res, readme = fixture(tmp_path)                       # env.json is CUDA
    summary = json.loads((res / "vault" / "summary.json").read_text())
    summary["configs"][0]["devices"] = ["cpu", "cuda"]
    (res / "vault" / "summary.json").write_text(json.dumps(summary))
    assert update.main(["--results", str(res), "--readme", str(readme)]) == 1
    assert readme.read_text() == README
