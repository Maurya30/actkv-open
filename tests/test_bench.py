from conftest import DEVICE, load_script

# the device the bench accepts on this host: with CUDA, triton runs compiled and needs --device cuda
ARGS = ["--device", DEVICE, "--small", "--iters", "1"]


def test_both_paths_keep_the_same_tokens_and_end_in_the_same_state():
    bench = load_script("benchmarks/bench_kernels.py")
    rows = bench.main(ARGS)
    assert rows and all(r["masks_match"] and r["outputs_match"] for r in rows)


def test_memory_is_measured_or_explicitly_missing():
    bench = load_script("benchmarks/bench_kernels.py")
    (row, *_) = bench.main(ARGS + ["--reqs", "1"])
    if DEVICE == "cuda":
        assert row["actkv_extra_mb"] >= 0 and row["gather_extra_mb"] >= 0
    else:                                  # no peak-memory counter off CUDA
        assert row["actkv_extra_mb"] is None and row["gather_extra_mb"] is None
        assert "not measured" in row["memory"]


def test_json_output_says_which_kernel_path_ran(tmp_path):
    import json
    bench = load_script("benchmarks/bench_kernels.py")
    out = tmp_path / "bench.json"
    bench.main(ARGS + ["--reqs", "1", "--out", str(out)])
    data = json.loads(out.read_text())
    assert data["kernel_path"] == bench.kernel_path() and data["note"]
    assert data["device"] == DEVICE and len(data["rows"]) == 1
    if DEVICE != "cuda":
        assert data["kernel_path"] != "triton"     # never compiled Triton on a CPU
