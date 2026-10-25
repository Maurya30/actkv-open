from conftest import load_script


def test_both_paths_keep_the_same_tokens_and_end_in_the_same_state():
    bench = load_script("benchmarks/bench_kernels.py")
    rows = bench.main(["--device", "cpu", "--small", "--iters", "1"])
    assert rows and all(r["masks_match"] and r["outputs_match"] for r in rows)


def test_memory_is_measured_or_explicitly_missing():
    bench = load_script("benchmarks/bench_kernels.py")
    (row, *_) = bench.main(["--device", "cpu", "--small", "--iters", "1", "--reqs", "1"])
    assert row["actkv_extra_mb"] is None and row["gather_extra_mb"] is None   # no peak counter on CPU
    assert "not measured" in row["memory"]
