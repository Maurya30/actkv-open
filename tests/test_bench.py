from conftest import load_script


def test_both_paths_keep_the_same_tokens_and_end_in_the_same_state():
    bench = load_script("benchmarks/bench_kernels.py")
    rows = bench.main(["--device", "cpu", "--small", "--iters", "1"])
    assert rows and all(r["masks_match"] and r["outputs_match"] for r in rows)
