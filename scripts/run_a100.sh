#!/usr/bin/env bash
# The whole Part A GPU run: environment info, the test suite with real Triton, the kernel benchmark,
# the HF eval sweeps on Qwen3-8B (vault, then ALFWorld if it installs cleanly), the summary tables
# and plots, and the README results section. Sized for about 4 hours on one A100 80GB; see RUNPOD.md.
#
#   bash scripts/run_a100.sh             # on the GPU machine
#   bash scripts/run_a100.sh --dry-run   # tiny random model on CPU, in a temp dir that is deleted after
#
# Resumable: every finished step leaves results/.done/<step>, and every episode is appended to its
# results file as soon as it ends, so after a preemption just run the script again. The sweeps stop
# starting new tasks before their hour budget runs out, and all configs always have the same tasks.
#
# Knobs (environment variables): MODEL (Qwen/Qwen3-8B), N_TASKS (50, an upper bound), HOURS_VAULT
# (2.75), HOURS_ALFWORLD (0.75, given to vault instead if ALFWorld doesn't install), ALFWORLD=0 to
# skip ALFWorld, PYTHON (python).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
MODEL=${MODEL:-Qwen/Qwen3-8B}
N_TASKS=${N_TASKS:-50}
HOURS_VAULT=${HOURS_VAULT:-2.75}
HOURS_ALFWORLD=${HOURS_ALFWORLD:-0.75}
ALFWORLD=${ALFWORLD:-1}
RESULTS=results
EVAL=(--model "$MODEL" --runs 1 --no-thinking --max-new-tokens 1024)
BENCH=(--device cuda)
DRY=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY=1
  TMP=$(mktemp -d)
  trap 'rm -rf "$TMP"; echo "== dry run: deleted $TMP"' EXIT
  RESULTS=$TMP/results
  "$PY" scripts/make_tiny_model.py "$TMP/tiny" > /dev/null 2>&1
  EVAL=(--model "$TMP/tiny" --dtype float32 --device cpu --runs 1 --no-thinking --max-new-tokens 8)
  BENCH=(--device cpu --small --iters 1)
  N_TASKS=2 HOURS_VAULT=1 ALFWORLD=0
fi
mkdir -p "$RESULTS/.done"

T0=$(date +%s)
elapsed() { local s=$(( $(date +%s) - T0 )); printf '%dh%02dm' $((s / 3600)) $((s % 3600 / 60)); }
step() {   # step NAME CMD...: run CMD once; a marker file records that it finished
  local name=$1; shift
  if [[ -f "$RESULTS/.done/$name" ]]; then echo "== $name: finished in an earlier run, skipping"; return; fi
  echo "== $name (+$(elapsed))"
  "$@"
  touch "$RESULTS/.done/$name"
  echo "== $name done (+$(elapsed))"
}

setup() {
  "$PY" -m pip install -q -e ".[triton,dev,eval]"
  "$PY" -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU visible'"
}

env_info() {
  "$PY" - "$RESULTS/env.json" <<'EOF'
import datetime, json, platform, subprocess, sys
import torch, transformers
try:
    import triton
    triton_version = triton.__version__
except ImportError:
    triton_version = None
cuda = torch.cuda.is_available()
def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip() or None
    except FileNotFoundError:
        return None
info = {"device": "cuda" if cuda else "cpu", "gpu": torch.cuda.get_device_name() if cuda else None,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if cuda else None,
        "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "triton": triton_version, "transformers": transformers.__version__,
        "git_sha": run(["git", "rev-parse", "HEAD"]), "git_dirty": bool(run(["git", "status", "--porcelain"])),
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat()}
json.dump(info, open(sys.argv[1], "w"), indent=2)
print(json.dumps(info, indent=2))
EOF
}

alfworld_setup() {   # optional: a failed install or download only skips the ALFWorld sweep
  if "$PY" -m pip install -q -e ".[alfworld]" > "$RESULTS/alfworld_install.log" 2>&1 &&
     alfworld-download >> "$RESULTS/alfworld_install.log" 2>&1; then
    echo ok > "$RESULTS/alfworld_status"
  else
    echo "failed, see alfworld_install.log" > "$RESULTS/alfworld_status"
  fi
  echo "ALFWorld: $(cat "$RESULTS/alfworld_status")"
}

tests() {
  "$PY" -m pytest -v -rs -p no:cacheprovider 2>&1 | tee "$RESULTS/pytest.txt"
  if (( ! DRY )); then   # on the GPU the kernels must really run, not fall back to the torch references
    "$PY" -c "from actkv.kernels import compaction, lse_attn; assert lse_attn.HAS_TRITON and compaction.HAS_TRITON"
    if grep -E "SKIPPED.*test_kernels" "$RESULTS/pytest.txt"; then echo "kernel tests skipped on the GPU" >&2; return 1; fi
  fi
}

bench() {
  "$PY" benchmarks/bench_kernels.py "${BENCH[@]}" --out "$RESULTS/bench_kernels.json" | tee "$RESULTS/bench_kernels.txt"
}

sweep() {   # sweep ENV HOURS N_TASKS: progress, time used and ETA are printed after every task
  mkdir -p "$RESULTS/$1"
  "$PY" scripts/run_eval.py --env "$1" --sweep "configs/$1_sweep.json" --out-dir "$RESULTS/$1" \
      --n-tasks "$3" --hours "$2" "${EVAL[@]}" 2>&1 | tee -a "$RESULTS/$1/sweep.log"
}

summarize() {   # cheap, so it runs every time: tables with sample sizes and CIs, plots, summary.json
  for d in "$RESULTS/vault" "$RESULTS/alfworld"; do
    [[ -f "$d/full.jsonl" ]] || continue
    echo "== summary: $d"
    "$PY" scripts/summarize.py --full "$d/full.jsonl" "$d"/*.jsonl --plot "$d/pareto.png" \
        --json "$d/summary.json" | tee "$d/summary.txt"
  done
}

(( DRY )) || step setup setup
step env_info env_info
(( ALFWORLD )) && step alfworld_setup alfworld_setup
step tests tests
step bench bench
ALF_OK=0
[[ "$ALFWORLD" == 1 && "$(cat "$RESULTS/alfworld_status" 2>/dev/null)" == ok ]] && ALF_OK=1
(( ALF_OK )) || HOURS_VAULT=$("$PY" -c "print($HOURS_VAULT + $HOURS_ALFWORLD)")
echo "== plan: vault $(grep -c '"name"' configs/vault_sweep.json) configs x up to $N_TASKS tasks in ${HOURS_VAULT}h" \
     "$( (( ALF_OK )) && echo "; ALFWorld $(grep -c '"name"' configs/alfworld_sweep.json) configs in ${HOURS_ALFWORLD}h")"
step vault sweep vault "$HOURS_VAULT" "$N_TASKS"
(( ALF_OK )) && step alfworld sweep alfworld "$HOURS_ALFWORLD" 134
summarize
if (( DRY )); then
  if "$PY" scripts/update_readme.py --results "$RESULTS" --check; then echo "CPU results were accepted" >&2; exit 1; fi
  echo "== update_readme refused the CPU dry-run results, as it should"
else
  "$PY" scripts/update_readme.py --results "$RESULTS"
fi
echo "== all done (+$(elapsed)). Commit results/ and README.md as described in RUNPOD.md."
