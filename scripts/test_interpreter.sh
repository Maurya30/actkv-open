#!/usr/bin/env bash
# Run the Triton kernels under TRITON_INTERPRET=1 in a Linux container and compare them with the
# torch references (tests/test_kernels.py). Triton only installs on Linux, so this is how the
# kernels' logic gets checked on a Mac. It checks correctness only; it says nothing about speed.
#
#   scripts/test_interpreter.sh          # needs Docker; IMAGE=... to override python:3.12-slim
set -euo pipefail
cd "$(dirname "$0")/.."

docker run --rm -v "$PWD":/src:ro "${IMAGE:-python:3.12-slim}" bash -euo pipefail -c '
  mkdir /work && cp -r /src/actkv /src/tests /src/pyproject.toml /work && cd /work
  pip install -q --root-user-action=ignore --index-url https://download.pytorch.org/whl/cpu torch
  pip install -q --root-user-action=ignore "triton>=3.0" pytest numpy
  python -c "import platform, torch, triton; print(\"linux\", platform.machine(), \"torch\", torch.__version__, \"triton\", triton.__version__)"
  # fail loudly if the kernels would silently fall back to the torch references
  python -c "from actkv.kernels import compaction, lse_attn; assert lse_attn.HAS_TRITON and compaction.HAS_TRITON"
  TRITON_INTERPRET=1 python -m pytest tests/test_kernels.py -v -rs -p no:cacheprovider -m "not gpu" | tee /tmp/out.txt
  if grep -q SKIPPED /tmp/out.txt; then echo "FAIL: kernel tests were skipped"; exit 1; fi
'
