import importlib.util
import os

import torch

# No GPU -> run Triton kernels through the interpreter so they are still exercised.
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_script(relpath: str):
    """Import a file under scripts/ or benchmarks/ as a module (those folders aren't packages)."""
    name = os.path.splitext(os.path.basename(relpath))[0]
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
