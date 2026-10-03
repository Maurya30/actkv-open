import os

import torch

# No GPU -> run Triton kernels through the interpreter so they are still exercised.
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
