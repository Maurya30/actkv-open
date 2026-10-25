import pytest
import torch

from actkv.hf.session import SamplingParams, sample

ACCEL = [d for d, ok in (("cuda", torch.cuda.is_available()), ("mps", torch.backends.mps.is_available())) if ok]


def test_seeded_sampling_is_reproducible():
    logits = torch.randn(500, generator=torch.Generator().manual_seed(0))
    draws = [sample(logits, SamplingParams(), torch.Generator().manual_seed(7)) for _ in range(2)]
    assert draws[0] == draws[1]


@pytest.mark.skipif(not ACCEL, reason="needs a CUDA or MPS device")
def test_seeded_sampling_works_with_logits_on_an_accelerator():
    """The session's generator lives on the CPU while the model's logits live on the GPU."""
    logits = torch.randn(500, generator=torch.Generator().manual_seed(0))
    sp = SamplingParams(top_k=20)
    t = sample(logits.to(ACCEL[0]), sp, torch.Generator().manual_seed(7))
    assert t in torch.topk(logits, 20).indices.tolist()
