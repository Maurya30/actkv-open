import os

import pytest
from packaging.requirements import Requirement

from conftest import ROOT


def test_triton_extra_installs_only_on_linux():
    """Triton ships Linux wheels only; without a marker `pip install .[triton]` fails on macOS."""
    tomllib = pytest.importorskip("tomllib", reason="tomllib is in the standard library from Python 3.11")
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        extra = tomllib.load(f)["project"]["optional-dependencies"]["triton"]
    (req,) = [Requirement(r) for r in extra]
    assert req.name == "triton" and req.marker is not None
    assert req.marker.evaluate({"sys_platform": "linux"})
    assert not req.marker.evaluate({"sys_platform": "darwin"})
