import pytest

np = pytest.importorskip("numpy")

from pysml.examples import upscaling


def test_upscaling_demo_shapes():
    output = upscaling.run_demo(scale_factor=3, preferred_device="cpu")
    assert output.shape == (1, 3, 96, 96)
