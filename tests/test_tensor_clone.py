import types

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pysml.tensor import Tensor


class _ShapeLessPointer:
    def __init__(self, value):
        self.value = value


class _StubBackend:
    BACKEND_NAME = "cuda"

    def __init__(self):
        self.copied = False
        self.converted = False

    def copy(self, data):
        self.copied = True
        raise TypeError("cannot copy raw pointer")

    def convert(self, data, dtype=None, device=None):
        self.converted = True
        # emulate array-like return with shape attribute
        return types.SimpleNamespace(shape=(1,), data=data)

    def copyto(self, dst, src):
        raise AssertionError("copyto should not be called when shape is missing")


def test_clone_handles_shapeless_data():
    backend = _StubBackend()

    tensor = Tensor.__new__(Tensor)
    tensor._backend = backend
    tensor._dtype = None
    tensor._grad = None
    tensor._requires_grad = False
    tensor._grad_fn = None
    tensor.device = "cuda:0"
    tensor.active_device = "cuda:0"
    tensor._version = 0
    tensor.data = _ShapeLessPointer(42)

    cloned = tensor.clone()

    assert backend.converted is True
    assert getattr(cloned.data, "shape", None) == (1,)


if __name__ == "__main__":
    pytest.main([__file__])
