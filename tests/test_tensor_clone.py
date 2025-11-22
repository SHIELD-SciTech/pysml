import pathlib
import sys
import threading
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pysml.dtype import bf16
from pysml.tensor import Tensor

try:
    import cupy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None
    cuda_backend = None
else:
    try:
        from pysml import cuda as cuda_backend
    except Exception:  # pragma: no cover
        cuda_backend = None


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

    def convert(self, data, dtype=None, device=None, shape=None):
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


@pytest.mark.skipif(cp is None or cuda_backend is None, reason="CuPy not available")
def test_clone_preserves_pointer_backed_cuda_data():
    # Create a raw CUDA allocation without array semantics
    mem = cp.cuda.alloc(cp.dtype(cp.float16).itemsize * 2)
    view = cp.ndarray((2,), dtype=cp.float16, memptr=mem)
    view[...] = cp.asarray([1.0, 2.0], dtype=cp.float16)

    tensor = Tensor.__new__(Tensor)
    tensor._backend = cuda_backend
    tensor._dtype = bf16()
    tensor._grad = None
    tensor._requires_grad = False
    tensor._grad_fn = None
    tensor._version = 0
    tensor._post_backward_hooks = []
    tensor._grad_lock = threading.Lock()
    tensor._freed = False

    try:
        device_id = cp.cuda.Device().id
    except Exception:
        device_id = 0

    tensor.device = device_id
    tensor.active_device = f"cuda:{device_id}"
    tensor._shape = (2,)
    tensor.data = mem

    cloned = tensor.clone()

    assert getattr(cloned.data, "shape", None) == (2,)
    assert cp.allclose(cloned.data, view)
    assert tensor.shape == (2,)


@pytest.mark.skipif(cp is None or cuda_backend is None, reason="CuPy not available")
def test_clone_infers_shape_for_pointer_data_when_missing():
    mem = cp.cuda.alloc(cp.dtype(cp.float16).itemsize * 2)
    view = cp.ndarray((2,), dtype=cp.float16, memptr=mem)
    view[...] = cp.asarray([3.0, 4.0], dtype=cp.float16)

    tensor = Tensor.__new__(Tensor)
    tensor._backend = cuda_backend
    tensor._dtype = bf16()
    tensor._grad = None
    tensor._requires_grad = False
    tensor._grad_fn = None
    tensor._version = 0
    tensor._post_backward_hooks = []
    tensor._grad_lock = threading.Lock()
    tensor._freed = False

    try:
        device_id = cp.cuda.Device().id
    except Exception:
        device_id = 0

    tensor.device = device_id
    tensor.active_device = f"cuda:{device_id}"
    tensor._shape = None
    tensor.data = mem

    cloned = tensor.clone()

    assert getattr(cloned.data, "shape", None) == (2,)
    assert cp.allclose(cloned.data, view)


if __name__ == "__main__":
    pytest.main([__file__])
