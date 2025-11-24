import numpy as np

import pysml
from pysml.tensor import Tensor
from pysml import engine


class DummyArray(np.ndarray):
    pass


class DummyBackend:
    BACKEND_NAME = "dummy"
    DEVICE_TYPE = "dummy"
    ndarray = DummyArray

    def __init__(self):
        self.seen_types = None

    def multiply(self, a, b, out=None):
        self.seen_types = (type(a), type(b))
        result = np.multiply(a, b)
        if out is not None:
            np.copyto(out, result)
            return out
        return result.view(DummyArray)

    def asarray(self, array):
        return np.asarray(array).view(DummyArray)

    def floor_divide(self, a, b, out=None):
        self.seen_types = (type(a), type(b))
        result = np.floor_divide(a, b)
        if out is not None:
            np.copyto(out, result)
            return out
        return result.view(DummyArray)

    def maximum(self, a, b, out=None):
        self.seen_types = (type(a), type(b))
        result = np.maximum(a, b)
        if out is not None:
            np.copyto(out, result)
            return out
        return result.view(DummyArray)

    def where(self, condition, x, y):
        self.seen_types = (type(condition), type(x), type(y))
        return np.where(condition, x, y).view(DummyArray)

    def equal(self, a, b):
        self.seen_types = (type(a), type(b))
        return np.equal(a, b)

    # Minimal stubs required by engine helpers
    def copyto(self, dst, src):
        np.copyto(dst, src)

    def get_device(self):
        return "dummy"

    float32 = np.float32
    float64 = np.float64
    int32 = np.int32
    int64 = np.int64
    bool = np.bool_
    complex64 = np.complex64
    complex128 = np.complex128


def _tensor_with_backend(array, backend):
    tensor = Tensor.__new__(Tensor)
    tensor._requires_grad = False
    tensor._grad = None
    tensor._grad_fn = None
    tensor._dtype = np.float32
    tensor._backend = backend
    tensor.device = None
    tensor.active_device = None
    tensor._version = 0
    tensor._shape = getattr(array, "shape", None)
    tensor.data = array
    return tensor


def test_numpy_other_converts_to_backend_array():
    backend = DummyBackend()
    lhs = _tensor_with_backend(backend.asarray([1.0, 2.0]), backend)
    other = np.array([3.0, 4.0])

    result = engine.multiply(lhs, other)

    assert backend.seen_types == (DummyArray, DummyArray)
    assert isinstance(result.data, DummyArray)
    np.testing.assert_array_equal(result.data, np.array([3.0, 8.0]))


def test_numpy_other_converts_for_floor_divide():
    backend = DummyBackend()
    lhs = _tensor_with_backend(backend.asarray([4.0, 9.0]), backend)
    other = np.array([2.0, 3.0])

    result = engine.floor_divide(lhs, other)

    assert backend.seen_types == (DummyArray, DummyArray)
    assert isinstance(result.data, DummyArray)
    np.testing.assert_array_equal(result.data, np.array([2.0, 3.0]))


def test_numpy_other_converts_for_maximum():
    backend = DummyBackend()
    lhs = _tensor_with_backend(backend.asarray([1.0, 5.0]), backend)
    other = np.array([3.0, 2.0])

    result = engine.maximum(lhs, other)

    assert backend.seen_types == (DummyArray, DummyArray)
    assert isinstance(result.data, DummyArray)
    np.testing.assert_array_equal(result.data, np.array([3.0, 5.0]))


def test_numpy_other_converts_for_where_and_comparisons():
    backend = DummyBackend()
    condition = np.array([True, False])
    x = _tensor_with_backend(backend.asarray([1.0, 2.0]), backend)
    y = np.array([10.0, 20.0])

    where_result = engine.where(condition, x, y)
    assert backend.seen_types == (DummyArray, DummyArray, DummyArray)
    assert isinstance(where_result.data, DummyArray)
    np.testing.assert_array_equal(where_result.data, np.array([1.0, 20.0]))

    eq_result = engine.equal(x, y)
    assert backend.seen_types == (DummyArray, DummyArray)
    assert isinstance(eq_result, np.ndarray)
    np.testing.assert_array_equal(eq_result, np.array([False, False]))
