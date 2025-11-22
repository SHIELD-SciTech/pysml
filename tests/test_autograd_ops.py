import pytest

np = pytest.importorskip("numpy")

import pysml
from pysml import Tensor


@pytest.mark.parametrize(
    "axis, keepdims",
    [
        (1, False),
        (0, True),
        ((0, 1), False),
    ],
)
def test_sum_backward_respects_axis_and_keepdims(axis, keepdims):
    data = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    tensor = Tensor(data, requires_grad=True)

    reduced = pysml.sum(tensor, axis=axis, keepdims=keepdims)
    upstream = np.ones_like(reduced.data)
    reduced.backward(Tensor(upstream))

    assert tensor.grad is not None
    assert np.allclose(tensor.grad.data, np.ones_like(data))


def test_mean_backward_scales_by_element_count():
    values = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    tensor = Tensor(values, requires_grad=True)

    reduced = pysml.mean(tensor, axis=0, keepdims=False)
    upstream = np.ones_like(reduced.data)
    reduced.backward(Tensor(upstream))

    assert tensor.grad is not None
    expected = np.full_like(values, 0.5)
    assert np.allclose(tensor.grad.data, expected)


def test_mean_backward_with_axis_and_no_keepdims_broadcasts_upstream():
    values = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float32)
    tensor = Tensor(values, requires_grad=True)

    reduced = pysml.mean(tensor, axis=1, keepdims=False)
    upstream = np.array([2.0, 4.0], dtype=np.float32)
    reduced.backward(Tensor(upstream))

    assert tensor.grad is not None
    expected = np.array([
        [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
        [4.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0],
    ], dtype=np.float32)
    assert np.allclose(tensor.grad.data, expected)


def test_broadcast_multiply_backward_reduces_expanded_axes():
    left = Tensor(np.ones((2, 3), dtype=np.float32), requires_grad=True)
    right = Tensor(np.array([[2.0, 3.0, 4.0]], dtype=np.float32), requires_grad=True)

    loss = pysml.sum(pysml.multiply(left, right))
    loss.backward()

    assert left.grad is not None
    assert right.grad is not None
    np.testing.assert_allclose(left.grad.data, np.array([[2.0, 3.0, 4.0]] * 2))
    np.testing.assert_allclose(right.grad.data, np.array([[2.0, 2.0, 2.0]]))
