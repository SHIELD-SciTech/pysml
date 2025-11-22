import numpy as np

from pysml.tensor import Tensor


def test_tensor_initialization_sets_shape_slot():
    tensor = Tensor(np.ones((2, 3)), requires_grad=False)

    # Shape should be available even when backend payloads do not expose it.
    assert tensor._shape == (2, 3)
    assert tensor.shape == (2, 3)


def test_tensor_allows_future_metadata_attributes():
    tensor = Tensor(np.array([1.0]), requires_grad=False)

    # __dict__ in __slots__ ensures forward compatibility for helper attrs.
    tensor.extra_meta = "ok"
    assert tensor.extra_meta == "ok"
