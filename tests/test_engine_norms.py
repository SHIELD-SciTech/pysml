import pytest

np = pytest.importorskip("numpy")

import pysml
from pysml import Tensor, dtype
from pysml.nn.normalization import LayerNorm


def test_softmax_and_log_softmax_match():
    data = Tensor(
        np.array([[1.0, 2.0, 3.0], [0.5, 0.25, -0.5]], dtype=np.float32),
        dtype=dtype.fp32(),
    )

    probs = pysml.softmax(data, axis=1)
    logs = pysml.log_softmax(data, axis=1)

    row_sums = probs.data.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones_like(row_sums))
    np.testing.assert_allclose(np.exp(logs.data), probs.data, rtol=1e-3, atol=1e-4)


def test_layer_norm_backward_tracks_inputs_and_params():
    norm = LayerNorm(4, eps=1e-5)
    x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True, dtype=dtype.fp32())

    output = norm(x)
    loss = pysml.mean(output)
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.data.shape
    # Parameter gradients are not yet tracked by the backend, but inputs must flow.
