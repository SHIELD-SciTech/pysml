import numpy as np
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pysml import Tensor
from pysml.nn.embedding import Embedding
from pysml.nn.module import Parameter


def test_embedding_casts_float_indices_and_backward():
    weights = Tensor(
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        requires_grad=True,
    )
    emb = Embedding(3, 2)
    emb.weight = Parameter(weights, requires_grad=True)

    float_indices = Tensor(np.array([[0.0, 2.0]], dtype=np.float32))

    output = emb(float_indices)
    expected = np.array([[1.0, 2.0], [5.0, 6.0]], dtype=np.float32)
    assert np.allclose(output.numpy(), expected)

    total = output.sum()
    total.backward()

    grad = emb.weight.grad.numpy()
    expected_grad = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    assert np.allclose(grad, expected_grad)
