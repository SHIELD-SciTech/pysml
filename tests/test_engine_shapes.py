import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pysml.tensor import Tensor


def test_wrap_result_sets_shape():
    a = Tensor([1.0, 2.0])
    b = Tensor([3.0, 4.0])

    out = a + b

    assert getattr(out, "_shape", None) == (2,)
    assert out.shape == (2,)


def test_detach_preserves_shape():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    detached = x.detach()

    assert detached._requires_grad is False
    assert getattr(detached, "_shape", None) == (2, 2)
    assert detached.shape == (2, 2)


if __name__ == "__main__":
    pytest.main([__file__])
