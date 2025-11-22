import numpy as np

from pysml.tensor import Tensor, gc


def test_tensor_free_idempotent(monkeypatch):
    calls = []

    def fake_collect():
        calls.append(True)

    monkeypatch.setattr(gc, "collect", fake_collect)

    t = Tensor(np.ones((2, 2), dtype=np.float32))
    t._grad = Tensor(np.ones_like(t.data))

    t.free()

    assert t._freed is True
    assert t.data is None
    assert t._grad is None
    assert calls == []

    # Second invocation should be a no-op and still not trigger collection
    t.free()
    assert calls == []
