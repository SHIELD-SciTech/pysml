import pytest

np = pytest.importorskip("numpy")

import pysml
from pysml import Tensor
from pysml.nn import Linear, RNNCell, Sequential
from pysml.nn.activation import Tanh


def device_params():
    devices = ["cpu"]
    if getattr(pysml.cuda, "is_available", lambda: False)():
        devices.append("cuda")
    else:
        devices.append(pytest.param("cuda", marks=pytest.mark.skip(reason="CUDA backend not available")))

    if getattr(pysml.xpu, "is_available", lambda: False)():
        devices.append("xpu")
    else:
        devices.append(pytest.param("xpu", marks=pytest.mark.skip(reason="XPU backend not available")))

    return devices


@pytest.mark.parametrize("device", device_params())
def test_linear_stack_backward_propagates(device):
    model = Sequential(Linear(4, 8), Tanh(), Linear(8, 2))
    model.to(device)

    inputs = Tensor(np.random.randn(5, 4).astype(np.float32), device=device, requires_grad=True)
    targets = Tensor(np.zeros((5, 2), dtype=np.float32), device=device)

    outputs = model(inputs)
    loss = pysml.mean((outputs - targets) ** 2)
    loss.backward()

    for param in model.parameters():
        assert param.grad is not None
        assert param.grad.shape == param.data.shape


@pytest.mark.parametrize("device", device_params())
def test_rnn_cell_sequence_backward(device):
    cell = RNNCell(input_size=3, hidden_size=4)
    cell.to(device)

    sequence = Tensor(np.random.randn(6, 3).astype(np.float32), device=device, requires_grad=True)
    state = None

    for step in pysml.split(sequence, 1, dim=0):
        state = cell(step, state)

    loss = pysml.mean(state)
    loss.backward()

    for param in cell.parameters():
        assert param.grad is not None
        assert param.grad.shape == param.data.shape

    assert sequence.grad is not None
    assert sequence.grad.shape == sequence.data.shape
