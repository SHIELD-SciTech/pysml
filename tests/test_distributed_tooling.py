import math
import os
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG_DIR = ROOT / "pysml"
if "pysml" not in sys.modules:
    stub = types.ModuleType("pysml")
    stub.__path__ = [str(PKG_DIR)]
    sys.modules["pysml"] = stub
if "pysml.distributed" not in sys.modules:
    dist_stub = types.ModuleType("pysml.distributed")
    dist_stub.__path__ = [str(PKG_DIR / "distributed")]
    sys.modules["pysml.distributed"] = dist_stub
if "pysml.nn" not in sys.modules:
    nn_stub = types.ModuleType("pysml.nn")
    nn_stub.__path__ = [str(PKG_DIR / "nn")]
    sys.modules["pysml.nn"] = nn_stub

from pysml.distributed.monitoring import DistributedMonitor
from pysml.distributed.checkpointing import save_rank_checkpoint, load_rank_checkpoint
from pysml.distributed.debugging import register_gradient_anomaly_detector, launch_rank_repl
from pysml.nn.module import Module, Parameter
from pysml.tensor import Tensor


class _ToyModel(Module):
    def __init__(self):
        super().__init__()
        self.weight = Parameter([1.0], requires_grad=True)

    def forward(self, x):
        return x


class _DummyOptim:
    def __init__(self, params):
        self._params = list(params)
        self.state = {"params": len(self._params)}

    def state_dict(self):
        return {"state": self.state}

    def load_state_dict(self, state_dict):
        self.state = state_dict.get("state", {})


def test_distributed_monitor_aggregates_metrics():
    lines = []
    monitor = DistributedMonitor(total_steps=4, log_every=5, writer=lines.append)
    monitor.update(batch_size=4, loss=2.0, duration=0.5)
    monitor.update(batch_size=4, loss=1.0, duration=0.5)
    metrics = monitor.report(force=True)

    assert metrics.samples == 8
    assert metrics.steps == 2
    assert math.isclose(metrics.average_loss, 1.5)
    assert math.isclose(metrics.throughput, 8 / 1.0)
    assert lines, "monitor should emit progress lines"


def test_rank_checkpoint_round_trip(tmp_path):
    model = _ToyModel()
    optimizer = _DummyOptim(model.parameters())
    directory = tmp_path / "ckpt"
    metadata = {"epoch": 3, "loss": 0.5}

    shard_path = save_rank_checkpoint(model, optimizer, str(directory), metadata=metadata)
    assert os.path.exists(shard_path)

    loaded = load_rank_checkpoint(model, optimizer, str(directory))
    assert loaded["checkpoint_metadata"] == metadata
    assert "world_size" in loaded["manifest"]


def test_gradient_anomaly_detector_flags_nan():
    model = _ToyModel()
    triggered = []

    handles = register_gradient_anomaly_detector(model, callback=lambda event: triggered.append(event))

    param = model.weight
    param.grad = Tensor([float("nan")], requires_grad=False)
    param.data._run_post_backward_hooks()

    for handle in handles:
        handle.remove()

    assert triggered and triggered[0]["parameter"].endswith("weight")


def test_launch_rank_repl_returns_banner():
    message = launch_rank_repl(interactive=False)
    assert "rank" in message
