import pathlib
import sys
import unittest

import pytest

np = pytest.importorskip("numpy")

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pysml import Tensor
from pysml.ddp.parallel.pipeline_parallel import PipelineParallel
from pysml.nn import Linear, Module, Sequential


class LinearStack(Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = Sequential(Linear(4, 4), Linear(4, 2))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PipelineWrapperTests(unittest.TestCase):
    def test_pipeline_parallel_splits_layers(self) -> None:
        wrapped = PipelineParallel(lambda: LinearStack().net, devices=["cpu:0", "cpu:1"], chunks=2)
        pipeline = wrapped()
        data = Tensor(np.ones((2, 4), dtype=np.float32))
        out = pipeline(data)
        self.assertEqual(out.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
