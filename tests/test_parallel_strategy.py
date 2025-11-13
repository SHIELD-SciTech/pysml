import os
import tempfile
import unittest

import pytest

np = pytest.importorskip("numpy")

from pysml import Tensor, distributed
from pysml.distributed import ParallelStrategy
from pysml.nn import Adam, Linear, Module
from pysml.save_load import load_checkpoint, save_checkpoint


class ToyModel(Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = Linear(4, 4)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.linear(inputs)


class IdentityStage(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class ParallelStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        distributed.shutdown()
        for key in ("WORLD_SIZE", "RANK"):
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        distributed.shutdown()

    def test_strategy_round_trip_serialization(self) -> None:
        strategy = ParallelStrategy.hybrid(
            data=1,
            pipeline=3,
            tensor=2,
            schedule="1f1b",
            chunks=2,
            activation_checkpoint=True,
            mode="2d",
            dims=(2, 1),
        )
        payload = strategy.to_dict()
        rebuilt = ParallelStrategy.from_dict(payload)
        self.assertEqual(strategy, rebuilt)

    def test_validation_enforces_rank_requirements(self) -> None:
        strategy = ParallelStrategy.hybrid(data=2, pipeline=1, tensor=2)
        with self.assertRaises(ValueError):
            strategy.validate(world_size=2)
        self.assertEqual(strategy.validate(world_size=4), 4)

    def test_validation_allows_pipeline_on_single_process(self) -> None:
        strategy = ParallelStrategy.pipeline(stages=4)
        self.assertEqual(strategy.validate(world_size=1), 1)

    def test_pipeline_application_requires_stage_definitions(self) -> None:
        strategy = ParallelStrategy.pipeline(stages=2)
        with self.assertRaises(ValueError):
            strategy.apply(ToyModel())

    def test_pipeline_application_runs_with_explicit_stages(self) -> None:
        strategy = ParallelStrategy.pipeline(stages=2)
        pipeline = strategy.apply(pipeline_stages=[IdentityStage(), IdentityStage()])
        out = pipeline(Tensor(np.ones((1, 4), dtype=np.float32)))
        self.assertEqual(out.shape, (1, 4))

    def test_checkpoint_serializes_parallel_strategy(self) -> None:
        model = ToyModel()
        optimizer = Adam(model.parameters(), lr=1e-3)
        strategy = ParallelStrategy.data(degree=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "checkpoint.pkl")
            save_checkpoint(
                model,
                optimizer,
                path,
                epoch=3,
                loss=0.5,
                strategy=strategy,
                distributed_state={"rank": 0},
            )
            metadata = load_checkpoint(model, optimizer, path)
        self.assertIn("parallel_strategy", metadata)
        self.assertIsInstance(metadata["parallel_strategy"], ParallelStrategy)
        self.assertEqual(metadata["distributed_state"]["rank"], 0)


if __name__ == "__main__":
    unittest.main()
