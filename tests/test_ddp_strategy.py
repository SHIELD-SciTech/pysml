import pathlib
import sys
import unittest

import pytest

pytest.importorskip("numpy")

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pysml.ddp.config.partition_config import ParallelStrategy


class StrategyTests(unittest.TestCase):
    def test_round_trip_serialization(self) -> None:
        strategy = ParallelStrategy.hybrid(
            data=2,
            pipeline=2,
            tensor=1,
            schedule="1f1b",
            chunks=4,
            activation_checkpoint=True,
        )
        payload = strategy.to_dict()
        rebuilt = ParallelStrategy.from_dict(payload)
        self.assertEqual(strategy, rebuilt)

    def test_validate_checks_world_size(self) -> None:
        strategy = ParallelStrategy.hybrid(data=2, pipeline=2, tensor=1)
        self.assertEqual(strategy.validate(world_size=4), 4)
        with self.assertRaises(ValueError):
            strategy.validate(world_size=2)

    def test_data_factory(self) -> None:
        strategy = ParallelStrategy.data(degree=3)
        self.assertEqual(strategy.data_parallel, 3)
        self.assertEqual(strategy.pipeline_parallel, 1)


if __name__ == "__main__":
    unittest.main()
