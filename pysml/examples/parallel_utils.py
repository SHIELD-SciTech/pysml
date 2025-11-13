"""Utilities for configuring example scripts across backends and parallel modes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pysml.distributed import ParallelStrategy
from pysml.nn import Module


@dataclass
class ExampleConfig:
    """Describe how an example should be executed.

    The dataclass mirrors CLI arguments you might pass to a launcher: select a
    backend string (``"cpu"``, ``"cuda"``, ``"xpu"``), dial in data/pipeline/tensor
    degrees, and optionally enable activation checkpointing. Example docstrings
    throughout ``pysml.examples`` reference ``ExampleConfig`` directly so readers
    can discover every knob from a single place.
    """

    backend: str = "cpu"
    data_parallel: int = 1
    pipeline_parallel: int = 1
    tensor_parallel: int = 1
    pipeline_schedule: str = "gpipe"
    pipeline_chunks: int = 1
    activation_checkpoint: bool = False

    def device(self) -> str:
        if any(self.backend.startswith(prefix) for prefix in ("cpu", "cuda", "xpu")):
            if ":" not in self.backend and self.backend != "cpu":
                return f"{self.backend}:0"
            return self.backend
        raise ValueError(f"Unsupported backend specification: {self.backend}")

    def build_strategy(self) -> Optional[ParallelStrategy]:
        if (
            self.data_parallel == 1
            and self.pipeline_parallel == 1
            and self.tensor_parallel == 1
        ):
            return None
        return ParallelStrategy.hybrid(
            data=self.data_parallel,
            pipeline=self.pipeline_parallel,
            tensor=self.tensor_parallel,
            schedule=self.pipeline_schedule,
            chunks=self.pipeline_chunks,
            activation_checkpoint=self.activation_checkpoint,
        )

    def apply(
        self,
        module: Module,
        *,
        pipeline_stages: Optional[list[Module]] = None,
        pipeline_kwargs: Optional[dict] = None,
    ) -> Module:
        module = module.to(self.device())
        stages = None
        if pipeline_stages is not None:
            stages = [stage.to(self.device()) for stage in pipeline_stages]

        strategy = self.build_strategy()

        if stages is not None:
            if strategy is None:
                strategy = ParallelStrategy.pipeline(
                    len(stages),
                    schedule=self.pipeline_schedule,
                    chunks=self.pipeline_chunks,
                    activation_checkpoint=self.activation_checkpoint,
                )
            kwargs = {"partitions": [1] * len(stages)}
            if pipeline_kwargs:
                kwargs.update(pipeline_kwargs)
            return strategy.apply(pipeline_stages=stages, pipeline_kwargs=kwargs)

        if strategy is None:
            return module
        return strategy.apply(module)


__all__ = ["ExampleConfig"]
