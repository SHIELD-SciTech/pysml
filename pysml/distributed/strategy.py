"""Composite parallelism configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from . import process_group, tensor_parallel
from .ddp import DistributedDataParallel
from ..nn.module import Module
from ..nn.pipeline import PipelineModule


@dataclass(frozen=True)
class ParallelStrategy:
    """Describe how a model should be partitioned across parallel dimensions."""

    data_parallel: int = 1
    pipeline_parallel: int = 1
    tensor_parallel: int = 1
    pipeline_schedule: str = "gpipe"
    pipeline_chunks: int = 1
    activation_checkpoint: bool = False
    tensor_parallel_mode: str = "1d"
    tensor_parallel_dims: Optional[tuple[int, ...]] = None

    def __post_init__(self) -> None:
        for field in ("data_parallel", "pipeline_parallel", "tensor_parallel", "pipeline_chunks"):
            value = getattr(self, field)
            if value <= 0:
                raise ValueError(f"{field} must be positive, received {value}")
        if self.pipeline_schedule not in {"gpipe", "1f1b"}:
            raise ValueError("pipeline_schedule must be either 'gpipe' or '1f1b'")
        object.__setattr__(self, "tensor_parallel_mode", self.tensor_parallel_mode.lower())
        dims = self.tensor_parallel_dims
        if dims is not None:
            object.__setattr__(self, "tensor_parallel_dims", tuple(int(d) for d in dims))

    # ------------------------------------------------------------------
    # Convenience builders
    # ------------------------------------------------------------------
    @classmethod
    def data(cls, degree: int) -> "ParallelStrategy":
        """Create a pure data parallel strategy."""

        return cls(data_parallel=degree)

    @classmethod
    def tensor(
        cls,
        degree: int,
        *,
        mode: str = "1d",
        dims: Optional[Sequence[int]] = None,
    ) -> "ParallelStrategy":
        """Create a pure tensor parallel strategy."""

        return cls(tensor_parallel=degree, tensor_parallel_mode=mode, tensor_parallel_dims=None if dims is None else tuple(dims))

    @classmethod
    def pipeline(
        cls,
        stages: int,
        *,
        schedule: str = "gpipe",
        chunks: int = 1,
        activation_checkpoint: bool = False,
    ) -> "ParallelStrategy":
        """Create a pure pipeline parallel strategy."""

        return cls(
            pipeline_parallel=stages,
            pipeline_schedule=schedule,
            pipeline_chunks=chunks,
            activation_checkpoint=activation_checkpoint,
        )

    @classmethod
    def hybrid(
        cls,
        *,
        data: int = 1,
        pipeline: int = 1,
        tensor: int = 1,
        schedule: str = "gpipe",
        chunks: int = 1,
        activation_checkpoint: bool = False,
        mode: str = "1d",
        dims: Optional[Sequence[int]] = None,
    ) -> "ParallelStrategy":
        """Create a combined data/pipeline/tensor strategy."""

        return cls(
            data_parallel=data,
            pipeline_parallel=pipeline,
            tensor_parallel=tensor,
            pipeline_schedule=schedule,
            pipeline_chunks=chunks,
            activation_checkpoint=activation_checkpoint,
            tensor_parallel_mode=mode,
            tensor_parallel_dims=None if dims is None else tuple(dims),
        )

    # ------------------------------------------------------------------
    # Derived properties and helpers
    # ------------------------------------------------------------------
    @property
    def requires_pipeline_stages(self) -> bool:
        return self.pipeline_parallel > 1

    @property
    def total_participants(self) -> int:
        return self.data_parallel * self.pipeline_parallel * self.tensor_parallel

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the strategy for checkpoint metadata."""

        return {
            "data_parallel": self.data_parallel,
            "pipeline_parallel": self.pipeline_parallel,
            "tensor_parallel": self.tensor_parallel,
            "pipeline_schedule": self.pipeline_schedule,
            "pipeline_chunks": self.pipeline_chunks,
            "activation_checkpoint": self.activation_checkpoint,
            "tensor_parallel_mode": self.tensor_parallel_mode,
            "tensor_parallel_dims": None if self.tensor_parallel_dims is None else list(self.tensor_parallel_dims),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ParallelStrategy":
        """Rebuild a strategy from serialized metadata."""

        return cls(
            data_parallel=int(payload.get("data_parallel", 1)),
            pipeline_parallel=int(payload.get("pipeline_parallel", 1)),
            tensor_parallel=int(payload.get("tensor_parallel", 1)),
            pipeline_schedule=payload.get("pipeline_schedule", "gpipe"),
            pipeline_chunks=int(payload.get("pipeline_chunks", 1)),
            activation_checkpoint=bool(payload.get("activation_checkpoint", False)),
            tensor_parallel_mode=payload.get("tensor_parallel_mode", "1d"),
            tensor_parallel_dims=None
            if payload.get("tensor_parallel_dims") is None
            else tuple(int(dim) for dim in payload["tensor_parallel_dims"]),
        )

    # ------------------------------------------------------------------
    # Validation and application
    # ------------------------------------------------------------------
    def validate(self, *, world_size: Optional[int] = None) -> int:
        """Ensure the strategy aligns with the active process topology."""

        if world_size is None:
            process_group.lazy_init_from_env()
            world_size = process_group.get_world_size()
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if self.tensor_parallel > world_size:
            raise ValueError(
                f"tensor_parallel={self.tensor_parallel} exceeds available ranks ({world_size})"
            )
        if world_size == 1:
            if self.data_parallel != 1:
                raise ValueError("data_parallel > 1 requires multiple processes")
            if self.tensor_parallel != 1:
                raise ValueError("tensor_parallel > 1 requires multiple processes")
            return world_size
        expected = self.data_parallel * self.tensor_parallel * self.pipeline_parallel
        if expected != world_size:
            raise ValueError(
                "In distributed contexts the world size must equal data_parallel * "
                "pipeline_parallel * tensor_parallel ("
                f"expected {expected}, got {world_size})."
            )
        return world_size

    def apply(
        self,
        module: Optional[Module] = None,
        *,
        pipeline_stages: Optional[Sequence[Module]] = None,
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
        ddp_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Module:
        """Wrap ``module`` according to the configured parallel strategy."""

        self.validate()
        wrapped: Module
        if self.tensor_parallel > 1:
            tensor_parallel.init_tensor_parallel(
                tp_size=self.tensor_parallel,
                mode=self.tensor_parallel_mode,
                dims=self.tensor_parallel_dims,
            )
        if pipeline_stages is not None:
            defaults = {
                "partitions": [1] * len(pipeline_stages),
                "schedule": self.pipeline_schedule,
                "chunks": self.pipeline_chunks,
                "activation_checkpoint": self.activation_checkpoint,
            }
            if pipeline_kwargs:
                defaults.update(pipeline_kwargs)
            wrapped = PipelineModule(pipeline_stages, **defaults)
        else:
            if module is None:
                raise ValueError("module must be provided when pipeline stages are omitted")
            if self.requires_pipeline_stages:
                raise ValueError(
                    "pipeline_parallel > 1 requires explicit pipeline stage definitions"
                )
            wrapped = module
        if self.data_parallel > 1:
            ddp_options = dict(ddp_kwargs or {})
            wrapped = DistributedDataParallel(wrapped, **ddp_options)
        return wrapped


__all__ = ["ParallelStrategy"]
