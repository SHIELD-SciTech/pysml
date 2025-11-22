"""Device partition discovery and strategy helpers for the custom DDP stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np


SUPPORTED_DEVICE_KINDS = ("cpu", "cuda", "xpu")


@dataclass
class DeviceSpec:
    """Describe the resources available on a logical device."""

    name: str
    memory_gb: float = 8.0
    compute_tflops: float = 10.0

    @property
    def kind(self) -> str:
        return self.name.split(":", 1)[0]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "memory_gb": self.memory_gb,
            "compute_tflops": self.compute_tflops,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DeviceSpec":
        return cls(
            name=payload["name"],
            memory_gb=float(payload.get("memory_gb", 8.0)),
            compute_tflops=float(payload.get("compute_tflops", 10.0)),
        )


@dataclass
class PartitionConfig:
    """Describe how modules should be split across a device mesh."""

    partitions: List[int]
    devices: List[DeviceSpec]
    chunks: int = 1

    def to_dict(self) -> dict:
        return {
            "partitions": list(self.partitions),
            "devices": [device.to_dict() for device in self.devices],
            "chunks": self.chunks,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "PartitionConfig":
        devices = [DeviceSpec.from_dict(info) for info in payload["devices"]]
        return cls(list(payload["partitions"]), devices, int(payload.get("chunks", 1)))


@dataclass
class ParallelStrategy:
    """High level description of the desired parallelism layout."""

    data_parallel: int = 1
    pipeline_parallel: int = 1
    tensor_parallel: int = 1
    schedule: str = "gpipe"
    chunks: int = 1
    activation_checkpoint: bool = False

    def to_dict(self) -> dict:
        return {
            "data_parallel": self.data_parallel,
            "pipeline_parallel": self.pipeline_parallel,
            "tensor_parallel": self.tensor_parallel,
            "schedule": self.schedule,
            "chunks": self.chunks,
            "activation_checkpoint": self.activation_checkpoint,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ParallelStrategy":
        return cls(
            data_parallel=int(payload.get("data_parallel", 1)),
            pipeline_parallel=int(payload.get("pipeline_parallel", 1)),
            tensor_parallel=int(payload.get("tensor_parallel", 1)),
            schedule=payload.get("schedule", "gpipe"),
            chunks=int(payload.get("chunks", 1)),
            activation_checkpoint=bool(payload.get("activation_checkpoint", False)),
        )

    @classmethod
    def pipeline(
        cls,
        stages: int,
        *,
        schedule: str = "gpipe",
        chunks: int = 1,
        activation_checkpoint: bool = False,
    ) -> "ParallelStrategy":
        return cls(
            pipeline_parallel=stages,
            data_parallel=1,
            tensor_parallel=1,
            schedule=schedule,
            chunks=chunks,
            activation_checkpoint=activation_checkpoint,
        )

    @classmethod
    def hybrid(
        cls,
        *,
        data: int,
        pipeline: int,
        tensor: int,
        schedule: str = "gpipe",
        chunks: int = 1,
        activation_checkpoint: bool = False,
    ) -> "ParallelStrategy":
        return cls(
            data_parallel=data,
            pipeline_parallel=pipeline,
            tensor_parallel=tensor,
            schedule=schedule,
            chunks=chunks,
            activation_checkpoint=activation_checkpoint,
        )

    @classmethod
    def data(cls, *, degree: int) -> "ParallelStrategy":
        return cls(data_parallel=degree)

    def validate(self, world_size: int) -> int:
        required = self.data_parallel * self.pipeline_parallel * self.tensor_parallel
        if world_size < required:
            raise ValueError(
                "world size must be >= data_parallel * pipeline_parallel * tensor_parallel"
            )
        return required


def detect_device_specs(devices: Optional[Sequence[str]]) -> List[DeviceSpec]:
    if devices is None:
        return [DeviceSpec(_normalize_device("cpu"), memory_gb=16.0, compute_tflops=1.0)]
    specs = []
    for device in devices:
        normalized = _normalize_device(device)
        # Rough heuristic: assume later devices have slightly less headroom.
        index = len(specs) + 1
        specs.append(
            DeviceSpec(
                normalized,
                memory_gb=max(4.0, 16.0 - index),
                compute_tflops=max(0.5, 10.0 - 0.5 * index),
            )
        )
    return specs


def plan_partitions(num_layers: int, devices: Sequence[DeviceSpec]) -> List[int]:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    weights = [max(device.memory_gb, 1.0) for device in devices]
    total = sum(weights)
    quotas = [num_layers * (weight / total) for weight in weights]
    partitions: List[int] = []
    remainders: List[float] = []
    for quota in quotas:
        share = max(1, int(quota))
        partitions.append(share)
        remainders.append(quota - share)

    allocated = sum(partitions)
    if allocated < num_layers:
        remaining = num_layers - allocated
        for idx in np.argsort(remainders)[::-1].tolist():
            if remaining == 0:
                break
            partitions[idx] += 1
            remaining -= 1
    elif allocated > num_layers:
        overflow = allocated - num_layers
        for idx in np.argsort(remainders).tolist():
            if overflow == 0:
                break
            if partitions[idx] > 1:
                partitions[idx] -= 1
                overflow -= 1
    return partitions


def build_partition_config(num_layers: int, devices: Optional[Sequence[str]], *, chunks: int = 1) -> PartitionConfig:
    specs = detect_device_specs(devices)
    partitions = plan_partitions(num_layers, specs)
    return PartitionConfig(partitions, specs, chunks)


def _normalize_device(device: str) -> str:
    normalized = str(device).lower()
    if normalized.startswith("cpu"):
        return "cpu"
    if ":" not in normalized:
        normalized = f"{normalized}:0"
    kind = normalized.split(":", 1)[0]
    if kind not in SUPPORTED_DEVICE_KINDS:
        raise ValueError(f"Unsupported device kind: {kind}")
    return normalized


__all__ = [
    "DeviceSpec",
    "PartitionConfig",
    "ParallelStrategy",
    "build_partition_config",
    "detect_device_specs",
    "plan_partitions",
]
