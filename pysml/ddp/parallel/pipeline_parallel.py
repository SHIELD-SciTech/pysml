"""Pipeline parallel wrapper exposing a drop-in constructor decorator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from pysml.nn import Module, Sequential
from pysml.nn.pipeline import PipelineModule
from pysml.tensor import Tensor

from ..config.partition_config import PartitionConfig, build_partition_config, detect_device_specs


@dataclass
class StagePlan:
    modules: List[Module]
    device: Optional[str]


def _extract_layers(module: Module) -> List[Module]:
    if isinstance(module, Sequential):
        return list(module._modules.values())
    layers = []
    for child in module.children():
        layers.append(child)
    if not layers:
        return [module]
    return layers


class PipelineParallel:
    """Decorator that partitions the provided module across devices."""

    def __init__(
        self,
        module_ctor: Callable[..., Module],
        *,
        devices: Optional[Sequence[str]] = None,
        chunks: int = 1,
        partitions: Optional[Sequence[int]] = None,
    ) -> None:
        self._module_ctor = module_ctor
        self._devices = detect_device_specs(devices)
        self._explicit_partitions = list(partitions) if partitions is not None else None
        self._chunks = max(1, chunks)

    def __call__(self, *args, **kwargs) -> PipelineModule:
        module = self._module_ctor(*args, **kwargs)
        layers = _extract_layers(module)
        if not layers:
            raise ValueError("PipelineParallel requires the module to expose at least one layer")
        if self._explicit_partitions is not None:
            plan = PartitionConfig(list(self._explicit_partitions), self._devices, self._chunks)
        else:
            plan = build_partition_config(len(layers), [spec.name for spec in self._devices], chunks=self._chunks)
        stages = []
        cursor = 0
        for count, spec in zip(plan.partitions, plan.devices):
            bucket = layers[cursor : cursor + count]
            cursor += count
            seq = Sequential(*bucket)
            stages.append(seq.to(spec.name))
        return PipelineModule(
            stages,
            partitions=[1] * len(stages),
            stage_devices=[spec.name for spec in plan.devices],
            stage_ranks=list(range(len(plan.devices))),
            chunks=self._chunks,
        )


__all__ = ["PipelineParallel", "StagePlan"]
