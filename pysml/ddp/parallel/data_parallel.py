"""Simple multi-device data parallel wrapper."""

from __future__ import annotations

import copy
from typing import List, Sequence

import numpy as np

from pysml.nn import Module
from pysml.tensor import Tensor

from ..config.partition_config import detect_device_specs


class _Replica(Module):
    def __init__(self, module: Module, device: str) -> None:
        super().__init__()
        self.module = module.to(device)
        self.device = device

    def forward(self, *args, **kwargs):  # type: ignore[override]
        migrated_args = [self._move(arg) for arg in args]
        migrated_kwargs = {key: self._move(value) for key, value in kwargs.items()}
        return self.module(*migrated_args, **migrated_kwargs)

    def _move(self, value):
        if isinstance(value, Tensor):
            return value.to(self.device, copy=True)
        return value


class DataParallel(Module):
    """Replicate modules across devices and aggregate outputs on the lead device."""

    def __init__(self, module_ctor, *, devices: Sequence[str] | None = None) -> None:
        super().__init__()
        self._module_ctor = module_ctor
        self._device_specs = detect_device_specs(devices)

    def __call__(self, *args, **kwargs) -> Module:  # type: ignore[override]
        module = self._module_ctor(*args, **kwargs)
        replicas = [module] + [copy.deepcopy(module) for _ in range(len(self._device_specs) - 1)]
        wrapped = [_Replica(replica, spec.name) for replica, spec in zip(replicas, self._device_specs)]
        return _DataParallelRuntime(wrapped)


class _DataParallelRuntime(Module):
    def __init__(self, replicas: List[_Replica]) -> None:
        super().__init__()
        self.replicas = replicas
        self.lead_device = replicas[0].device

    def forward(self, *args, **kwargs):  # type: ignore[override]
        if not self.replicas:
            raise RuntimeError("DataParallel runtime requires at least one replica")
        lead_inputs = self._split_args(args)
        outputs = []
        for replica, payload in zip(self.replicas, lead_inputs):
            replica_args, replica_kwargs = payload
            outputs.append(replica(*replica_args, **replica_kwargs))
        if isinstance(outputs[0], Tensor):
            stacked = np.concatenate([output.numpy() for output in outputs], axis=0)
            return Tensor(stacked).to(self.lead_device)
        return outputs

    def _split_args(self, args):
        if not args:
            return [([], {}) for _ in self.replicas]
        first = args[0]
        if not isinstance(first, Tensor) or first.shape[0] < len(self.replicas):
            return [(args, {}) for _ in self.replicas]
        splits = np.array_split(first.numpy(), len(self.replicas), axis=0)
        buckets = []
        for idx, split in enumerate(splits):
            tensor = Tensor(split)
            tensor.to(self.replicas[idx].device)
            buckets.append(((tensor,) + args[1:], {}))
        return buckets


__all__ = ["DataParallel"]
