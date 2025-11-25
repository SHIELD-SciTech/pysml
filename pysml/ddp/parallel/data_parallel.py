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
        lead_inputs = self._split_args(args, kwargs)
        outputs = []
        for replica, payload in zip(self.replicas, lead_inputs):
            replica_args, replica_kwargs = payload
            outputs.append(replica(*replica_args, **replica_kwargs))
        if isinstance(outputs[0], Tensor):
            stacked = np.concatenate([output.numpy() for output in outputs], axis=0)
            return Tensor(stacked).to(self.lead_device)
        return outputs

    def _split_args(self, args, kwargs=None):
        kwargs = kwargs or {}
        if not args and not kwargs:
            return [([], {}) for _ in self.replicas]

        num_replicas = len(self.replicas)

        def _batch_dim(value):
            if isinstance(value, Tensor) and value.ndim > 0:
                return value.shape[0]
            return None

        candidate_dims = []
        for value in args:
            dim = _batch_dim(value)
            if dim is not None:
                candidate_dims.append(dim)
        for value in kwargs.values():
            dim = _batch_dim(value)
            if dim is not None:
                candidate_dims.append(dim)

        if not candidate_dims or min(candidate_dims) < num_replicas:
            return [
                (
                    tuple(self.replicas[idx]._move(arg) if isinstance(arg, Tensor) else arg for arg in args),
                    {
                        key: self.replicas[idx]._move(value) if isinstance(value, Tensor) else value
                        for key, value in kwargs.items()
                    },
                )
                for idx in range(num_replicas)
            ]

        batch = min(candidate_dims)
        indices = np.array_split(np.arange(batch), num_replicas)

        def _split_value(value, target_device):
            if not isinstance(value, Tensor):
                return value
            if value.ndim == 0 or value.shape[0] != batch:
                return value.clone().to(target_device, copy=True)
            shards = [value.numpy()[chunk] for chunk in indices]
            tensors = [Tensor(shard).to(target_device, copy=True) for shard in shards]
            return tensors

        buckets = []
        for idx, replica in enumerate(self.replicas):
            shard_args = []
            shard_kwargs = {}
            for arg in args:
                pieces = _split_value(arg, replica.device)
                shard_args.append(pieces[idx] if isinstance(pieces, list) else pieces)
            for key, value in kwargs.items():
                pieces = _split_value(value, replica.device)
                shard_kwargs[key] = pieces[idx] if isinstance(pieces, list) else pieces
            buckets.append((tuple(shard_args), shard_kwargs))
        return buckets


__all__ = ["DataParallel"]
