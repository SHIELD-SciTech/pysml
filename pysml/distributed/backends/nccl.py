"""NCCL based collectives for CUDA tensors."""

from __future__ import annotations

from .torch_backend import TorchDistributedBackend


class NCCLBackend(TorchDistributedBackend):
    name = "nccl"
    backend_name = "nccl"
    device_types = ("cuda",)


__all__ = ["NCCLBackend"]
