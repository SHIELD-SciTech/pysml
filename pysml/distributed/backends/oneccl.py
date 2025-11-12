"""oneCCL based collectives for Intel XPU devices."""

from __future__ import annotations

from .torch_backend import TorchDistributedBackend


class OneCCLBackend(TorchDistributedBackend):
    name = "oneccl"
    backend_name = "ccl"
    device_types = ("xpu",)


__all__ = ["OneCCLBackend"]
