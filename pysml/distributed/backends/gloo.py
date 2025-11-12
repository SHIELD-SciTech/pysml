"""Gloo based collectives."""

from __future__ import annotations

from .torch_backend import TorchDistributedBackend


class GlooBackend(TorchDistributedBackend):
    name = "gloo"
    backend_name = "gloo"
    device_types = ("cpu",)


__all__ = ["GlooBackend"]
