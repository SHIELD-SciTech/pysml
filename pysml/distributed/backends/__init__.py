"""Backend registry for :mod:`pysml.distributed`."""

from __future__ import annotations

from typing import Dict, Iterable, Type

from .base import CollectiveBackend, DummyBackend
from .gloo import GlooBackend
from .nccl import NCCLBackend
from .oneccl import OneCCLBackend


def iter_backend_classes() -> Iterable[Type[CollectiveBackend]]:
    yield GlooBackend
    yield NCCLBackend
    yield OneCCLBackend


def create_backend(name: str | None = None) -> CollectiveBackend:
    if name is None:
        return DummyBackend()
    key = name.lower()
    for backend_cls in iter_backend_classes():
        if backend_cls.name == key:
            return backend_cls()
    if key == "dummy":
        return DummyBackend()
    raise ValueError(f"Unknown distributed backend: {name}")


def detect_available_backends() -> Dict[str, CollectiveBackend]:
    registry: Dict[str, CollectiveBackend] = {}
    for backend_cls in iter_backend_classes():
        backend = backend_cls()
        if backend.is_available():
            registry[backend.name] = backend
    if not registry:
        dummy = DummyBackend()
        registry[dummy.name] = dummy
    return registry


__all__ = [
    "CollectiveBackend",
    "DummyBackend",
    "create_backend",
    "detect_available_backends",
]
