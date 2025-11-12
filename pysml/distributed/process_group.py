"""Process group management utilities."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .backends import CollectiveBackend, DummyBackend, create_backend, detect_available_backends


_GLOBAL_BACKENDS: Dict[str, CollectiveBackend] = {}
_DEFAULT_BACKEND: CollectiveBackend = DummyBackend()
_INITIALIZED = False


def is_initialized() -> bool:
    return _INITIALIZED


def init_process_group(
    backend: Optional[str] = None,
    *,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    init_method: Optional[str] = None,
    timeout: Optional[float] = None,
    device_types: Optional[list[str]] = None,
    **kwargs: Any,
) -> CollectiveBackend:
    """Initialise the distributed context.

    Args:
        backend: Name of the backend to use. When ``None`` the backend is
            selected automatically based on environment variables and available
            dependencies.
        rank: Rank of the current process.
        world_size: Total number of participating processes.
        init_method: Optional initialisation URI passed through to the backend.
        timeout: Optional timeout in seconds for the initialisation.
        device_types: Explicit device types handled by this backend. When
            omitted the backend's default mapping is used.
        **kwargs: Additional backend specific keyword arguments.
    """

    global _INITIALIZED, _DEFAULT_BACKEND

    if backend is None:
        backend = _select_backend_from_env()

    communicator = create_backend(backend)
    communicator.initialize(
        rank=_env_rank() if rank is None else rank,
        world_size=_env_world_size() if world_size is None else world_size,
        init_method=init_method or os.getenv("PYSML_INIT_METHOD"),
        timeout=timeout,
        **kwargs,
    )

    _DEFAULT_BACKEND = communicator
    targets = communicator.device_types if device_types is None else tuple(device_types)
    for device in targets:
        _GLOBAL_BACKENDS[device] = communicator
    _INITIALIZED = True
    return communicator


def shutdown() -> None:
    """Destroy the active process group."""

    global _INITIALIZED, _DEFAULT_BACKEND
    for backend in set(_GLOBAL_BACKENDS.values()):
        backend.finalize()
    _GLOBAL_BACKENDS.clear()
    _DEFAULT_BACKEND = DummyBackend()
    _INITIALIZED = False


def get_backend(device: Optional[str] = None) -> CollectiveBackend:
    if device is not None and device in _GLOBAL_BACKENDS:
        return _GLOBAL_BACKENDS[device]
    return _DEFAULT_BACKEND


def get_rank(device: Optional[str] = None) -> int:
    backend = get_backend(device)
    return backend.rank


def get_world_size(device: Optional[str] = None) -> int:
    backend = get_backend(device)
    return backend.world_size


def lazy_init_from_env() -> None:
    """Initialise a process group when environment variables request it."""

    global _DEFAULT_BACKEND, _INITIALIZED

    if is_initialized():
        return

    world_size = _env_world_size()
    if world_size <= 1:
        available = detect_available_backends()
        _DEFAULT_BACKEND = next(iter(available.values()))
        for backend in available.values():
            for device in backend.device_types:
                _GLOBAL_BACKENDS.setdefault(device, backend)
        _INITIALIZED = True
        return

    init_process_group()


def _select_backend_from_env() -> Optional[str]:
    backend = os.getenv("PYSML_DIST_BACKEND")
    if backend:
        return backend

    device_hint = os.getenv("PYSML_DIST_DEVICE")
    if device_hint:
        device_hint = device_hint.lower()
        if device_hint.startswith("cuda"):
            return "nccl"
        if device_hint.startswith("xpu"):
            return "oneccl"
    if os.getenv("CUDA_VISIBLE_DEVICES"):
        return "nccl"
    if os.getenv("XPUM_VISIBLE_DEVICES") or os.getenv("ZE_AFFINITY_MASK"):
        return "oneccl"
    return "gloo"


def _env_rank() -> int:
    rank = os.getenv("RANK") or os.getenv("PMI_RANK") or os.getenv("OMPI_COMM_WORLD_RANK")
    return int(rank) if rank is not None else 0


def _env_world_size() -> int:
    world_size = (
        os.getenv("WORLD_SIZE")
        or os.getenv("PMI_SIZE")
        or os.getenv("OMPI_COMM_WORLD_SIZE")
    )
    return int(world_size) if world_size is not None else 1


__all__ = [
    "init_process_group",
    "shutdown",
    "is_initialized",
    "get_backend",
    "get_rank",
    "get_world_size",
    "lazy_init_from_env",
]
