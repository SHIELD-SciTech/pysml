"""User facing collective communication helpers."""

from __future__ import annotations

from typing import Any, Iterable

from .routing import get_communicator


def all_reduce(tensor: Any, op: str = "sum") -> Any:
    communicator = get_communicator(tensor)
    return communicator.all_reduce(tensor, op=op)


def broadcast(tensor: Any, src: int = 0) -> Any:
    communicator = get_communicator(tensor)
    return communicator.broadcast(tensor, src=src)


def all_gather(tensor: Any) -> Iterable[Any]:
    communicator = get_communicator(tensor)
    return communicator.all_gather(tensor)


def reduce_scatter(tensors: Iterable[Any], op: str = "sum") -> Any:
    tensors = list(tensors)
    if not tensors:
        raise ValueError("reduce_scatter requires a non-empty iterable")
    communicator = get_communicator(tensors[0])
    return communicator.reduce_scatter(tensors, op=op)


def barrier(device: str | None = None) -> None:
    communicator = get_communicator(device)
    communicator.barrier()


__all__ = [
    "all_reduce",
    "broadcast",
    "all_gather",
    "reduce_scatter",
    "barrier",
]
