"""Device-agnostic collective primitives implemented without torch.distributed."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np

from pysml.tensor import Tensor

SUPPORTED_KINDS = {"cpu", "cuda", "xpu"}


@dataclass
class MailboxEntry:
    """Message envelope used by :class:`Communicator`."""

    src: str
    payload: Tensor


@dataclass
class Communicator:
    """Single-process communicator coordinating device transfers."""

    devices: Sequence[str]
    name: str = "global"
    mailboxes: Dict[str, List[MailboxEntry]] = field(default_factory=dict)
    _reductions: Dict[str, List[Tensor]] = field(default_factory=dict)
    _active_device: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.devices:
            raise ValueError("Communicator requires at least one device")
        normalized = []
        for device in self.devices:
            normalized.append(_normalize_device(device))
        self.devices = tuple(normalized)
        for device in self.devices:
            self.mailboxes.setdefault(device, [])

    @property
    def world_size(self) -> int:
        return len(self.devices)

    def rank(self, device: Optional[str] = None) -> int:
        target = _normalize_device(device or self._active_device or self.devices[0])
        try:
            return self.devices.index(target)
        except ValueError:
            raise ValueError(f"Device {target!r} is not part of communicator {self.name!r}")

    @contextmanager
    def device(self, device: str) -> Iterator[None]:
        previous = self._active_device
        self._active_device = _normalize_device(device)
        try:
            yield
        finally:
            self._active_device = previous

    # ------------------------------------------------------------------
    # Point-to-point primitives
    # ------------------------------------------------------------------
    def send(self, tensor: Tensor, dst: str, *, src: Optional[str] = None) -> None:
        src_device = _normalize_device(src or self._active_device or self.devices[0])
        dst_device = _normalize_device(dst)
        if dst_device not in self.mailboxes:
            raise ValueError(f"Device {dst_device!r} is not registered in communicator {self.name}")
        payload = _move_tensor(tensor, dst_device)
        self.mailboxes[dst_device].append(MailboxEntry(src_device, payload))

    def recv(self, device: Optional[str] = None, *, src: Optional[str] = None) -> Tensor:
        target = _normalize_device(device or self._active_device or self.devices[0])
        inbox = self.mailboxes.setdefault(target, [])
        if not inbox:
            raise RuntimeError(f"No pending messages for device {target!r}")
        if src is None:
            entry = inbox.pop(0)
            return entry.payload
        normalized_src = _normalize_device(src)
        for idx, entry in enumerate(inbox):
            if entry.src == normalized_src:
                inbox.pop(idx)
                return entry.payload
        raise RuntimeError(f"No message from {normalized_src!r} for device {target!r}")

    # ------------------------------------------------------------------
    # Collective primitives
    # ------------------------------------------------------------------
    def all_reduce(self, tensor: Tensor, *, op: str = "sum", tag: Optional[str] = None) -> Tensor:
        bucket_id = tag or "default"
        payload = _move_tensor(tensor, "cpu")
        bucket = self._reductions.setdefault(bucket_id, [])
        bucket.append(payload.clone())
        if len(bucket) < self.world_size:
            return payload
        reduced = _reduce_tensors(bucket, op)
        self._reductions.pop(bucket_id, None)
        return reduced

    def broadcast(self, tensor: Tensor, *, src: Optional[str] = None) -> List[Tensor]:
        source = _normalize_device(src or self.devices[0])
        payload = _move_tensor(tensor, source)
        return [_move_tensor(payload, device) for device in self.devices]

    def gather(self, tensor: Tensor, *, dst: Optional[str] = None) -> List[Tensor]:
        target = _normalize_device(dst or self.devices[0])
        payload = _move_tensor(tensor, target)
        bucket = self.mailboxes.setdefault(target, [])
        bucket.append(MailboxEntry(_normalize_device(self._active_device or target), payload))
        return [entry.payload for entry in bucket]


_GLOBAL_COMMUNICATOR: Optional[Communicator] = None


def register_global_communicator(devices: Sequence[str]) -> Communicator:
    """Create and register the default communicator for ad-hoc use."""

    global _GLOBAL_COMMUNICATOR
    _GLOBAL_COMMUNICATOR = Communicator(list(devices))
    return _GLOBAL_COMMUNICATOR


def default_communicator() -> Communicator:
    global _GLOBAL_COMMUNICATOR
    if _GLOBAL_COMMUNICATOR is None:
        _GLOBAL_COMMUNICATOR = Communicator(["cpu:0"])
    return _GLOBAL_COMMUNICATOR


# ----------------------------------------------------------------------
# Convenience wrappers mirroring the public API that previously lived in
# ``pysml.distributed.collectives``. The new implementation relies solely on
# :class:`Tensor` methods and NumPy fallbacks, so no torch/oneCCL bindings are
# required.
# ----------------------------------------------------------------------

def send(tensor: Tensor, dst: str, *, communicator: Optional[Communicator] = None) -> None:
    (communicator or default_communicator()).send(tensor, dst)


def recv(device: Optional[str] = None, *, communicator: Optional[Communicator] = None) -> Tensor:
    return (communicator or default_communicator()).recv(device)


def all_reduce(
    tensor: Tensor,
    *,
    op: str = "sum",
    tag: Optional[str] = None,
    communicator: Optional[Communicator] = None,
) -> Tensor:
    return (communicator or default_communicator()).all_reduce(tensor, op=op, tag=tag)


def broadcast(
    tensor: Tensor,
    *,
    src: Optional[str] = None,
    communicator: Optional[Communicator] = None,
) -> List[Tensor]:
    return (communicator or default_communicator()).broadcast(tensor, src=src)


def gather(
    tensor: Tensor,
    *,
    dst: Optional[str] = None,
    communicator: Optional[Communicator] = None,
) -> List[Tensor]:
    return (communicator or default_communicator()).gather(tensor, dst=dst)


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def _normalize_device(device: str) -> str:
    if device is None:
        raise ValueError("device must be specified")
    if ":" not in device and device != "cpu":
        device = f"{device}:0"
    kind = device.split(":", 1)[0]
    if kind not in SUPPORTED_KINDS:
        raise ValueError(f"Unsupported device kind: {kind}")
    return device


def _move_tensor(tensor: Tensor, device: str) -> Tensor:
    replica = tensor.clone()
    replica.to(device, copy=True)
    return replica


def _reduce_tensors(tensors: Iterable[Tensor], op: str) -> Tensor:
    arrays = [tensor.numpy() for tensor in tensors]
    if not arrays:
        raise RuntimeError("all_reduce received no tensors")
    if op == "sum":
        reduced = np.sum(arrays, axis=0)
    elif op == "mean":
        reduced = np.mean(arrays, axis=0)
    elif op == "max":
        reduced = np.max(arrays, axis=0)
    else:
        raise ValueError(f"Unsupported reduction op: {op}")
    result = tensors[0]._new_like(tensors[0]._backend.asarray(reduced))
    return result


__all__ = [
    "Communicator",
    "MailboxEntry",
    "all_reduce",
    "broadcast",
    "default_communicator",
    "gather",
    "recv",
    "register_global_communicator",
    "send",
]
