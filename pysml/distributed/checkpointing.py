"""Utilities for distributed checkpointing and elastic restarts."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from ..save_load import load, save
from . import collectives, process_group, tensor_parallel

_MANIFEST_NAME = "manifest.json"


@dataclass
class CheckpointManifest:
    """Metadata describing a distributed checkpoint layout."""

    world_size: int
    tensor_parallel_size: int
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    shards: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "world_size": self.world_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "shards": self.shards,
        }

    @classmethod
    def from_file(cls, directory: str) -> "CheckpointManifest":
        path = os.path.join(directory, _MANIFEST_NAME)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint manifest missing: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            world_size=int(payload.get("world_size", 1)),
            tensor_parallel_size=int(payload.get("tensor_parallel_size", 1)),
            timestamp=float(payload.get("timestamp", time.time())),
            metadata=dict(payload.get("metadata", {})),
            shards=dict(payload.get("shards", {})),
        )

    def write(self, directory: str) -> None:
        path = os.path.join(directory, _MANIFEST_NAME)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)


def _default_checkpoint_dir(directory: str) -> str:
    directory = os.path.abspath(directory)
    os.makedirs(directory, exist_ok=True)
    return directory


def _serialize_strategy(strategy: Any) -> Any:
    if strategy is None:
        return None
    if hasattr(strategy, "to_dict"):
        return strategy.to_dict()
    raise TypeError("strategy must expose to_dict() for serialization")


def _maybe_tensor_parallel_rank() -> tuple[int, int]:
    try:
        group = tensor_parallel.get_tensor_parallel_group()
        return group.local_rank, group.size
    except Exception:
        return 0, 1


def tensor_parallel_shard_state_dict(state_dict: Dict[str, Any], dim: int = 0) -> Dict[str, Any]:
    """Shard tensors across tensor-parallel ranks along ``dim``."""

    rank, size = _maybe_tensor_parallel_rank()
    if size <= 1:
        return dict(state_dict)

    sharded: Dict[str, Any] = {}
    for key, value in state_dict.items():
        if not hasattr(value, "shape") or not hasattr(value, "__getitem__"):
            sharded[key] = value
            continue
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) == 0:
            sharded[key] = value
            continue
        dim_len = int(shape[dim])
        if dim_len <= size:
            start = min(rank, dim_len - 1)
            end = start + 1
        else:
            chunk = dim_len // size
            remainder = dim_len % size
            start = rank * chunk + min(rank, remainder)
            end = start + chunk + (1 if rank < remainder else 0)
        slicer = [slice(None)] * len(shape)
        slicer[dim] = slice(start, end)
        sharded[key] = value[tuple(slicer)]
    return sharded


def save_rank_checkpoint(
    model,
    optimizer,
    directory: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    strategy: Any = None,
    shard_hook: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> str:
    """Persist a checkpoint shard for the current rank."""

    directory = _default_checkpoint_dir(directory)
    process_group.lazy_init_from_env()
    backend = process_group.get_backend()
    rank = backend.rank
    world_size = max(backend.world_size, 1)
    tp_rank, tp_size = _maybe_tensor_parallel_rank()

    state_dict = model.state_dict()
    if shard_hook is not None:
        state_dict = shard_hook(state_dict)
    optimizer_state = optimizer.state_dict() if optimizer is not None else None

    payload = {
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer_state,
        "metadata": metadata or {},
    }
    if strategy is not None:
        payload["parallel_strategy"] = _serialize_strategy(strategy)

    filename = os.path.join(directory, f"rank{rank:05d}.pt")
    save(payload, filename)

    manifest_entry = {
        "rank": rank,
        "path": os.path.basename(filename),
        "tensor_parallel_rank": tp_rank,
        "tensor_parallel_size": tp_size,
    }
    temp_manifest = os.path.join(directory, f".manifest.rank{rank:05d}.json")
    with open(temp_manifest, "w", encoding="utf-8") as handle:
        json.dump(manifest_entry, handle, indent=2)

    collectives.barrier()

    if rank == 0:
        manifest = CheckpointManifest(world_size=world_size, tensor_parallel_size=tp_size, metadata=metadata or {})
        for candidate in sorted(os.listdir(directory)):
            if not candidate.startswith(".manifest.rank") or not candidate.endswith(".json"):
                continue
            path = os.path.join(directory, candidate)
            with open(path, "r", encoding="utf-8") as handle:
                entry = json.load(handle)
            manifest.shards[str(entry["rank"])] = entry
            os.remove(path)
        manifest.write(directory)

    collectives.barrier()
    return filename


def load_rank_checkpoint(
    model,
    optimizer,
    directory: str,
    *,
    map_location: Optional[str] = None,
    strict: bool = True,
    restore_optimizer: bool = True,
) -> Dict[str, Any]:
    """Load the shard for the current rank, supporting elastic restarts."""

    directory = _default_checkpoint_dir(directory)
    manifest = CheckpointManifest.from_file(directory)
    process_group.lazy_init_from_env()
    backend = process_group.get_backend()
    rank = backend.rank

    available = {int(key): value for key, value in manifest.shards.items()}
    if not available:
        raise RuntimeError("Manifest does not contain shard metadata")
    target_rank = rank if rank in available else sorted(available)[rank % len(available)]
    entry = available[target_rank]
    path = os.path.join(directory, entry["path"])
    checkpoint = load(path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if restore_optimizer and optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    result = {
        "manifest": manifest.to_dict(),
        "checkpoint_metadata": dict(checkpoint.get("metadata", {})),
    }
    if "parallel_strategy" in checkpoint:
        result["parallel_strategy"] = checkpoint["parallel_strategy"]
    return result


__all__ = [
    "save_rank_checkpoint",
    "load_rank_checkpoint",
    "tensor_parallel_shard_state_dict",
    "CheckpointManifest",
]
