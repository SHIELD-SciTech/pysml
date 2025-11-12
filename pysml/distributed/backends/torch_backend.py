"""Thin wrappers around ``torch.distributed`` collectives."""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Optional

from .base import CollectiveBackend

try:  # pragma: no cover - optional dependency
    import torch
    import torch.distributed as dist
except Exception:  # pragma: no cover - dependency not available
    torch = None
    dist = None


class TorchDistributedBackend(CollectiveBackend):
    """Base class for ``torch.distributed`` backed communicators."""

    backend_name: str = "gloo"

    def __init__(self) -> None:
        super().__init__()
        self._group_kwargs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        if dist is None:
            return False
        if not dist.is_available():
            return False
        if self.backend_name == "nccl" and (torch is None or not torch.cuda.is_available()):
            return False
        if self.backend_name == "ccl" and not _has_xpu():
            return False
        return True

    def initialize(
        self,
        *,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        init_method: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().initialize(rank=rank, world_size=world_size)
        if self.world_size <= 1 or not self.is_available():
            if not self.is_available():
                self.add_note(f"torch.distributed backend '{self.backend_name}' unavailable; falling back to local execution")
            return

        init_kwargs = {
            "backend": self.backend_name,
            "rank": self.rank,
            "world_size": self.world_size,
        }
        if init_method is not None:
            init_kwargs["init_method"] = init_method
        if timeout is not None:
            init_kwargs["timeout"] = timeout
        init_kwargs.update(kwargs)
        with _suppress_already_initialized():
            dist.init_process_group(**init_kwargs)

    def finalize(self) -> None:
        if self.is_initialized and dist is not None and dist.is_initialized():  # pragma: no cover - environment sensitive
            with contextlib.suppress(RuntimeError):
                dist.destroy_process_group()
        super().finalize()

    # ------------------------------------------------------------------
    # Collectives
    # ------------------------------------------------------------------
    def barrier(self) -> None:  # pragma: no cover - depends on runtime env
        if self.world_size <= 1 or dist is None or not dist.is_initialized():
            return
        dist.barrier()

    # Utility helpers ---------------------------------------------------
    def _tensor_to_torch(self, tensor: Any) -> tuple["torch.Tensor", Callable[["torch.Tensor"], None]]:
        if torch is None:
            raise RuntimeError("torch is required for TorchDistributedBackend")

        from typing import TYPE_CHECKING

        if TYPE_CHECKING:  # pragma: no cover - typing only
            from pysml.tensor import Tensor  # noqa: F401

        device = _torch_device_for_tensor(tensor)
        np_array = _to_numpy_array(tensor)
        torch_tensor = torch.as_tensor(np_array, device=device)

        def restore(updated: "torch.Tensor") -> None:
            _update_tensor_from_torch(tensor, updated)

        return torch_tensor, restore

    def _perform_collective(
        self,
        tensor: Any,
        call: Callable[["torch.Tensor"], None],
    ) -> Any:
        if self.world_size <= 1 or dist is None or not dist.is_initialized():
            return tensor

        torch_tensor, restore = self._tensor_to_torch(tensor)
        call(torch_tensor)
        restore(torch_tensor)
        return tensor

    def all_reduce(self, tensor: Any, op: str = "sum") -> Any:
        def _call(t: "torch.Tensor") -> None:
            reduce_op = _map_reduce_op(op)
            dist.all_reduce(t, op=reduce_op)

        return self._perform_collective(tensor, _call)

    def broadcast(self, tensor: Any, src: int = 0) -> Any:
        def _call(t: "torch.Tensor") -> None:
            dist.broadcast(t, src=src)

        return self._perform_collective(tensor, _call)

    def all_gather(self, tensor: Any) -> list[Any]:
        if self.world_size <= 1 or dist is None or not dist.is_initialized():
            return [tensor]

        torch_tensor, _ = self._tensor_to_torch(tensor)
        gather_list = [torch.zeros_like(torch_tensor) for _ in range(self.world_size)]
        dist.all_gather(gather_list, torch_tensor)
        results = []
        for chunk in gather_list:
            clone = _clone_tensor_like(tensor)
            _update_tensor_from_torch(clone, chunk)
            results.append(clone)
        return results

    def reduce_scatter(self, tensors: list[Any], op: str = "sum") -> Any:
        if not tensors:
            raise ValueError("reduce_scatter expects at least one tensor")

        if self.world_size <= 1 or dist is None or not dist.is_initialized():
            return tensors[0]

        torch_tensors = []
        restores = []
        for tensor in tensors:
            torch_tensor, restore = self._tensor_to_torch(tensor)
            torch_tensors.append(torch_tensor)
            restores.append(restore)
        output = torch.zeros_like(torch_tensors[0])
        reduce_op = _map_reduce_op(op)
        dist.reduce_scatter(output, torch_tensors, op=reduce_op)
        restores[0](output)
        return tensors[0]


def _map_reduce_op(name: str):
    if dist is None:
        raise RuntimeError("torch.distributed is not available")
    mapping = {
        "sum": dist.ReduceOp.SUM,
        "avg": dist.ReduceOp.AVG,
        "mean": dist.ReduceOp.AVG,
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
        "product": dist.ReduceOp.PRODUCT,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported reduction op: {name}")
    return mapping[key]


def _torch_device_for_tensor(tensor: Any) -> "torch.device":
    if torch is None:
        raise RuntimeError("torch is required")

    device_str = getattr(tensor, "active_device", None)
    if isinstance(device_str, str):
        if device_str.startswith("cuda") and torch.cuda.is_available():
            idx = _parse_index(device_str)
            return torch.device("cuda", idx if idx is not None else torch.cuda.current_device())
        if device_str.startswith("xpu") and _has_xpu():
            idx = _parse_index(device_str)
            return torch.device("xpu", idx if idx is not None else 0)
    return torch.device("cpu")


def _parse_index(device: str) -> Optional[int]:
    if ":" not in device:
        return None
    _, value = device.split(":", 1)
    if not value:
        return None
    return int(value)


def _to_numpy_array(tensor: Any):
    import numpy as np

    data = getattr(tensor, "data", tensor)
    if hasattr(data, "get"):
        return data.get()
    if hasattr(data, "asnumpy"):
        return data.asnumpy()
    if hasattr(data, "to_numpy"):
        return data.to_numpy()
    return np.asarray(data)


def _update_tensor_from_torch(tensor: Any, value: "torch.Tensor") -> None:
    if hasattr(tensor, "data"):
        backend = getattr(tensor, "_backend", None)
        dtype = getattr(tensor, "_dtype", None)
        device = getattr(tensor, "device", None)
        if backend is not None and dtype is not None:
            converted = backend.convert(value.cpu().numpy(), dtype, device=device)
            copyto = getattr(backend, "copyto", None)
            if copyto is not None:
                copyto(tensor.data, converted)
            else:
                tensor.data = converted
    # Non-PySML tensors do not need in-place updates.


def _clone_tensor_like(source: Any) -> Any:
    if hasattr(source, "_new_like"):
        backend = getattr(source, "_backend", None)
        data = getattr(source, "data", None)
        requires_grad = getattr(source, "_requires_grad", False)
        if backend is not None and data is not None:
            copier = getattr(backend, "copy", None)
            if copier is not None:
                data = copier(data)
        return source._new_like(data, requires_grad=requires_grad)
    return source


def _has_xpu() -> bool:
    return bool(getattr(torch, "xpu", None) and torch.xpu.is_available()) if torch is not None else False


@contextlib.contextmanager
def _suppress_already_initialized():  # pragma: no cover - depends on environment
    try:
        yield
    except RuntimeError as exc:
        if "reinitialize" not in str(exc).lower():
            raise


__all__ = ["TorchDistributedBackend"]
