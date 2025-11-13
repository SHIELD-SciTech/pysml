"""Distributed data parallel module wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from pysml import autograd
from pysml.nn.module import Module, Parameter
from pysml.tensor import Tensor

from . import process_group
from .routing import get_communicator


@dataclass
class _BucketState:
    index: int
    params: List[Parameter]

    def __post_init__(self) -> None:
        self.pending: int = len(self.params)
        self.grad_ready: Dict[int, bool] = {id(p.data): False for p in self.params}
        self.reduced: bool = False
        self.queued: bool = False

    def reset(self) -> None:
        self.pending = len(self.params)
        self.reduced = False
        self.queued = False
        for key in self.grad_ready:
            self.grad_ready[key] = False


class DistributedDataParallel(Module):
    """Wrap a :class:`~pysml.nn.Module` and run gradient all-reduce automatically."""

    def __init__(
        self,
        module: Module,
        *,
        bucket_cap_mb: float = 25.0,
        average_gradients: bool = True,
        broadcast_buffers: bool = True,
        mixed_precision: bool = False,
        process_group_override=None,
    ) -> None:
        super().__init__()
        self.module = module
        self.average_gradients = average_gradients
        self.broadcast_buffers = broadcast_buffers
        self._use_mixed_precision = mixed_precision
        self._bucket_bytes_cap = max(int(bucket_cap_mb * 1024 * 1024), 1)
        self._delay_allreduce = 0
        self._hooks: List[Tensor._PostBackwardHookHandle] = []
        self._buckets: List[_BucketState] = []
        self._param_to_bucket: Dict[int, _BucketState] = {}
        self._bucket_structure_hash: Optional[int] = None

        if process_group_override is not None:
            self._communicator = process_group_override
        else:
            self._communicator = self._infer_communicator(module)
        self._world_size = self._communicator.world_size

        self._rebuild_buckets()
        if self.broadcast_buffers:
            self._broadcast_module_buffers()

    # ------------------------------------------------------------------
    # Helper properties
    # ------------------------------------------------------------------
    @property
    def communicator(self):
        return self._communicator

    @property
    def world_size(self) -> int:
        return self._world_size

    # ------------------------------------------------------------------
    # Bucket management
    # ------------------------------------------------------------------
    def _infer_communicator(self, module: Module):
        for param in module.parameters():
            return get_communicator(param.data)
        return process_group.get_backend()

    def _tensor_nbytes(self, tensor: Tensor) -> int:
        data = getattr(tensor, "data", None)
        if hasattr(data, "nbytes"):
            try:
                return int(data.nbytes)
            except TypeError:
                pass
        if hasattr(data, "size") and hasattr(data, "itemsize"):
            try:
                return int(data.size * data.itemsize)
            except TypeError:
                pass
        if hasattr(tensor, "shape"):
            numel = 1
            for dim in tensor.shape:
                numel *= int(dim)
            return int(numel) * 4
        return 0

    def _rebuild_buckets(self) -> None:
        params = [p for p in self.module.parameters() if getattr(p, "requires_grad", True)]
        structure_signature = hash(tuple(id(p) for p in params))
        if structure_signature == self._bucket_structure_hash and self._buckets:
            self._reset_bucket_state()
            return

        self._clear_autograd_hooks()
        self._buckets = []
        self._param_to_bucket.clear()

        current_bucket: List[Parameter] = []
        current_size = 0
        bucket_cap = self._bucket_bytes_cap

        for param in params:
            tensor = param.data if isinstance(param, Parameter) else param
            param_bytes = self._tensor_nbytes(tensor)
            if current_bucket and current_size + param_bytes > bucket_cap:
                bucket = _BucketState(len(self._buckets), current_bucket)
                self._register_bucket(bucket)
                current_bucket = []
                current_size = 0
            current_bucket.append(param)
            current_size += param_bytes

        if current_bucket:
            bucket = _BucketState(len(self._buckets), current_bucket)
            self._register_bucket(bucket)

        self._bucket_structure_hash = structure_signature
        self._register_autograd_hooks()

    def _register_bucket(self, bucket: _BucketState) -> None:
        self._buckets.append(bucket)
        for param in bucket.params:
            self._param_to_bucket[id(param.data)] = bucket

    def _clear_autograd_hooks(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def _register_autograd_hooks(self) -> None:
        for bucket in self._buckets:
            for param in bucket.params:
                tensor_id = id(param.data)

                def hook_closure(bucket_state: _BucketState, t_id: int):
                    def _hook(tensor: Tensor) -> None:
                        self._mark_parameter_ready(bucket_state, t_id)

                    def _reset() -> None:
                        self._reset_bucket_state(bucket_state)

                    _hook.reset = _reset  # type: ignore[attr-defined]
                    return _hook

                hook = hook_closure(bucket, tensor_id)
                handle = autograd.register_post_backward_hook(param.data, hook)
                self._hooks.append(handle)

    def _reset_bucket_state(self, bucket: Optional[_BucketState] = None) -> None:
        targets = self._buckets if bucket is None else [bucket]
        for target in targets:
            target.reset()

    def _reset_all(self) -> None:
        self._reset_bucket_state()

    # ------------------------------------------------------------------
    # Gradient hooks
    # ------------------------------------------------------------------
    def _mark_parameter_ready(self, bucket: _BucketState, tensor_id: int) -> None:
        if bucket.grad_ready.get(tensor_id, False):
            return
        bucket.grad_ready[tensor_id] = True
        bucket.pending = max(bucket.pending - 1, 0)

        if bucket.pending == 0:
            if self._delay_allreduce > 0:
                bucket.queued = True
            else:
                self._all_reduce_bucket(bucket)

    def _prepare_gradient_tensor(self, grad: Tensor):
        def _identity() -> None:
            return None

        if not self._use_mixed_precision:
            return grad, _identity

        dtype_obj = getattr(grad, "_dtype", None)
        precision = getattr(dtype_obj, "precission", None)
        if precision in {"fp16", "bf16"} and hasattr(grad, "to"):
            try:
                from pysml import dtype as _dtype_mod

                fp32_tensor = grad.to(dtype=_dtype_mod.fp32())

                def _restore() -> None:
                    if fp32_tensor is grad:
                        return
                    converted = fp32_tensor.to(dtype=dtype_obj)
                    grad.data = converted.data

                return fp32_tensor, _restore
            except Exception:
                pass

        return grad, _identity

    def _all_reduce_bucket(self, bucket: _BucketState) -> None:
        if bucket.reduced:
            return

        if self._world_size <= 1:
            bucket.reduced = True
            bucket.queued = False
            return

        for param in bucket.params:
            grad = param.grad
            if grad is None:
                continue

            tensor, restore = self._prepare_gradient_tensor(grad)
            self._communicator.all_reduce(tensor, op="sum")
            if self.average_gradients and self._world_size > 0:
                tensor.data = tensor._backend.divide(tensor.data, self._world_size)
            restore()

        bucket.reduced = True
        bucket.queued = False

    def _flush_queued_buckets(self) -> None:
        for bucket in self._buckets:
            if bucket.pending == 0 and (bucket.queued or not bucket.reduced):
                self._all_reduce_bucket(bucket)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):  # type: ignore[override]
        self._rebuild_buckets()
        self._reset_all()
        return self.module(*args, **kwargs)

    def no_sync(self):
        return _NoSync(self)

    def synchronize_gradients(self) -> None:
        self._flush_queued_buckets()

    def train(self, mode: bool = True):  # type: ignore[override]
        self.module.train(mode)
        return super().train(mode)

    def eval(self):  # type: ignore[override]
        return self.train(False)

    def _broadcast_module_buffers(self) -> None:
        if self._world_size <= 1:
            return
        for _, buffer in self.module.named_buffers():
            if isinstance(buffer, Tensor):
                self._communicator.broadcast(buffer, src=0)


class _NoSync:
    def __init__(self, module: DistributedDataParallel) -> None:
        self._module = module

    def __enter__(self):
        self._module._delay_allreduce += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self._module._delay_allreduce = max(self._module._delay_allreduce - 1, 0)
        if self._module._delay_allreduce == 0:
            self._module._flush_queued_buckets()


__all__ = ["DistributedDataParallel"]
