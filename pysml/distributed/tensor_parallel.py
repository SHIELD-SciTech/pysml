"""Tensor parallel group management and helpers."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from . import collectives, process_group
from .. import engine
from ..tensor import Tensor
from ..nn.module import Parameter


@dataclass(frozen=True)
class TensorParallelGroup:
    """Light-weight description of a tensor parallel subgroup."""

    mode: str
    size: int
    ranks: Tuple[int, ...]
    dims: Tuple[int, ...]
    rank: int
    group_index: int
    world_size: int
    experts_per_rank: Optional[int] = None

    @property
    def local_rank(self) -> int:
        return self.rank % max(1, self.size)


_DEFAULT_GROUP: Optional[TensorParallelGroup] = None


def _balanced_split_sizes(total: int, num_chunks: int) -> List[int]:
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    base = total // num_chunks
    remainder = total % num_chunks
    sizes = [base] * num_chunks
    for idx in range(remainder):
        sizes[idx] += 1
    return sizes


def _partition_range(total: int, num_chunks: int, index: int) -> tuple[int, int]:
    sizes = _balanced_split_sizes(total, num_chunks)
    start = sum(sizes[:index])
    end = start + sizes[index]
    return start, end


def _infer_dims(mode: str, size: int, dims: Optional[Sequence[int]]) -> Tuple[int, ...]:
    if mode == "2d":
        if dims is not None:
            inferred = tuple(int(v) for v in dims)
        else:
            side = int(math.sqrt(size))
            while side > 1 and size % side != 0:
                side -= 1
            inferred = (side, size // side)
        if math.prod(inferred) != size:
            raise ValueError("Product of 2D tensor parallel dims must match the TP size")
        return inferred
    if dims is None:
        return (size,)
    inferred = tuple(int(v) for v in dims)
    if math.prod(inferred) != size:
        raise ValueError("Provided dims do not multiply to tensor parallel size")
    return inferred


def init_tensor_parallel(
    *,
    tp_size: Optional[int] = None,
    mode: Optional[str] = None,
    dims: Optional[Sequence[int]] = None,
    experts_per_rank: Optional[int] = None,
) -> TensorParallelGroup:
    """Initialise tensor parallel metadata.

    The helper slices the default process group into equally sized tensor
    parallel groups. For single-process execution the function simply records
    the configuration and returns a degenerate group of size ``1``.
    """

    global _DEFAULT_GROUP

    if not process_group.is_initialized():
        process_group.lazy_init_from_env()

    backend = process_group.get_backend()
    world_size = backend.world_size
    rank = backend.rank

    env_size = os.getenv("PYSML_TP_SIZE")
    if tp_size is None and env_size is not None:
        tp_size = int(env_size)
    if tp_size is None:
        tp_size = world_size
    if tp_size <= 0:
        raise ValueError("tp_size must be positive")

    tp_size = min(tp_size, world_size)
    num_groups = max(1, world_size // tp_size)
    group_index = min(rank // tp_size, num_groups - 1)
    ranks = tuple(range(group_index * tp_size, min(world_size, (group_index + 1) * tp_size)))

    env_mode = os.getenv("PYSML_TP_MODE", "1d")
    if mode is None:
        mode = env_mode
    mode = mode.lower()

    env_dims = os.getenv("PYSML_TP_DIMS")
    if dims is None and env_dims:
        dims = tuple(int(part) for part in env_dims.split("x") if part)

    resolved_dims = _infer_dims(mode, len(ranks) or 1, dims)

    group = TensorParallelGroup(
        mode=mode,
        size=len(ranks) or 1,
        ranks=ranks or (0,),
        dims=resolved_dims,
        rank=rank,
        group_index=group_index,
        world_size=world_size,
        experts_per_rank=experts_per_rank,
    )
    _DEFAULT_GROUP = group
    return group


def get_tensor_parallel_group() -> TensorParallelGroup:
    global _DEFAULT_GROUP
    if _DEFAULT_GROUP is None:
        _DEFAULT_GROUP = init_tensor_parallel()
    return _DEFAULT_GROUP


def get_tensor_parallel_world_size() -> int:
    return get_tensor_parallel_group().size


def get_tensor_parallel_rank() -> int:
    group = get_tensor_parallel_group()
    return group.rank % max(1, group.size)


def _filter_group_members(items: Sequence, group: TensorParallelGroup) -> list:
    if len(items) == group.world_size:
        return [items[idx] for idx in group.ranks]
    return list(items)


def gather_from_tensor_parallel_region(tensor: Tensor, *, dim: int = -1, group: Optional[TensorParallelGroup] = None) -> Tensor:
    group = group or get_tensor_parallel_group()
    if group.size == 1:
        return tensor
    gathered = collectives.all_gather(tensor)
    relevant = _filter_group_members(gathered, group)
    return engine.concatenate(relevant, axis=dim)


def reduce_from_tensor_parallel_region(
    tensor: Tensor,
    *,
    op: str = "sum",
    group: Optional[TensorParallelGroup] = None,
) -> Tensor:
    group = group or get_tensor_parallel_group()
    if group.size == 1:
        return tensor

    gathered = collectives.all_gather(tensor)
    tensors = _filter_group_members(gathered, group)
    result = tensors[0]
    if op == "max":
        for shard in tensors[1:]:
            result = engine.maximum(result, shard)
        return result

    for shard in tensors[1:]:
        result = engine.add(result, shard)
    if op == "mean":
        divisor = Tensor.__new__(Tensor)
        backend = result._backend
        divisor._backend = backend
        divisor._dtype = result._dtype
        divisor._requires_grad = False
        divisor._grad = None
        divisor.device = result.device
        divisor.active_device = result.active_device
        divisor.data = backend.asarray([group.size])
        result = engine.divide(result, divisor)
    return result


def scatter_to_tensor_parallel_region(
    tensor: Tensor,
    *,
    dim: int = -1,
    group: Optional[TensorParallelGroup] = None,
) -> Tensor:
    group = group or get_tensor_parallel_group()
    if group.size == 1:
        return tensor

    sizes = _balanced_split_sizes(tensor.shape[dim], group.size)
    chunks = engine.split(tensor, sizes, dim=dim)
    local_rank = get_tensor_parallel_rank()
    return chunks[local_rank]


def partition_linear_features(
    *, total_features: int, group: Optional[TensorParallelGroup] = None
) -> tuple[int, int, int]:
    group = group or get_tensor_parallel_group()
    start, end = _partition_range(total_features, group.size, get_tensor_parallel_rank())
    return start, end, end - start


def register_sharded_parameter(param: Parameter, *, reduce: str = "sum", group: Optional[TensorParallelGroup] = None) -> None:
    """Attach a gradient reduction hook to ``param`` for shared shards."""

    tensor = param.data if isinstance(param, Parameter) else param
    if not isinstance(tensor, Tensor):
        raise TypeError("register_sharded_parameter expects a Parameter or Tensor")

    group = group or get_tensor_parallel_group()
    if group.size == 1:
        return

    class _Reducer:
        def __init__(self, reduce: str, group: TensorParallelGroup) -> None:
            self.reduce = reduce
            self.group = group

        def __call__(self, owner: Tensor) -> None:
            if owner.grad is None:
                return
            reduced = reduce_from_tensor_parallel_region(owner.grad, op="sum", group=self.group)
            if self.reduce == "mean":
                backend = reduced._backend
                scale = backend.asarray([self.group.size])
                factor = owner.grad._new_like(scale, requires_grad=False)
                factor.data = scale
                reduced = engine.divide(reduced, factor)
            owner.grad = reduced

        def reset(self) -> None:
            pass

    tensor.register_post_backward_hook(_Reducer(reduce, group))


__all__ = [
    "TensorParallelGroup",
    "init_tensor_parallel",
    "get_tensor_parallel_group",
    "get_tensor_parallel_world_size",
    "get_tensor_parallel_rank",
    "gather_from_tensor_parallel_region",
    "reduce_from_tensor_parallel_region",
    "scatter_to_tensor_parallel_region",
    "partition_linear_features",
    "register_sharded_parameter",
]
