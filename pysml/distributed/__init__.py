"""Distributed training utilities for PySML."""

from __future__ import annotations

from . import collectives
from .collectives import (
    all_gather,
    all_reduce,
    barrier,
    broadcast,
    gather,
    recv,
    reduce_scatter,
    scatter,
    send,
)
from .ddp import DistributedDataParallel
from .tensor_parallel import (
    TensorParallelGroup,
    gather_from_tensor_parallel_region,
    get_tensor_parallel_group,
    get_tensor_parallel_rank,
    get_tensor_parallel_world_size,
    init_tensor_parallel,
    partition_linear_features,
    reduce_from_tensor_parallel_region,
    register_sharded_parameter,
    scatter_to_tensor_parallel_region,
)
from .process_group import (
    get_backend,
    get_rank,
    get_world_size,
    init_process_group,
    is_initialized,
    lazy_init_from_env,
    shutdown,
)
from .routing import get_communicator

__all__ = [
    "init_process_group",
    "shutdown",
    "is_initialized",
    "get_backend",
    "get_rank",
    "get_world_size",
    "lazy_init_from_env",
    "get_communicator",
    "all_reduce",
    "broadcast",
    "all_gather",
    "reduce_scatter",
    "barrier",
    "send",
    "recv",
    "gather",
    "scatter",
    "DistributedDataParallel",
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
