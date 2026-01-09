

from __future__ import annotations

import os
import sys
import subprocess
import socket
import time
from typing import Optional, List, Callable, Any, Dict
from dataclasses import dataclass
import multiprocessing as mp

@dataclass
class LaunchConfig:

    world_size: int
    master_addr: str = "localhost"
    master_port: int = 29500
    backend: str = "socket"  # 'socket', 'nccl', 'gloo'
    
    # Device configuration
    devices: Optional[List[str]] = None  # e.g., ['cuda:0', 'cuda:1', 'xpu:0']
    
    # Multi-node configuration
    nnodes: int = 1
    node_rank: int = 0
    nproc_per_node: Optional[int] = None

def get_free_port() -> int:

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def _worker_fn(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    device: str,
    fn: Callable,
    args: tuple,
    kwargs: dict,
):

    from .comm import init_process_group
    
    # Parse device
    device_type = device.split(':')[0]
    device_id = int(device.split(':')[1]) if ':' in device else 0
    
    # Initialize process group
    init_process_group(
        rank=rank,
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
        device_type=device_type,
        device_id=device_id,
    )
    
    # Call user function
    fn(rank, world_size, *args, **kwargs)

def launch(
    fn: Callable,
    config: LaunchConfig,
    args: tuple = (),
    kwargs: Optional[Dict] = None,
):

    if kwargs is None:
        kwargs = {}
    
    world_size = config.world_size
    
    # Determine devices
    if config.devices is None:
        # Auto-detect devices
        devices = []
        
        # Try CUDA first
        try:
            from ..cuda import backend as cuda_backend
            num_cuda = cuda_backend.device_count()
            for i in range(min(num_cuda, world_size)):
                devices.append(f"cuda:{i}")
        except Exception:
            num_cuda = 0
        
        # Then XPU
        if len(devices) < world_size:
            try:
                from ..xpu import backend as xpu_backend
                num_xpu = xpu_backend.device_count()
                for i in range(min(num_xpu, world_size - len(devices))):
                    devices.append(f"xpu:{i}")
            except Exception:
                pass
        
        # Fall back to CPU
        while len(devices) < world_size:
            devices.append("cpu")
    else:
        devices = config.devices
    
    if len(devices) < world_size:
        raise ValueError(f"Not enough devices ({len(devices)}) for world_size ({world_size})")
    
    # Spawn processes
    processes = []
    
    ctx = mp.get_context('spawn')
    
    for rank in range(world_size):
        p = ctx.Process(
            target=_worker_fn,
            args=(
                rank,
                world_size,
                config.master_addr,
                config.master_port,
                devices[rank],
                fn,
                args,
                kwargs,
            )
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes
    for p in processes:
        p.join()
    
    # Check for errors
    for rank, p in enumerate(processes):
        if p.exitcode != 0:
            raise RuntimeError(f"Process {rank} exited with code {p.exitcode}")

def run_rank(
    fn: Callable,
    rank: int,
    world_size: int,
    device: str,
    master_addr: str = "localhost",
    master_port: int = 29500,
    args: tuple = (),
    kwargs: Optional[Dict] = None,
):

    if kwargs is None:
        kwargs = {}
    
    _worker_fn(
        rank=rank,
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
        device=device,
        fn=fn,
        args=args,
        kwargs=kwargs,
    )

class DistributedSampler:

    
    def __init__(
        self,
        dataset_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        from .comm import get_rank, get_world_size, is_initialized
        
        if num_replicas is None:
            num_replicas = get_world_size() if is_initialized() else 1
        if rank is None:
            rank = get_rank() if is_initialized() else 0
        
        self.dataset_size = dataset_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        
        # Calculate samples per rank
        if drop_last:
            self.num_samples = dataset_size // num_replicas
        else:
            self.num_samples = (dataset_size + num_replicas - 1) // num_replicas
        
        self.total_size = self.num_samples * num_replicas
    
    def __iter__(self):
        import numpy as np
        
        # Generate indices
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            indices = rng.permutation(self.dataset_size).tolist()
        else:
            indices = list(range(self.dataset_size))
        
        # Pad to make divisible
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                indices += indices[:padding_size]
        
        # Subsample for this rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        
        return iter(indices)
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch: int):

        self.epoch = epoch

def all_reduce_dict(data: Dict[str, Any], op: str = "mean") -> Dict[str, Any]:

    from .comm import get_process_group, is_initialized
    
    if not is_initialized():
        return data
    
    pg = get_process_group()
    result = {}
    
    for key, value in data.items():
        if hasattr(value, 'data'):
            result[key] = pg.all_reduce(value.data, op=op)
        else:
            result[key] = pg.all_reduce(value, op=op)
    
    return result

def broadcast_object(obj: Any, src: int = 0) -> Any:

    import pickle
    from .comm import get_process_group, get_rank, is_initialized
    
    if not is_initialized():
        return obj
    
    pg = get_process_group()
    rank = get_rank()
    
    if rank == src:
        data = pickle.dumps(obj)
        # Send size first
        size = len(data)
    else:
        size = None
        data = None
    
    # Broadcast size
    import numpy as np
    size_arr = np.array([size if size is not None else 0], dtype=np.int64)
    size_arr = pg.broadcast(size_arr, src=src)
    size = int(size_arr[0])
    
    # Broadcast data
    if rank == src:
        data_arr = np.frombuffer(data, dtype=np.uint8)
    else:
        data_arr = np.zeros(size, dtype=np.uint8)
    
    data_arr = pg.broadcast(data_arr, src=src)
    
    if rank != src:
        obj = pickle.loads(data_arr.tobytes())
    
    return obj

def barrier():

    from .comm import get_process_group, is_initialized
    
    if is_initialized():
        get_process_group().barrier()

def print_rank0(*args, **kwargs):

    from .comm import get_rank, is_initialized
    
    if not is_initialized() or get_rank() == 0:
        print(*args, **kwargs)

__all__ = [
    'LaunchConfig',
    'get_free_port',
    'launch',
    'run_rank',
    'DistributedSampler',
    'all_reduce_dict',
    'broadcast_object',
    'barrier',
    'print_rank0',
]
