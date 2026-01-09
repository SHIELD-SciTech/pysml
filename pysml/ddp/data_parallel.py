

from __future__ import annotations

from typing import Optional, List, Any, Dict, Callable
import threading

from ..nn.module import Module, Parameter
from ..tensor import Tensor
from .comm import (
    CommunicationBackend,
    get_process_group,
    get_rank,
    get_world_size,
    is_initialized,
)

class DistributedDataParallel(Module):

    
    def __init__(
        self,
        module: Module,
        device_ids: Optional[List[int]] = None,
        output_device: Optional[int] = None,
        broadcast_buffers: bool = True,
        find_unused_parameters: bool = False,
        gradient_as_bucket_view: bool = False,
        bucket_cap_mb: float = 25.0,
    ):
        super().__init__()
        
        if not is_initialized():
            raise RuntimeError(
                "Default process group has not been initialized. "
                "Call init_process_group first."
            )
        
        self.module = module
        self.broadcast_buffers = broadcast_buffers
        self.find_unused_parameters = find_unused_parameters
        self.bucket_cap_mb = bucket_cap_mb
        
        self._process_group = get_process_group()
        self._rank = get_rank()
        self._world_size = get_world_size()
        
        # Synchronize model parameters from rank 0
        self._sync_params()
        
        # Register gradient synchronization hooks
        self._grad_hooks = []
        self._register_grad_hooks()
        
        # Gradient buckets for efficient all-reduce
        self._buckets: List[List[Parameter]] = []
        self._bucket_indices: Dict[int, int] = {}
        self._pending_grads: Dict[int, Any] = {}
        self._grad_ready_count = 0
        self._grad_lock = threading.Lock()
        
        self._build_buckets()
    
    def _sync_params(self):

        for name, param in self.module.named_parameters():
            if param is not None:
                synced_data = self._process_group.broadcast(param.data, src=0)
                if self._rank != 0:
                    param.data = synced_data
        
        if self.broadcast_buffers:
            for name, buffer in self.module.named_buffers():
                if buffer is not None:
                    synced_data = self._process_group.broadcast(buffer.data, src=0)
                    if self._rank != 0:
                        buffer.data = synced_data
    
    def _build_buckets(self):

        # Group parameters into buckets by size
        params = list(self.module.parameters())
        
        current_bucket = []
        current_size = 0
        bucket_size_bytes = self.bucket_cap_mb * 1024 * 1024
        
        for i, param in enumerate(params):
            param_size = param.numel() * 4  # Assume 4 bytes per element
            
            if current_size + param_size > bucket_size_bytes and current_bucket:
                self._buckets.append(current_bucket)
                current_bucket = []
                current_size = 0
            
            current_bucket.append(param)
            self._bucket_indices[id(param)] = len(self._buckets)
            current_size += param_size
        
        if current_bucket:
            self._buckets.append(current_bucket)
    
    def _register_grad_hooks(self):

        for param in self.module.parameters():
            if param.requires_grad:
                # Register post-accumulate-grad hook
                hook_handle = param.register_post_backward_hook(
                    self._make_grad_hook(param)
                )
                self._grad_hooks.append(hook_handle)
    
    def _make_grad_hook(self, param: Parameter) -> Callable:

        def hook(p: Tensor):
            if p.grad is None:
                return
            
            with self._grad_lock:
                self._pending_grads[id(param)] = p.grad
                self._grad_ready_count += 1
                
                # Check if all gradients in bucket are ready
                bucket_idx = self._bucket_indices.get(id(param), 0)
                bucket = self._buckets[bucket_idx] if bucket_idx < len(self._buckets) else []
                
                bucket_ready = all(
                    id(bp) in self._pending_grads 
                    for bp in bucket 
                    if bp.requires_grad
                )
                
                if bucket_ready:
                    self._sync_bucket(bucket_idx)
        
        return hook
    
    def _sync_bucket(self, bucket_idx: int):

        if bucket_idx >= len(self._buckets):
            return
        
        bucket = self._buckets[bucket_idx]
        
        for param in bucket:
            if id(param) in self._pending_grads:
                grad = self._pending_grads[id(param)]
                
                # Get gradient data
                if isinstance(grad, Tensor):
                    grad_data = grad.data
                else:
                    grad_data = grad
                
                # All-reduce gradient (average across ranks)
                synced_grad = self._process_group.all_reduce(grad_data, op="mean")
                
                # Update gradient
                if isinstance(param.grad, Tensor):
                    param.grad.data = synced_grad
                else:
                    param.grad = param._new_like(synced_grad, requires_grad=False)
                
                del self._pending_grads[id(param)]
    
    def forward(self, *args, **kwargs):

        # Sync buffers if needed
        if self.broadcast_buffers and self.training:
            self._sync_buffers()
        
        # Reset gradient tracking
        with self._grad_lock:
            self._pending_grads.clear()
            self._grad_ready_count = 0
        
        return self.module(*args, **kwargs)
    
    def _sync_buffers(self):

        for name, buffer in self.module.named_buffers():
            if buffer is not None:
                synced_data = self._process_group.broadcast(buffer.data, src=0)
                if self._rank != 0:
                    buffer.data = synced_data
    
    def sync_gradients(self):

        for param in self.module.parameters():
            if param.grad is not None:
                grad_data = param.grad.data if isinstance(param.grad, Tensor) else param.grad
                synced_grad = self._process_group.all_reduce(grad_data, op="mean")
                
                if isinstance(param.grad, Tensor):
                    param.grad.data = synced_grad
                else:
                    param.grad = param._new_like(synced_grad, requires_grad=False)
    
    def state_dict(self, *args, **kwargs):

        return self.module.state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):

        return self.module.load_state_dict(*args, **kwargs)
    
    def parameters(self, recurse: bool = True):

        return self.module.parameters(recurse=recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):

        return self.module.named_parameters(prefix=prefix, recurse=recurse)
    
    def train(self, mode: bool = True):

        self.training = mode
        self.module.train(mode)
        return self
    
    def eval(self):

        return self.train(False)
    
    def __repr__(self):
        return f"DistributedDataParallel(\n  (module): {self.module}\n)"

# Alias for compatibility
DDP = DistributedDataParallel

class DataParallelContext:

    
    def __init__(self):
        self._process_group = get_process_group()
        self._rank = get_rank()
        self._world_size = get_world_size()
    
    def scatter_batch(self, batch: Tensor, dim: int = 0) -> Tensor:

        if self._world_size == 1:
            return batch
        
        # Split batch
        batch_size = batch.shape[dim]
        chunk_size = batch_size // self._world_size
        
        start_idx = self._rank * chunk_size
        end_idx = start_idx + chunk_size
        
        # Handle last rank getting remainder
        if self._rank == self._world_size - 1:
            end_idx = batch_size
        
        # Create slice
        slices = [slice(None)] * batch.ndim
        slices[dim] = slice(start_idx, end_idx)
        
        return batch[tuple(slices)]
    
    def gather_outputs(self, output: Tensor, dim: int = 0) -> Tensor:

        if self._world_size == 1:
            return output
        
        # All-gather outputs
        all_outputs = self._process_group.all_gather(output)
        
        # Concatenate along dimension
        backend = output._backend if hasattr(output, '_backend') else self._process_group._backend
        return backend.concatenate(all_outputs, axis=dim)
    
    def reduce_loss(self, loss: Tensor, op: str = "mean") -> Tensor:

        if self._world_size == 1:
            return loss
        
        loss_data = loss.data if isinstance(loss, Tensor) else loss
        reduced = self._process_group.all_reduce(loss_data, op=op)
        
        if isinstance(loss, Tensor):
            return loss._new_like(reduced, requires_grad=loss._requires_grad)
        return reduced

def get_data_parallel_context() -> DataParallelContext:

    return DataParallelContext()

__all__ = [
    'DistributedDataParallel',
    'DDP',
    'DataParallelContext',
    'get_data_parallel_context',
]
