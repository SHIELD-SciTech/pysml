

from __future__ import annotations

from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import math

from ..nn.module import Module
from ..tensor import Tensor
from .comm import (
    CommunicationBackend,
    get_process_group,
    get_rank,
    get_world_size,
    init_process_group,
    is_initialized,
)
from .data_parallel import DistributedDataParallel
from .pipeline_parallel import PipelineParallel, PipelineSchedule, DistributedPipelineParallel

@dataclass
class DeviceConfig:

    device_type: str  # 'cuda' or 'xpu'
    device_id: int
    rank: int
    hostname: str = "localhost"
    
    @property
    def device_string(self) -> str:
        return f"{self.device_type}:{self.device_id}"

@dataclass
class ParallelConfig:

    # Data parallelism
    data_parallel_size: int = 1
    
    # Pipeline parallelism  
    pipeline_parallel_size: int = 1
    num_micro_batches: int = 1
    pipeline_schedule: PipelineSchedule = PipelineSchedule.GPIPE
    
    # Tensor parallelism (future)
    tensor_parallel_size: int = 1
    
    # Device mapping
    devices: List[DeviceConfig] = field(default_factory=list)
    
    # Communication
    master_addr: str = "localhost"
    master_port: int = 29500
    
    @property
    def world_size(self) -> int:
        return self.data_parallel_size * self.pipeline_parallel_size * self.tensor_parallel_size
    
    def get_data_parallel_group(self, rank: int) -> List[int]:

        pp_size = self.pipeline_parallel_size
        dp_size = self.data_parallel_size
        
        # Ranks are organized as: [dp_rank * pp_size + pp_rank]
        pp_rank = rank % pp_size
        group = [dp_rank * pp_size + pp_rank for dp_rank in range(dp_size)]
        return group
    
    def get_pipeline_parallel_group(self, rank: int) -> List[int]:

        pp_size = self.pipeline_parallel_size
        dp_size = self.data_parallel_size
        
        dp_rank = rank // pp_size
        group = [dp_rank * pp_size + pp_rank for pp_rank in range(pp_size)]
        return group
    
    def get_data_parallel_rank(self, global_rank: int) -> int:

        return global_rank // self.pipeline_parallel_size
    
    def get_pipeline_parallel_rank(self, global_rank: int) -> int:

        return global_rank % self.pipeline_parallel_size

class HybridParallel(Module):

    
    def __init__(
        self,
        stages: List[Module],
        config: ParallelConfig,
    ):
        super().__init__()
        
        self.config = config
        self.stages = stages
        
        if not is_initialized():
            raise RuntimeError("Process group not initialized")
        
        self._rank = get_rank()
        self._world_size = get_world_size()
        self._process_group = get_process_group()
        
        # Validate configuration
        if len(stages) != config.pipeline_parallel_size:
            raise ValueError(
                f"Number of stages ({len(stages)}) must match "
                f"pipeline_parallel_size ({config.pipeline_parallel_size})"
            )
        
        if config.world_size != self._world_size:
            raise ValueError(
                f"Config world size ({config.world_size}) doesn't match "
                f"actual world size ({self._world_size})"
            )
        
        # Determine this rank's role
        self._dp_rank = config.get_data_parallel_rank(self._rank)
        self._pp_rank = config.get_pipeline_parallel_rank(self._rank)
        self._dp_group = config.get_data_parallel_group(self._rank)
        self._pp_group = config.get_pipeline_parallel_group(self._rank)
        
        # Get this rank's stage
        self._local_stage = stages[self._pp_rank]
        
        # Move to appropriate device
        if config.devices and self._rank < len(config.devices):
            device = config.devices[self._rank].device_string
        else:
            # Default device assignment
            device = f"cuda:{self._rank % 8}"  # Assume up to 8 GPUs
        
        self._device = device
        self._local_stage.to(device)
        
        # Storage for pipeline execution
        self._activations: Dict[int, Tensor] = {}
        self._inputs: Dict[int, Tensor] = {}
    
    @property
    def local_stage(self) -> Module:

        return self._local_stage
    
    def _is_first_stage(self) -> bool:
        return self._pp_rank == 0
    
    def _is_last_stage(self) -> bool:
        return self._pp_rank == self.config.pipeline_parallel_size - 1
    
    def _get_prev_rank(self) -> Optional[int]:
        if self._is_first_stage():
            return None
        return self._pp_group[self._pp_rank - 1]
    
    def _get_next_rank(self) -> Optional[int]:
        if self._is_last_stage():
            return None
        return self._pp_group[self._pp_rank + 1]
    
    def _sync_gradients_dp(self):

        if self.config.data_parallel_size <= 1:
            return
        
        # All-reduce gradients within DP group
        for param in self._local_stage.parameters():
            if param.grad is not None:
                grad_data = param.grad.data if isinstance(param.grad, Tensor) else param.grad
                
                # Manual all-reduce within DP group
                # Sum gradients from all ranks in DP group
                all_grads = self._process_group.all_gather(grad_data)
                
                # Average only grads from DP group members
                dp_grads = [all_grads[r] for r in self._dp_group]
                
                backend = self._process_group._backend
                summed = dp_grads[0]
                for g in dp_grads[1:]:
                    summed = backend.add(summed, g)
                averaged = backend.divide(summed, len(dp_grads))
                
                if isinstance(param.grad, Tensor):
                    param.grad.data = averaged
                else:
                    param.grad = averaged
    
    def forward(self, x: Optional[Tensor] = None) -> Optional[Tensor]:

        self._activations.clear()
        self._inputs.clear()
        
        num_micro_batches = self.config.num_micro_batches
        outputs = []
        
        for mb_id in range(num_micro_batches):
            # Get input for this micro-batch
            if self._is_first_stage():
                if x is None:
                    raise ValueError("First stage requires input")
                
                batch_size = x.shape[0]
                mb_size = batch_size // num_micro_batches
                start = mb_id * mb_size
                end = start + mb_size if mb_id < num_micro_batches - 1 else batch_size
                current = x[start:end]
            else:
                # Receive from previous pipeline stage
                prev_rank = self._get_prev_rank()
                current_data = self._process_group.recv(prev_rank)
                current = Tensor(current_data, device=self._device, requires_grad=True)
            
            # Store input
            self._inputs[mb_id] = current
            
            # Forward through local stage
            output = self._local_stage(current)
            
            # Store activation
            self._activations[mb_id] = output
            
            # Send to next stage or collect output
            if not self._is_last_stage():
                next_rank = self._get_next_rank()
                self._process_group.send(output, next_rank)
            else:
                outputs.append(output)
        
        # Last stage returns merged outputs
        if self._is_last_stage() and outputs:
            backend = outputs[0]._backend
            data_list = [o.data for o in outputs]
            merged = backend.concatenate(data_list, axis=0)
            return outputs[0]._new_like(merged, requires_grad=True)
        
        return None
    
    def backward(self, grad_output: Optional[Tensor] = None):

        num_micro_batches = self.config.num_micro_batches
        
        for mb_id in reversed(range(num_micro_batches)):
            # Get gradient for this micro-batch
            if self._is_last_stage():
                if grad_output is not None:
                    batch_size = grad_output.shape[0]
                    mb_size = batch_size // num_micro_batches
                    start = mb_id * mb_size
                    end = start + mb_size if mb_id < num_micro_batches - 1 else batch_size
                    grad = grad_output[start:end]
                else:
                    activation = self._activations[mb_id]
                    grad = activation._new_like(
                        activation._backend.ones(activation.shape),
                        requires_grad=False
                    )
            else:
                # Receive gradient from next stage
                next_rank = self._get_next_rank()
                grad_data = self._process_group.recv(next_rank)
                grad = Tensor(grad_data, device=self._device, requires_grad=False)
            
            # Backward through local stage
            activation = self._activations[mb_id]
            if activation._requires_grad:
                activation.backward(grad, retain_graph=True)
            
            # Send gradient to previous stage
            if not self._is_first_stage():
                prev_rank = self._get_prev_rank()
                input_tensor = self._inputs[mb_id]
                if input_tensor.grad is not None:
                    self._process_group.send(input_tensor.grad, prev_rank)
        
        # Synchronize gradients across data parallel group
        self._sync_gradients_dp()
    
    def parameters(self, recurse: bool = True):

        return self._local_stage.parameters(recurse=recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):

        return self._local_stage.named_parameters(prefix=prefix, recurse=recurse)
    
    def state_dict(self):

        return self._local_stage.state_dict()
    
    def load_state_dict(self, state_dict, strict=True):

        return self._local_stage.load_state_dict(state_dict, strict=strict)
    
    def train(self, mode: bool = True):
        self.training = mode
        self._local_stage.train(mode)
        return self
    
    def eval(self):
        return self.train(False)

def create_parallel_model(
    model_fn,
    config: ParallelConfig,
    split_points: Optional[List[str]] = None,
) -> HybridParallel:

    # Create base model
    model = model_fn()
    
    if config.pipeline_parallel_size == 1:
        # No pipeline parallelism, just wrap in DDP
        stages = [model]
    else:
        # Split model into stages
        if split_points is None:
            # Auto-split based on number of children
            children = list(model.children())
            num_stages = config.pipeline_parallel_size
            
            if len(children) < num_stages:
                raise ValueError(
                    f"Model has {len(children)} children but "
                    f"pipeline_parallel_size is {num_stages}"
                )
            
            # Distribute children across stages
            children_per_stage = len(children) // num_stages
            stages = []
            
            from ..nn.container import Sequential
            
            for i in range(num_stages):
                start = i * children_per_stage
                end = start + children_per_stage if i < num_stages - 1 else len(children)
                stage_modules = children[start:end]
                stages.append(Sequential(*stage_modules))
        else:
            # Use specified split points
            raise NotImplementedError("Custom split points not yet implemented")
    
    return HybridParallel(stages, config)

__all__ = [
    'DeviceConfig',
    'ParallelConfig',
    'HybridParallel',
    'create_parallel_model',
]
