

from __future__ import annotations

from typing import Optional, List, Dict, Any, Callable, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import threading
import queue

from ..nn.module import Module, Parameter
from ..tensor import Tensor
from .comm import (
    CommunicationBackend,
    get_process_group,
    get_rank,
    get_world_size,
    is_initialized,
)

class PipelineSchedule(Enum):

    GPIPE = "gpipe"           # All forwards, then all backwards
    ONE_F_ONE_B = "1f1b"      # Interleaved forward/backward
    INTERLEAVED = "interleaved"  # Interleaved 1F1B with multiple stages per rank

@dataclass
class PipelineStage:

    stage_id: int
    module: Module
    device: str
    device_id: int

@dataclass 
class MicroBatch:

    micro_batch_id: int
    data: Any
    target: Optional[Any] = None
    loss: Optional[Tensor] = None

class PipelineParallel(Module):

    
    def __init__(
        self,
        modules: List[Module],
        devices: Optional[List[str]] = None,
        chunks: int = 1,
        schedule: Union[str, PipelineSchedule] = PipelineSchedule.GPIPE,
        checkpoint_fn: Optional[Callable] = None,
    ):
        super().__init__()
        
        self.num_stages = len(modules)
        self.chunks = chunks
        self.checkpoint_fn = checkpoint_fn
        
        if isinstance(schedule, str):
            schedule = PipelineSchedule(schedule)
        self.schedule = schedule
        
        # Determine devices
        if devices is None:
            devices = [f"cuda:{i}" for i in range(self.num_stages)]
        
        if len(devices) != self.num_stages:
            raise ValueError(f"Number of devices ({len(devices)}) must match number of stages ({self.num_stages})")
        
        self.devices = devices
        
        # Create pipeline stages
        self.stages: List[PipelineStage] = []
        for i, (module, device) in enumerate(zip(modules, devices)):
            # Move module to device
            module.to(device)
            
            device_type = device.split(':')[0]
            device_id = int(device.split(':')[1]) if ':' in device else 0
            
            stage = PipelineStage(
                stage_id=i,
                module=module,
                device=device,
                device_id=device_id,
            )
            self.stages.append(stage)
            
            # Register as submodule
            self.add_module(f"stage_{i}", module)
        
        # Communication queues for pipeline
        self._forward_queues: List[queue.Queue] = [queue.Queue() for _ in range(self.num_stages)]
        self._backward_queues: List[queue.Queue] = [queue.Queue() for _ in range(self.num_stages)]
        
        # Storage for activations (needed for backward)
        self._activations: Dict[int, Dict[int, Tensor]] = {}  # stage_id -> {micro_batch_id -> activation}
        self._inputs: Dict[int, Dict[int, Tensor]] = {}  # stage_id -> {micro_batch_id -> input}
    
    def _get_backend(self, device: str):

        device_type = device.split(':')[0]
        if device_type == 'cuda':
            from ..cuda import backend
            return backend
        elif device_type == 'xpu':
            from ..xpu import backend
            return backend
        else:
            from ..cpu import backend
            return backend
    
    def _move_to_device(self, tensor: Tensor, device: str) -> Tensor:

        return tensor.to(device)
    
    def _split_batch(self, batch: Tensor) -> List[Tensor]:

        batch_size = batch.shape[0]
        micro_batch_size = batch_size // self.chunks
        
        micro_batches = []
        for i in range(self.chunks):
            start = i * micro_batch_size
            end = start + micro_batch_size if i < self.chunks - 1 else batch_size
            micro_batches.append(batch[start:end])
        
        return micro_batches
    
    def _merge_outputs(self, outputs: List[Tensor]) -> Tensor:

        if not outputs:
            return None
        
        backend = outputs[0]._backend
        data_list = [o.data for o in outputs]
        merged_data = backend.concatenate(data_list, axis=0)
        
        return outputs[0]._new_like(merged_data, requires_grad=any(o._requires_grad for o in outputs))
    
    def _forward_stage(self, stage: PipelineStage, x: Tensor, micro_batch_id: int) -> Tensor:

        # Store input for backward
        if stage.stage_id not in self._inputs:
            self._inputs[stage.stage_id] = {}
        self._inputs[stage.stage_id][micro_batch_id] = x
        
        # Forward through stage module
        if self.checkpoint_fn is not None:
            output = self.checkpoint_fn(stage.module, x)
        else:
            output = stage.module(x)
        
        # Store activation for backward
        if stage.stage_id not in self._activations:
            self._activations[stage.stage_id] = {}
        self._activations[stage.stage_id][micro_batch_id] = output
        
        return output
    
    def _backward_stage(self, stage: PipelineStage, grad_output: Tensor, micro_batch_id: int) -> Tensor:

        # Get stored activation
        activation = self._activations.get(stage.stage_id, {}).get(micro_batch_id)
        
        if activation is None:
            raise RuntimeError(f"No activation stored for stage {stage.stage_id}, micro-batch {micro_batch_id}")
        
        # Backward through activation
        if activation._requires_grad and activation._grad_fn is not None:
            activation.backward(grad_output, retain_graph=True)
        
        # Get gradient w.r.t. input
        input_tensor = self._inputs.get(stage.stage_id, {}).get(micro_batch_id)
        if input_tensor is not None and input_tensor.grad is not None:
            return input_tensor.grad
        
        return grad_output  # Pass through if no gradient computed
    
    def forward_gpipe(self, x: Tensor) -> Tensor:

        # Split into micro-batches
        micro_batches = self._split_batch(x)
        
        # Forward all micro-batches through all stages
        outputs = []
        
        for mb_id, micro_batch in enumerate(micro_batches):
            current = micro_batch
            
            for stage in self.stages:
                # Move to stage device
                current = self._move_to_device(current, stage.device)
                # Forward through stage
                current = self._forward_stage(stage, current, mb_id)
            
            outputs.append(current)
        
        # Merge outputs
        return self._merge_outputs(outputs)
    
    def forward_1f1b(self, x: Tensor) -> Tensor:

        micro_batches = self._split_batch(x)
        num_micro_batches = len(micro_batches)
        num_stages = self.num_stages
        
        # Track state
        outputs = [None] * num_micro_batches
        in_flight = {}  # micro_batch_id -> current_stage
        
        # Initialize: send first micro-batches into pipeline
        warmup_steps = min(num_stages, num_micro_batches)
        
        # Warmup phase: fill the pipeline
        for mb_id in range(warmup_steps):
            current = micro_batches[mb_id]
            
            for stage_id in range(min(mb_id + 1, num_stages)):
                stage = self.stages[stage_id]
                current = self._move_to_device(current, stage.device)
                current = self._forward_stage(stage, current, mb_id)
            
            in_flight[mb_id] = min(mb_id + 1, num_stages) - 1
        
        # Steady state: 1F1B
        forward_mb = warmup_steps
        backward_mb = 0
        
        while backward_mb < num_micro_batches:
            # Forward step (if more micro-batches to process)
            if forward_mb < num_micro_batches:
                current = micro_batches[forward_mb]
                
                for stage in self.stages:
                    current = self._move_to_device(current, stage.device)
                    current = self._forward_stage(stage, current, forward_mb)
                
                in_flight[forward_mb] = num_stages - 1
                forward_mb += 1
            
            # Backward step for completed micro-batch
            if backward_mb in in_flight and in_flight[backward_mb] == num_stages - 1:
                # Get output
                output = self._activations[num_stages - 1][backward_mb]
                outputs[backward_mb] = output
                
                # Note: actual backward happens in loss.backward()
                del in_flight[backward_mb]
                backward_mb += 1
        
        return self._merge_outputs(outputs)
    
    def forward(self, x: Tensor, schedule: Optional[PipelineSchedule] = None) -> Tensor:

        schedule = schedule or self.schedule
        
        # Clear activation storage
        self._activations.clear()
        self._inputs.clear()
        
        if schedule == PipelineSchedule.GPIPE:
            return self.forward_gpipe(x)
        elif schedule == PipelineSchedule.ONE_F_ONE_B:
            return self.forward_1f1b(x)
        else:
            return self.forward_gpipe(x)  # Default to GPipe
    
    def parameters(self, recurse: bool = True):

        for stage in self.stages:
            yield from stage.module.parameters(recurse=recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):

        for i, stage in enumerate(self.stages):
            stage_prefix = f"{prefix}stage_{i}." if prefix else f"stage_{i}."
            yield from stage.module.named_parameters(prefix=stage_prefix, recurse=recurse)
    
    def state_dict(self, destination=None, prefix=''):

        state = {}
        for i, stage in enumerate(self.stages):
            stage_state = stage.module.state_dict()
            for key, value in stage_state.items():
                state[f"stage_{i}.{key}"] = value
        return state
    
    def load_state_dict(self, state_dict, strict=True):

        for i, stage in enumerate(self.stages):
            stage_state = {}
            prefix = f"stage_{i}."
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    stage_state[key[len(prefix):]] = value
            stage.module.load_state_dict(stage_state, strict=strict)
    
    def train(self, mode: bool = True):

        self.training = mode
        for stage in self.stages:
            stage.module.train(mode)
        return self
    
    def eval(self):

        return self.train(False)
    
    def get_stage(self, stage_id: int) -> Module:

        return self.stages[stage_id].module
    
    def num_parameters(self, only_trainable: bool = False) -> int:

        total = 0
        for stage in self.stages:
            total += stage.module.num_parameters(only_trainable=only_trainable)
        return total
    
    def __repr__(self):
        lines = [f"PipelineParallel("]
        lines.append(f"  num_stages={self.num_stages},")
        lines.append(f"  chunks={self.chunks},")
        lines.append(f"  schedule={self.schedule.value},")
        lines.append(f"  stages=[")
        for stage in self.stages:
            lines.append(f"    Stage {stage.stage_id} on {stage.device}: {stage.module._get_name()}")
        lines.append(f"  ]")
        lines.append(")")
        return "\n".join(lines)

class DistributedPipelineParallel(Module):

    
    def __init__(
        self,
        module: Module,
        num_stages: int,
        stage_id: Optional[int] = None,
        chunks: int = 1,
        schedule: Union[str, PipelineSchedule] = PipelineSchedule.GPIPE,
    ):
        super().__init__()
        
        if not is_initialized():
            raise RuntimeError("Process group not initialized. Call init_process_group first.")
        
        self._process_group = get_process_group()
        self._rank = get_rank()
        self._world_size = get_world_size()
        
        self.module = module
        self.num_stages = num_stages
        self.stage_id = stage_id if stage_id is not None else self._rank
        self.chunks = chunks
        
        if isinstance(schedule, str):
            schedule = PipelineSchedule(schedule)
        self.schedule = schedule
        
        # Validate setup
        if self.stage_id >= num_stages:
            raise ValueError(f"Stage ID {self.stage_id} >= num_stages {num_stages}")
        
        # Determine neighbors
        self._prev_rank = self.stage_id - 1 if self.stage_id > 0 else None
        self._next_rank = self.stage_id + 1 if self.stage_id < num_stages - 1 else None
        
        # Activation storage
        self._activations: Dict[int, Tensor] = {}
        self._inputs: Dict[int, Tensor] = {}
    
    def _recv_from_prev(self) -> Tensor:

        if self._prev_rank is None:
            raise RuntimeError("No previous stage")
        return self._process_group.recv(self._prev_rank)
    
    def _send_to_next(self, tensor: Tensor):

        if self._next_rank is None:
            raise RuntimeError("No next stage")
        self._process_group.send(tensor, self._next_rank)
    
    def _recv_grad_from_next(self) -> Tensor:

        if self._next_rank is None:
            raise RuntimeError("No next stage")
        return self._process_group.recv(self._next_rank)
    
    def _send_grad_to_prev(self, grad: Tensor):

        if self._prev_rank is None:
            raise RuntimeError("No previous stage")
        self._process_group.send(grad, self._prev_rank)
    
    def forward(self, x: Optional[Tensor] = None) -> Optional[Tensor]:

        self._activations.clear()
        self._inputs.clear()
        
        outputs = []
        
        for mb_id in range(self.chunks):
            # Get input
            if self.stage_id == 0:
                # First stage: use provided input
                if x is None:
                    raise ValueError("First stage requires input tensor")
                batch_size = x.shape[0]
                mb_size = batch_size // self.chunks
                start = mb_id * mb_size
                end = start + mb_size if mb_id < self.chunks - 1 else batch_size
                current = x[start:end]
            else:
                # Receive from previous stage
                current = Tensor(self._recv_from_prev(), requires_grad=True)
            
            # Store input
            self._inputs[mb_id] = current
            
            # Forward through local module
            output = self.module(current)
            
            # Store activation
            self._activations[mb_id] = output
            
            # Send to next stage or collect output
            if self._next_rank is not None:
                self._send_to_next(output)
            else:
                outputs.append(output)
        
        # Last stage returns merged output
        if self._next_rank is None:
            backend = outputs[0]._backend
            data_list = [o.data for o in outputs]
            merged = backend.concatenate(data_list, axis=0)
            return outputs[0]._new_like(merged, requires_grad=True)
        
        return None
    
    def backward_stage(self, grad_output: Optional[Tensor] = None):

        for mb_id in reversed(range(self.chunks)):
            # Get gradient
            if self._next_rank is None:
                # Last stage: use provided gradient or ones
                if grad_output is not None:
                    batch_size = grad_output.shape[0]
                    mb_size = batch_size // self.chunks
                    start = mb_id * mb_size
                    end = start + mb_size if mb_id < self.chunks - 1 else batch_size
                    grad = grad_output[start:end]
                else:
                    # Default to ones
                    activation = self._activations[mb_id]
                    grad = activation._new_like(
                        activation._backend.ones(activation.shape),
                        requires_grad=False
                    )
            else:
                # Receive gradient from next stage
                grad = Tensor(self._recv_grad_from_next(), requires_grad=False)
            
            # Backward through activation
            activation = self._activations[mb_id]
            if activation._requires_grad:
                activation.backward(grad, retain_graph=True)
            
            # Send gradient to previous stage
            if self._prev_rank is not None:
                input_tensor = self._inputs[mb_id]
                if input_tensor.grad is not None:
                    self._send_grad_to_prev(input_tensor.grad)
    
    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse=recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self.module.named_parameters(prefix=prefix, recurse=recurse)
    
    def state_dict(self):
        return self.module.state_dict()
    
    def load_state_dict(self, state_dict, strict=True):
        return self.module.load_state_dict(state_dict, strict=strict)

__all__ = [
    'PipelineSchedule',
    'PipelineStage',
    'MicroBatch',
    'PipelineParallel',
    'DistributedPipelineParallel',
]
