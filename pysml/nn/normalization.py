

from __future__ import annotations

from typing import Optional, Union, Tuple, List

from .module import Module, Parameter
from ..tensor import Tensor
from ..dtype import bf16
from .. import engine

class LayerNorm(Module):

    
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool
    
    def __init__(
        self,
        normalized_shape: Union[int, List[int], Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        dtype=None,
        device: str = 'cpu'
    ):
        super().__init__()
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if dtype is None:
            dtype = bf16()
        
        if elementwise_affine:
            from ..cpu import backend as cpu_backend
            weight_data = cpu_backend.ones(self.normalized_shape)
            bias_data = cpu_backend.zeros(self.normalized_shape)
            
            self.weight = Parameter(Tensor(weight_data, dtype=dtype, device=device, requires_grad=True))
            self.bias = Parameter(Tensor(bias_data, dtype=dtype, device=device, requires_grad=True))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: Tensor) -> Tensor:
        return engine.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    
    def extra_repr(self) -> str:
        return f'{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'

class RMSNorm(Module):

    
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    
    def __init__(
        self,
        normalized_shape: Union[int, List[int], Tuple[int, ...]],
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        dtype=None,
        device: str = 'cpu'
    ):
        super().__init__()
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if dtype is None:
            dtype = bf16()
        
        if elementwise_affine:
            from ..cpu import backend as cpu_backend
            weight_data = cpu_backend.ones(self.normalized_shape)
            self.weight = Parameter(Tensor(weight_data, dtype=dtype, device=device, requires_grad=True))
        else:
            self.register_parameter('weight', None)
    
    def forward(self, x: Tensor) -> Tensor:
        return engine.rms_norm(x, self.normalized_shape, self.weight, self.eps)
    
    def extra_repr(self) -> str:
        return f'{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'

class BatchNorm1d(Module):

    
    __constants__ = ['num_features', 'eps', 'momentum', 'affine', 'track_running_stats']
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        dtype=None,
        device: str = 'cpu'
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if dtype is None:
            dtype = bf16()
        
        from ..cpu import backend as cpu_backend
        
        if affine:
            weight_data = cpu_backend.ones((num_features,))
            bias_data = cpu_backend.zeros((num_features,))
            self.weight = Parameter(Tensor(weight_data, dtype=dtype, device=device, requires_grad=True))
            self.bias = Parameter(Tensor(bias_data, dtype=dtype, device=device, requires_grad=True))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        if track_running_stats:
            self.register_buffer('running_mean', Tensor(cpu_backend.zeros((num_features,)), dtype=dtype, device=device))
            self.register_buffer('running_var', Tensor(cpu_backend.ones((num_features,)), dtype=dtype, device=device))
            self.register_buffer('num_batches_tracked', Tensor([0], dtype=dtype, device=device))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)
    
    def forward(self, x: Tensor) -> Tensor:
        running_mean = self.running_mean.data if self.running_mean is not None else None
        running_var = self.running_var.data if self.running_var is not None else None
        
        return engine.batch_norm(
            x, running_mean, running_var,
            self.weight, self.bias,
            self.training, self.momentum, self.eps
        )
    
    def extra_repr(self) -> str:
        return (f'{self.num_features}, eps={self.eps}, momentum={self.momentum}, '
                f'affine={self.affine}, track_running_stats={self.track_running_stats}')

class BatchNorm2d(BatchNorm1d):

    pass

class GroupNorm(Module):

    
    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine']
    
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        dtype=None,
        device: str = 'cpu'
    ):
        super().__init__()
        
        if num_channels % num_groups != 0:
            raise ValueError(f'num_channels ({num_channels}) must be divisible by num_groups ({num_groups})')
        
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        if dtype is None:
            dtype = bf16()
        
        if affine:
            from ..cpu import backend as cpu_backend
            weight_data = cpu_backend.ones((num_channels,))
            bias_data = cpu_backend.zeros((num_channels,))
            self.weight = Parameter(Tensor(weight_data, dtype=dtype, device=device, requires_grad=True))
            self.bias = Parameter(Tensor(bias_data, dtype=dtype, device=device, requires_grad=True))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: Tensor) -> Tensor:
        return engine.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    
    def extra_repr(self) -> str:
        return f'{self.num_groups}, {self.num_channels}, eps={self.eps}, affine={self.affine}'

__all__ = ['LayerNorm', 'RMSNorm', 'BatchNorm1d', 'BatchNorm2d', 'GroupNorm']
