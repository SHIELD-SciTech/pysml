

from __future__ import annotations

import math
from typing import Optional

from .module import Module, Parameter
from ..tensor import Tensor
from ..dtype import bf16
from .. import engine

class Linear(Module):

    
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Parameter
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype=None,
        device: str = 'cpu'
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        if dtype is None:
            dtype = bf16()
        
        # Initialize weight with Kaiming uniform
        k = 1.0 / in_features
        bound = math.sqrt(k)
        
        # Create weight tensor
        from ..cpu import backend as cpu_backend
        weight_data = cpu_backend.uniform(-bound, bound, (out_features, in_features), dtype=None, device=None)
        weight_tensor = Tensor(weight_data, dtype=dtype, device=device, requires_grad=True)
        self.weight = Parameter(weight_tensor)
        
        if bias:
            bias_data = cpu_backend.uniform(-bound, bound, (out_features,), dtype=None, device=None)
            bias_tensor = Tensor(bias_data, dtype=dtype, device=device, requires_grad=True)
            self.bias = Parameter(bias_tensor)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: Tensor) -> Tensor:
        # x @ W^T + b
        output = engine.matmul(x, engine.transpose(self.weight))
        if self.bias is not None:
            output = engine.add(output, self.bias)
        return output
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

class Bilinear(Module):

    
    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias: bool = True,
        dtype=None,
        device: str = 'cpu'
    ):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        
        if dtype is None:
            dtype = bf16()
        
        k = 1.0 / in1_features
        bound = math.sqrt(k)
        
        from ..cpu import backend as cpu_backend
        weight_data = cpu_backend.uniform(-bound, bound, (out_features, in1_features, in2_features), dtype=None, device=None)
        weight_tensor = Tensor(weight_data, dtype=dtype, device=device, requires_grad=True)
        self.weight = Parameter(weight_tensor)
        
        if bias:
            bias_data = cpu_backend.zeros((out_features,))
            bias_tensor = Tensor(bias_data, dtype=dtype, device=device, requires_grad=True)
            self.bias = Parameter(bias_tensor)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        # Bilinear: output[k] = x1^T @ weight[k] @ x2
        # For batch processing, this is more complex
        backend = x1._backend
        
        batch_size = x1.shape[0] if x1.ndim > 1 else 1
        
        # Reshape for batch computation
        if x1.ndim == 1:
            x1 = x1.unsqueeze(0)
        if x2.ndim == 1:
            x2 = x2.unsqueeze(0)
        
        # x1: (batch, in1), x2: (batch, in2), weight: (out, in1, in2)
        # output: (batch, out)
        
        # Compute x1 @ weight -> (batch, out, in2)
        x1_expanded = x1.unsqueeze(1)  # (batch, 1, in1)
        # weight: (out, in1, in2) -> we need (batch, out, in1, in2) effectively
        
        # Simpler approach: loop over output features
        outputs = []
        for k in range(self.out_features):
            # weight[k]: (in1, in2)
            # x1 @ weight[k] @ x2^T -> scalar per batch
            w_k = self.weight.data[k]  # (in1, in2)
            temp = engine.matmul(x1, Tensor(w_k, dtype=x1._dtype, device=x1.active_device))  # (batch, in2)
            out_k = engine.sum_with_grad(engine.multiply(temp, x2), axis=1, keepdims=True)  # (batch, 1)
            outputs.append(out_k)
        
        output = engine.concatenate(outputs, axis=1)  # (batch, out)
        
        if self.bias is not None:
            output = engine.add(output, self.bias)
        
        return output
    
    def extra_repr(self) -> str:
        return (f'in1_features={self.in1_features}, in2_features={self.in2_features}, '
                f'out_features={self.out_features}, bias={self.bias is not None}')

__all__ = ['Linear', 'Bilinear']
