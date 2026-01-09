

from __future__ import annotations

from typing import Optional

from .module import Module
from ..tensor import Tensor
from .. import engine

class ReLU(Module):

    
    __constants__ = ['inplace']
    inplace: bool
    
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    
    def forward(self, x: Tensor) -> Tensor:
        return engine.relu(x)
    
    def extra_repr(self) -> str:
        return 'inplace=True' if self.inplace else ''

class LeakyReLU(Module):

    
    __constants__ = ['negative_slope', 'inplace']
    
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace
    
    def forward(self, x: Tensor) -> Tensor:
        backend = x._backend
        pos = engine.relu(x)
        neg = engine.multiply(engine.minimum(x, Tensor([0.0], dtype=x._dtype, device=x.active_device)), self.negative_slope)
        return engine.add(pos, neg)
    
    def extra_repr(self) -> str:
        return f'negative_slope={self.negative_slope}' + (', inplace=True' if self.inplace else '')

class GELU(Module):

    
    __constants__ = ['approximate']
    approximate: str
    
    def __init__(self, approximate: str = 'none'):
        super().__init__()
        self.approximate = approximate
    
    def forward(self, x: Tensor) -> Tensor:
        return engine.gelu(x)
    
    def extra_repr(self) -> str:
        return f"approximate='{self.approximate}'"

class SiLU(Module):

    
    __constants__ = ['inplace']
    inplace: bool
    
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    
    def forward(self, x: Tensor) -> Tensor:
        return engine.silu(x)
    
    def extra_repr(self) -> str:
        return 'inplace=True' if self.inplace else ''

class Mish(Module):

    
    __constants__ = ['inplace']
    inplace: bool
    
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    
    def forward(self, x: Tensor) -> Tensor:
        backend = x._backend
        # softplus(x) = log(1 + exp(x))
        softplus = engine.log(engine.add(Tensor([1.0], dtype=x._dtype, device=x.active_device), engine.exp(x)))
        return engine.multiply(x, engine.tanh(softplus))
    
    def extra_repr(self) -> str:
        return 'inplace=True' if self.inplace else ''

class Sigmoid(Module):

    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        return engine.sigmoid(x)

class Tanh(Module):

    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        return engine.tanh(x)

class Softmax(Module):

    
    __constants__ = ['dim']
    dim: Optional[int]
    
    def __init__(self, dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        dim = self.dim if self.dim is not None else -1
        return engine.softmax(x, axis=dim)
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}'

class LogSoftmax(Module):

    
    __constants__ = ['dim']
    dim: Optional[int]
    
    def __init__(self, dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        dim = self.dim if self.dim is not None else -1
        return engine.log_softmax(x, axis=dim)
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}'

__all__ = ['ReLU', 'LeakyReLU', 'GELU', 'SiLU', 'Mish', 'Sigmoid', 'Tanh', 'Softmax', 'LogSoftmax']
