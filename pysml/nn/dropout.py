

from __future__ import annotations

from .module import Module
from ..tensor import Tensor
from .. import engine

class Dropout(Module):

    
    __constants__ = ['p', 'inplace']
    p: float
    inplace: bool
    
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.inplace = inplace
    
    def forward(self, x: Tensor) -> Tensor:
        return engine.dropout(x, p=self.p, training=self.training)
    
    def extra_repr(self) -> str:
        return f'p={self.p}, inplace={self.inplace}'

class Dropout2d(Module):

    
    __constants__ = ['p', 'inplace']
    p: float
    inplace: bool
    
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.inplace = inplace
    
    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        
        # For 2D dropout, we zero entire channels
        # x shape: (N, C, H, W)
        backend = x._backend
        
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (N, C, H, W), got {x.ndim}D")
        
        N, C, H, W = x.shape
        
        # Create mask for channels only (N, C, 1, 1)
        keep_prob = 1.0 - self.p
        mask_shape = (N, C, 1, 1)
        rand_vals = backend.rand(mask_shape, dtype=x.data.dtype, device=x.device)
        mask = backend.greater(rand_vals, self.p)
        mask = backend.astype(mask, x.data.dtype)
        
        # Scale and apply mask
        scaled_mask = backend.divide(mask, keep_prob)
        result_data = backend.multiply(x.data, scaled_mask)
        
        return x._new_like(result_data, requires_grad=x._requires_grad)
    
    def extra_repr(self) -> str:
        return f'p={self.p}, inplace={self.inplace}'

__all__ = ['Dropout', 'Dropout2d']
