

from __future__ import annotations

import math
from typing import Optional

from .module import Module, Parameter
from ..tensor import Tensor
from ..dtype import bf16
from .. import engine

class Embedding(Module):

    
    __constants__ = ['num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse']
    
    num_embeddings: int
    embedding_dim: int
    padding_idx: Optional[int]
    max_norm: Optional[float]
    norm_type: float
    scale_grad_by_freq: bool
    sparse: bool
    weight: Parameter
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        dtype=None,
        device: str = 'cpu',
        _weight: Optional[Tensor] = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        
        if dtype is None:
            dtype = bf16()
        
        if _weight is None:
            # Initialize with normal distribution
            from ..cpu import backend as cpu_backend
            weight_data = cpu_backend.randn((num_embeddings, embedding_dim), dtype=None, device=None)
            weight_tensor = Tensor(weight_data, dtype=dtype, device=device, requires_grad=True)
            self.weight = Parameter(weight_tensor)
        else:
            self.weight = Parameter(_weight)
        
        if padding_idx is not None:
            # Zero out the padding embedding
            if padding_idx >= 0:
                self.weight.data[padding_idx].fill(0)
    
    def forward(self, indices: Tensor) -> Tensor:

        return engine.embedding(self.weight, indices, padding_idx=self.padding_idx)
    
    def extra_repr(self) -> str:
        s = f'{self.num_embeddings}, {self.embedding_dim}'
        if self.padding_idx is not None:
            s += f', padding_idx={self.padding_idx}'
        if self.max_norm is not None:
            s += f', max_norm={self.max_norm}'
        if self.norm_type != 2.0:
            s += f', norm_type={self.norm_type}'
        if self.scale_grad_by_freq:
            s += f', scale_grad_by_freq={self.scale_grad_by_freq}'
        if self.sparse:
            s += ', sparse=True'
        return s
    
    @classmethod
    def from_pretrained(
        cls,
        embeddings: Tensor,
        freeze: bool = True,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False
    ) -> 'Embedding':

        rows, cols = embeddings.shape
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            _weight=embeddings
        )
        
        if freeze:
            embedding.weight.requires_grad_(False)
        
        return embedding

__all__ = ['Embedding']
