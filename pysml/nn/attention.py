

from __future__ import annotations

import math
from typing import Optional, Tuple

from .module import Module, Parameter
from .linear import Linear
from ..tensor import Tensor
from ..dtype import bf16
from .. import engine

class ScaledDotProductAttention(Module):

    
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False
    ) -> Tuple[Tensor, Tensor]:

        backend = query._backend
        
        # Get dimensions
        d_k = query.shape[-1]
        scale = 1.0 / math.sqrt(d_k)
        
        # Compute attention scores: QK^T / sqrt(d_k)
        # query: (..., seq_len_q, d_k), key: (..., seq_len_k, d_k)
        # scores: (..., seq_len_q, seq_len_k)
        key_t = engine.transpose(key, axes=list(range(key.ndim - 2)) + [key.ndim - 1, key.ndim - 2])
        scores = engine.matmul(query, key_t)
        scores = engine.multiply(scores, scale)
        
        # Apply causal mask if needed
        if is_causal:
            seq_len_q = query.shape[-2]
            seq_len_k = key.shape[-2]
            causal_mask = backend.triu(backend.ones((seq_len_q, seq_len_k)), k=1)
            causal_mask = causal_mask * float('-inf')
            scores_data = backend.add(scores.data, causal_mask)
            scores = query._new_like(scores_data, requires_grad=scores._requires_grad)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            mask_data = attn_mask.data if isinstance(attn_mask, Tensor) else attn_mask
            # Mask should be 0 where attention is allowed, -inf where masked
            scores_data = backend.add(scores.data, mask_data)
            scores = query._new_like(scores_data, requires_grad=scores._requires_grad)
        
        # Softmax over last dimension (key dimension)
        attn_weights = engine.softmax(scores, axis=-1)
        
        # Apply dropout
        if self.training and self.dropout > 0:
            attn_weights = engine.dropout(attn_weights, p=self.dropout, training=True)
        
        # Apply attention to values
        # attn_weights: (..., seq_len_q, seq_len_k), value: (..., seq_len_k, d_v)
        # output: (..., seq_len_q, d_v)
        output = engine.matmul(attn_weights, value)
        
        return output, attn_weights

class MultiheadAttention(Module):

    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = True,
        dtype=None,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        if dtype is None:
            dtype = bf16()
        
        # Projection layers
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias, dtype=dtype, device=device)
        self.k_proj = Linear(self.kdim, embed_dim, bias=bias, dtype=dtype, device=device)
        self.v_proj = Linear(self.vdim, embed_dim, bias=bias, dtype=dtype, device=device)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias, dtype=dtype, device=device)
        
        # Attention
        self.attention = ScaledDotProductAttention(dropout=dropout)
        
        # Optional bias for key/value
        if add_bias_kv:
            from ..cpu import backend as cpu_backend
            self.bias_k = Parameter(Tensor(
                cpu_backend.zeros((1, 1, embed_dim)),
                dtype=dtype, device=device, requires_grad=True
            ))
            self.bias_v = Parameter(Tensor(
                cpu_backend.zeros((1, 1, embed_dim)),
                dtype=dtype, device=device, requires_grad=True
            ))
        else:
            self.register_parameter('bias_k', None)
            self.register_parameter('bias_v', None)
    
    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        need_weights: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:

        if key is None:
            key = query
        if value is None:
            value = query
        
        # Get batch size and sequence lengths
        if self.batch_first:
            batch_size, seq_len_q, _ = query.shape
            seq_len_k = key.shape[1]
        else:
            seq_len_q, batch_size, _ = query.shape
            seq_len_k = key.shape[0]
            # Transpose to batch_first for easier processing
            query = query.permute(1, 0, 2)
            key = key.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        
        # Project Q, K, V
        q = self.q_proj(query)  # (batch, seq_len_q, embed_dim)
        k = self.k_proj(key)    # (batch, seq_len_k, embed_dim)
        v = self.v_proj(value)  # (batch, seq_len_k, embed_dim)
        
        # Add bias_k and bias_v if present
        if self.bias_k is not None:
            # Expand bias to batch size
            bias_k_expanded = self.bias_k.expand(batch_size, 1, self.embed_dim)
            bias_v_expanded = self.bias_v.expand(batch_size, 1, self.embed_dim)
            k = engine.concatenate([k, bias_k_expanded], axis=1)
            v = engine.concatenate([v, bias_v_expanded], axis=1)
            seq_len_k += 1
        
        # Reshape for multi-head: (batch, seq, embed) -> (batch, heads, seq, head_dim)
        q = q.reshape(batch_size, seq_len_q, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # (batch, heads, seq_q, head_dim)
        
        k = k.reshape(batch_size, seq_len_k, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)  # (batch, heads, seq_k, head_dim)
        
        v = v.reshape(batch_size, seq_len_k, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)  # (batch, heads, seq_k, head_dim)
        
        # Prepare attention mask
        combined_mask = None
        backend = query._backend
        
        if attn_mask is not None:
            mask_data = attn_mask.data if isinstance(attn_mask, Tensor) else attn_mask
            # Convert boolean mask to float mask
            if mask_data.dtype == bool:
                mask_data = backend.where(mask_data, float('-inf'), 0.0)
            combined_mask = mask_data
        
        if key_padding_mask is not None:
            pad_mask_data = key_padding_mask.data if isinstance(key_padding_mask, Tensor) else key_padding_mask
            # Convert to attention mask format: (batch, seq_k) -> (batch, 1, 1, seq_k)
            if pad_mask_data.dtype == bool:
                pad_mask_data = backend.where(pad_mask_data, float('-inf'), 0.0)
            pad_mask_data = backend.reshape(pad_mask_data, (batch_size, 1, 1, seq_len_k))
            
            if combined_mask is None:
                combined_mask = pad_mask_data
            else:
                combined_mask = backend.add(combined_mask, pad_mask_data)
        
        combined_mask_tensor = None
        if combined_mask is not None:
            combined_mask_tensor = query._new_like(combined_mask, requires_grad=False)
        
        # Apply attention
        attn_output, attn_weights = self.attention(q, k, v, combined_mask_tensor, is_causal)
        
        # Reshape back: (batch, heads, seq_q, head_dim) -> (batch, seq_q, embed)
        attn_output = attn_output.permute(0, 2, 1, 3)  # (batch, seq_q, heads, head_dim)
        attn_output = attn_output.reshape(batch_size, seq_len_q, self.embed_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        # Transpose back if not batch_first
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        
        if need_weights:
            # Average attention weights across heads
            attn_weights_avg = attn_weights.mean(axis=1)  # (batch, seq_q, seq_k)
            return output, attn_weights_avg
        else:
            return output, None
    
    def extra_repr(self) -> str:
        return f'embed_dim={self.embed_dim}, num_heads={self.num_heads}, dropout={self.dropout}'

__all__ = ['ScaledDotProductAttention', 'MultiheadAttention']
