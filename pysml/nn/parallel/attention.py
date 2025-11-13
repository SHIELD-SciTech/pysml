"""Tensor parallel aware attention primitives."""

from __future__ import annotations

import math
from typing import Optional

from ..module import Module
from ..dropout import Dropout
from ... import engine
from ...tensor import Tensor
from ...distributed import tensor_parallel as tp
from .linear import ColumnParallelLinear, RowParallelLinear


class TensorParallelMultiheadAttention(Module):
    """Self-attention where heads are partitioned across tensor-parallel ranks."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        bias: bool = True,
        group: Optional[tp.TensorParallelGroup] = None,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.group = group or tp.get_tensor_parallel_group()
        tp_size = self.group.size
        if num_heads % tp_size != 0:
            raise ValueError("num_heads must be divisible by tensor parallel size")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.local_heads = num_heads // max(1, tp_size)
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv_proj = ColumnParallelLinear(hidden_size, 3 * hidden_size, bias=bias, gather_output=False, group=self.group)
        self.out_proj = RowParallelLinear(hidden_size, hidden_size, bias=bias, group=self.group, input_is_parallel=True)
        self.dropout = Dropout(dropout) if dropout > 0 else None

    def forward(self, hidden: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        batch, seqlen, _ = hidden.shape
        qkv = self.qkv_proj(hidden)
        q, k, v = engine.split(qkv, 3, dim=-1)
        q = self._reshape_to_heads(q, batch, seqlen)
        k = self._reshape_to_heads(k, batch, seqlen)
        v = self._reshape_to_heads(v, batch, seqlen)

        k_t = engine.transpose(k, (0, 1, 3, 2))
        scores = engine.matmul(q, k_t)
        scores = engine.multiply(scores, self.scale)
        if attn_mask is not None:
            scores = self._apply_mask(scores, attn_mask)
        attn = engine.softmax(scores, axis=-1)
        if self.dropout is not None:
            attn = self.dropout(attn)
        context = engine.matmul(attn, v)
        context = self._merge_heads(context, batch, seqlen)
        output = self.out_proj(context)
        return output

    def _reshape_to_heads(self, tensor: Tensor, batch: int, seqlen: int) -> Tensor:
        reshaped = tensor.reshape((batch, seqlen, self.local_heads, self.head_dim))
        return engine.permute(reshaped, (0, 2, 1, 3))

    def _merge_heads(self, tensor: Tensor, batch: int, seqlen: int) -> Tensor:
        merged = engine.permute(tensor, (0, 2, 1, 3))
        return merged.reshape((batch, seqlen, self.local_heads * self.head_dim))

    def _apply_mask(self, scores: Tensor, mask: Tensor) -> Tensor:
        backend = scores._backend
        mask_tensor = mask
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.reshape((1, 1, mask_tensor.shape[0], mask_tensor.shape[1]))
        elif mask_tensor.ndim == 3:
            mask_tensor = mask_tensor.reshape((mask_tensor.shape[0], 1, mask_tensor.shape[1], mask_tensor.shape[2]))
        broadcast = backend.broadcast_to(mask_tensor.data, scores.shape)
        wrapped = scores._new_like(broadcast, requires_grad=False)
        return engine.add(scores, wrapped)
