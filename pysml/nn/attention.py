from __future__ import annotations

import math
from typing import Optional

from .dropout import Dropout
from .linear import Linear
from .module import Module, Parameter


class MultiHeadAttention(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_p = dropout

        self.kdim = kdim if kdim is not None else d_model
        self.vdim = vdim if vdim is not None else d_model

        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = Linear(d_model, d_model, bias=bias)
        self.k_proj = Linear(self.kdim, d_model, bias=bias)
        self.v_proj = Linear(self.vdim, d_model, bias=bias)
        self.out_proj = Linear(d_model, d_model, bias=bias)

        self.dropout = Dropout(dropout) if dropout > 0 else None

        self.add_bias_kv = add_bias_kv
        if add_bias_kv:
            self.bias_k = Parameter(self._initialize_bias())
            self.bias_v = Parameter(self._initialize_bias())

        self.add_zero_attn = add_zero_attn

    def _initialize_bias(self):
        from .. import Tensor
        import numpy as np

        data = np.zeros((1, 1, self.d_model), dtype=np.float32)
        return Tensor(data, requires_grad=True)

    def forward(
        self,
        query,
        key=None,
        value=None,
        attn_mask=None,
        key_padding_mask=None,
        need_weights: bool = False,
    ):
        from .. import engine

        if key is None:
            key = query
        if value is None:
            value = key

        batch_size, target_len, _ = query.shape
        src_len = key.shape[1]

        if self.add_bias_kv:
            key, value = self._append_bias(key, value)
            src_len = key.shape[1]
            if key_padding_mask is not None:
                key_padding_mask = self._pad_mask_with_false_column(key_padding_mask)

        if self.add_zero_attn:
            key, value = self._append_zero_attn(key, value)
            src_len = key.shape[1]
            if key_padding_mask is not None:
                key_padding_mask = self._pad_mask_with_false_column(key_padding_mask)

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = self._split_heads(q, batch_size, target_len)
        k = self._split_heads(k, batch_size, src_len)
        v = self._split_heads(v, batch_size, src_len)

        k_t = engine.transpose(k, (0, 1, 3, 2))
        scores = engine.matmul(q, k_t)
        scores = engine.multiply(scores, self.scale)

        if attn_mask is not None:
            scores = self._apply_attention_mask(scores, attn_mask)

        if key_padding_mask is not None:
            scores = self._apply_key_padding_mask(scores, key_padding_mask)

        attn_weights = engine.softmax(scores, axis=-1)

        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        attn_output = engine.matmul(attn_weights, v)
        attn_output = self._combine_heads(attn_output, batch_size, target_len)
        output = self.out_proj(attn_output)

        if need_weights:
            # Match PyTorch's behaviour: average weights across heads.
            weights = engine.mean(attn_weights, axis=1)
            return output, weights
        return output

    def _split_heads(self, tensor, batch_size: int, seq_len: int):
        from .. import engine

        reshaped = tensor.reshape((batch_size, seq_len, self.num_heads, self.head_dim))
        return engine.permute(reshaped, (0, 2, 1, 3))

    def _combine_heads(self, tensor, batch_size: int, seq_len: int):
        from .. import engine

        transposed = engine.permute(tensor, (0, 2, 1, 3))
        return transposed.reshape((batch_size, seq_len, self.d_model))

    def _append_bias(self, key, value):
        from .. import engine

        batch_size = key.shape[0]
        bias_k = self._expand_bias(self.bias_k.data, batch_size)
        bias_v = self._expand_bias(self.bias_v.data, batch_size)
        key = engine.concatenate([key, bias_k], axis=1)
        value = engine.concatenate([value, bias_v], axis=1)
        return key, value

    def _append_zero_attn(self, key, value):
        backend = key._backend
        zero_k = backend.zeros((key.shape[0], 1, key.shape[2]), dtype=key.data.dtype)
        zero_v = backend.zeros((value.shape[0], 1, value.shape[2]), dtype=value.data.dtype)
        zero_k_tensor = self._wrap_constant_like(key, zero_k, requires_grad=False)
        zero_v_tensor = self._wrap_constant_like(value, zero_v, requires_grad=False)

        from .. import engine

        key = engine.concatenate([key, zero_k_tensor], axis=1)
        value = engine.concatenate([value, zero_v_tensor], axis=1)
        return key, value

    def _expand_bias(self, bias, batch_size: int):
        backend = bias._backend
        tiled = backend.tile(bias.data, (batch_size, 1, 1))
        return self._wrap_constant_like(bias, tiled, requires_grad=bias._requires_grad)

    def _pad_mask_with_false_column(self, mask):
        backend = mask._backend
        pad = backend.zeros((mask.shape[0], 1), dtype=backend.bool)
        padded = backend.concatenate([mask.data, pad], axis=1)
        return self._wrap_constant_like(mask, padded, requires_grad=False)

    def _apply_attention_mask(self, scores, mask):
        from .. import engine

        mask_tensor = self._ensure_tensor(mask, reference=scores)
        backend = scores._backend

        mask_data = mask_tensor.data
        if mask_data.ndim == 2:
            mask_data = backend.expand_dims(backend.expand_dims(mask_data, 0), 0)
        elif mask_data.ndim == 3:
            mask_data = backend.expand_dims(mask_data, 1)
        elif mask_data.ndim != 4:
            raise ValueError("attn_mask must be 2D, 3D, or 4D")

        broadcast = backend.broadcast_to(mask_data, scores.shape)
        if mask_data.dtype == backend.bool:
            condition = self._wrap_constant_like(scores, broadcast, requires_grad=False)
            large_neg = -1e9
            return engine.where(condition, large_neg, scores)

        additive = self._wrap_constant_like(scores, broadcast, requires_grad=False)
        return engine.add(scores, additive)

    def _apply_key_padding_mask(self, scores, key_padding_mask):
        from .. import engine

        mask_tensor = self._ensure_tensor(key_padding_mask, reference=scores)
        backend = scores._backend

        if mask_tensor.data.ndim != 2:
            raise ValueError("key_padding_mask must be 2D (batch, src_len)")

        expanded = backend.expand_dims(backend.expand_dims(mask_tensor.data, 1), 1)
        broadcast = backend.broadcast_to(expanded, scores.shape)
        condition = self._wrap_constant_like(scores, broadcast, requires_grad=False)
        large_neg = -1e9
        return engine.where(condition, large_neg, scores)

    def _ensure_tensor(self, maybe_tensor, reference):
        if hasattr(maybe_tensor, 'data'):
            return maybe_tensor

        from .. import Tensor

        tensor = Tensor(maybe_tensor, requires_grad=False)
        device = reference.active_device if reference.active_device is not None else 'cpu'
        tensor.to(device)
        return tensor

    def _wrap_constant_like(self, template, data, requires_grad: Optional[bool] = None):
        from ..tensor import Tensor

        tensor = Tensor.__new__(Tensor)
        tensor._backend = template._backend
        tensor._dtype = template._dtype
        tensor.device = template.device
        tensor.active_device = template.active_device
        tensor.data = data
        tensor._requires_grad = template._requires_grad if requires_grad is None else requires_grad
        tensor._grad = None
        return tensor

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, num_heads={self.num_heads}, dropout={self.dropout_p}, "
            f"kdim={self.kdim}, vdim={self.vdim}, add_bias_kv={self.add_bias_kv}, "
            f"add_zero_attn={self.add_zero_attn}"
        )


class MultiHeadSelfAttention(Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout=dropout, bias=bias)

    def forward(self, x, attn_mask=None, key_padding_mask=None, need_weights: bool = False):
        return self.attention(x, x, x, attn_mask, key_padding_mask, need_weights)

    def extra_repr(self) -> str:
        return self.attention.extra_repr()


class CrossAttention(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            bias=bias,
            kdim=kdim,
            vdim=vdim,
        )

    def forward(
        self,
        query,
        key,
        value,
        attn_mask=None,
        key_padding_mask=None,
        need_weights: bool = False,
    ):
        return self.attention(query, key, value, attn_mask, key_padding_mask, need_weights)

    def extra_repr(self) -> str:
        return self.attention.extra_repr()


def create_causal_mask(seq_len: int, device: str = 'cpu'):
    import numpy as np
    from .. import Tensor

    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    tensor = Tensor(mask, requires_grad=False)
    tensor.to(device)
    return tensor


def create_padding_mask(lengths, max_len: Optional[int] = None, device: str = 'cpu'):
    import numpy as np
    from .. import Tensor

    batch_size = len(lengths)
    if max_len is None:
        max_len = int(np.max(lengths))

    mask = np.zeros((batch_size, max_len), dtype=bool)
    for i, length in enumerate(lengths):
        length = int(length)
        if length < max_len:
            mask[i, length:] = True

    tensor = Tensor(mask, requires_grad=False)
    tensor.to(device)
    return tensor


__all__ = [
    'MultiHeadAttention',
    'MultiHeadSelfAttention',
    'CrossAttention',
    'create_causal_mask',
    'create_padding_mask',
]
