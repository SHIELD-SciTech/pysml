"""Backend-aware functional helpers used by higher level modules."""
from __future__ import annotations

from typing import Tuple

import pysml
from pysml import Tensor


def _ones_like(reference: Tensor) -> Tensor:
    data = reference._backend.ones_like(reference.data)
    return reference._new_like(data, requires_grad=False)


def attention_free_time_mix(
    avg_key: Tensor,
    avg_value: Tensor,
    key: Tensor,
    value: Tensor,
    decay: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Fuse the RWKV time-mix update into a single helper."""

    one = _ones_like(decay)
    inverse_decay = pysml.subtract(one, decay)
    new_avg_key = pysml.add(
        pysml.multiply(avg_key, decay),
        pysml.multiply(key, inverse_decay),
    )
    new_avg_value = pysml.add(
        pysml.multiply(avg_value, decay),
        pysml.multiply(value, inverse_decay),
    )
    mix = pysml.multiply(new_avg_key, new_avg_value)
    return mix, new_avg_key, new_avg_value


def convolutional_residual_block(
    x: Tensor,
    conv1,
    norm1,
    activation,
    conv2,
    norm2,
    residual_conv=None,
):
    """Helper that computes diffusion-style residual blocks consistently."""

    residual = x if residual_conv is None else residual_conv(x)
    h = activation(norm1(conv1(x)))
    h = norm2(conv2(h))
    return activation(pysml.add(h, residual))


__all__ = ["attention_free_time_mix", "convolutional_residual_block"]
