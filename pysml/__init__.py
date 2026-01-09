# PySML - Our custom deep learning framework
# Supports CPU, CUDA, and Intel XPU backends with full autograd

from __future__ import annotations

from .tensor import Tensor
from .dtype import fp32, fp16, bf16
from . import engine
from .engine import (
    matmul, add, subtract, multiply, divide, negative,
    relu, gelu, silu, sigmoid, tanh, softmax, log_softmax,
    layer_norm, rms_norm, batch_norm, group_norm,
    dropout, embedding,
    transpose, reshape, concatenate, stack, split,
    sum as tensor_sum, mean, max as tensor_max, min as tensor_min,
    exp, log, sqrt, abs, power, clip, sin, cos,
)
from . import cpu
from . import cuda
from . import xpu
from . import utils
from . import ddp
from . import nn
from . import optim

__version__ = '0.1.0'

__all__ = [
    'Tensor', 'fp32', 'fp16', 'bf16', 'engine',
    'matmul', 'add', 'subtract', 'multiply', 'divide', 'negative',
    'relu', 'gelu', 'silu', 'sigmoid', 'tanh', 'softmax', 'log_softmax',
    'layer_norm', 'rms_norm', 'batch_norm', 'group_norm',
    'dropout', 'embedding', 'transpose', 'reshape', 'concatenate', 'stack', 'split',
    'tensor_sum', 'mean', 'tensor_max', 'tensor_min',
    'exp', 'log', 'sqrt', 'abs', 'power', 'clip', 'sin', 'cos',
    'xpu', 'cuda', 'cpu', 'ddp', 'utils', 'nn', 'optim',
]
