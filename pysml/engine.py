"""
PySML Engine (v0.4.8-final)
Unified device and backend management for CPU, CUDA, and XPU.
Includes automatic mixed-precision (AMP) and gradient scaling support.
"""

import math
import numpy as np
from contextlib import contextmanager

# =============================================================================
# Global backend management
# =============================================================================
_backend = None
_device = "cpu"
_amp_enabled = False
_amp_dtype = "float32"
_scaler = None

# Cached backends
from importlib import import_module

def _import_backend(name: str):
    try:
        return import_module(f"pysml.{name}.backend")
    except ImportError:
        return None


def set_device(device: str):
    """
    Set current compute device ('cpu', 'cuda', or 'xpu').

    Example:
        >>> set_device("cuda:0")
    """
    global _backend, _device, _amp_enabled, _scaler

    device = device.lower()
    if device.startswith("cuda"):
        _backend = _import_backend("cuda")
        if _backend and getattr(_backend, "AVAILABLE", False):
            _backend.set_device(int(device.split(":")[1]) if ":" in device else 0)
            _device = device
            try:
                from pysml.cuda import amp
                _amp_enabled = True
                _scaler = amp.GradScaler()
            except Exception:
                _amp_enabled = False
        else:
            raise RuntimeError("CUDA backend not available")

    elif device.startswith("xpu"):
        _backend = _import_backend("xpu")
        if _backend and getattr(_backend, "AVAILABLE", False):
            _device = device
            try:
                from pysml.xpu import amp
                _amp_enabled = True
                _scaler = amp.GradScaler()
            except Exception:
                _amp_enabled = False
        else:
            raise RuntimeError("Intel XPU backend not available")

    else:
        # CPU fallback
        _backend = np
        _device = "cpu"
        _amp_enabled = False
        _scaler = None

def get_device():
    """Return current device string."""
    return _device

def get_backend():
    """Return current backend module."""
    return _backend

def get_available_devices():
    """Return list of devices across all supported backends."""
    devices = ["cpu"]
    for name in ("cuda", "xpu"):
        b = _import_backend(name)
        if b and getattr(b, "AVAILABLE", False):
            get_list = getattr(b, "get_available_devices", lambda: [])
            devices.extend(get_list())
    return devices


def to_device(arr, device=None):
    """Convert NumPy array to device array if needed."""
    global _backend
    if device is None:
        device = _device
    if device == "cpu" or _backend is None:
        return np.asarray(arr)
    if hasattr(_backend, "asarray"):
        return _backend.asarray(arr)
    return arr


# =============================================================================
# AMP Management
# =============================================================================
def amp_enabled():
    """Check if AMP is enabled for current backend."""
    return _amp_enabled

def get_amp_dtype():
    return _amp_dtype

@contextmanager
def autocast(dtype="float16"):
    """
    Enable autocast for mixed precision within a context.
    Example:
        >>> with autocast("float16"):
        ...     y = matmul(x, w)
    """
    global _amp_dtype
    if not _amp_enabled or not hasattr(_backend, "autocast"):
        yield
    else:
        with _backend.autocast(dtype=dtype):
            old_dtype = _amp_dtype
            _amp_dtype = dtype
            try:
                yield
            finally:
                _amp_dtype = old_dtype


def get_scaler():
    """Return the active GradScaler if AMP is enabled."""
    return _scaler


# =============================================================================
# Tensor creation
# =============================================================================
def zeros(*shape, **kwargs): return _backend.zeros(shape, **kwargs)
def ones(*shape, **kwargs): return _backend.ones(shape, **kwargs)
def full(*shape, **kwargs): return _backend.full(*shape, **kwargs)
def eye(*args, **kwargs): return _backend.eye(*args, **kwargs)
def arange(*args, **kwargs): return _backend.arange(*args, **kwargs)
def linspace(*args, **kwargs): return _backend.linspace(*args, **kwargs)
def randn(*args, **kwargs): return _backend.random.randn(*args, **kwargs)
def rand(*args, **kwargs): return _backend.random.rand(*args, **kwargs)
def randint(*args, **kwargs): return _backend.random.randint(*args, **kwargs)

# =============================================================================
# Arithmetic and Linear Algebra
# =============================================================================
def add(x, y): return _backend.add(x, y)
def subtract(x, y): return _backend.subtract(x, y)
def multiply(x, y): return _backend.multiply(x, y)
def divide(x, y): return _backend.divide(x, y)
def power(x, y): return _backend.power(x, y)
def negative(x): return _backend.negative(x)
def positive(x): return _backend.positive(x)

def matmul(x, y):
    if _amp_enabled and hasattr(_backend, "amp"):
        fn = getattr(_backend.amp, "autocast_function", None)
        if fn: return fn(_backend.matmul)(x, y)
    return _backend.matmul(x, y)

def dot(x, y): return _backend.dot(x, y)
def outer(x, y): return _backend.outer(x, y)
def inner(x, y): return _backend.inner(x, y)
def einsum(expr, *args): return _backend.einsum(expr, *args)


# =============================================================================
# Reductions
# =============================================================================
def sum(x, **kwargs): return _backend.sum(x, **kwargs)
def mean(x, **kwargs): return _backend.mean(x, **kwargs)
def max(x, **kwargs): return _backend.max(x, **kwargs)
def min(x, **kwargs): return _backend.min(x, **kwargs)
def var(x, **kwargs): return _backend.var(x, **kwargs)
def std(x, **kwargs): return _backend.std(x, **kwargs)

# =============================================================================
# Activations / math
# =============================================================================
def relu(x): return _backend.maximum(0, x)
def sigmoid(x): return 1 / (1 + _backend.exp(-x))
def tanh(x): return _backend.tanh(x)
def softmax(x, axis=-1):
    ex = _backend.exp(x - _backend.max(x, axis=axis, keepdims=True))
    return ex / _backend.sum(ex, axis=axis, keepdims=True)
def gelu(x):
    return 0.5 * x * (1 + _backend.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * _backend.power(x, 3))))
def exp(x): return _backend.exp(x)
def log(x): return _backend.log(x)
def sqrt(x): return _backend.sqrt(x)
def abs(x): return _backend.abs(x)
def sin(x): return _backend.sin(x)
def cos(x): return _backend.cos(x)

# =============================================================================
# Shape operations
# =============================================================================
def reshape(x, shape): return _backend.reshape(x, shape)
def transpose(x, axes=None): return _backend.transpose(x, axes)
def squeeze(x, axis=None): return _backend.squeeze(x, axis)
def unsqueeze(x, axis=0): return _backend.expand_dims(x, axis)
def concatenate(xs, axis=0): return _backend.concatenate(xs, axis)
def stack(xs, axis=0): return _backend.stack(xs, axis)
def flatten(x): return _backend.flatten(x)

# =============================================================================
# Comparison
# =============================================================================
def equal(a, b): return _backend.equal(a, b)
def not_equal(a, b): return _backend.not_equal(a, b)
def less(a, b): return _backend.less(a, b)
def less_equal(a, b): return _backend.less_equal(a, b)
def greater(a, b): return _backend.greater(a, b)
def greater_equal(a, b): return _backend.greater_equal(a, b)

# =============================================================================
# Advanced operations
# =============================================================================
def clip(x, a, b): return _backend.clip(x, a, b)
def where(cond, a, b): return _backend.where(cond, a, b)
def maximum(a, b): return _backend.maximum(a, b)
def minimum(a, b): return _backend.minimum(a, b)

# =============================================================================
# Loss functions
# =============================================================================
def mse_loss(pred, target):
    diff = subtract(pred, target)
    return mean(multiply(diff, diff))

def cross_entropy_loss(pred, target, eps=1e-9):
    pred = clip(pred, eps, 1 - eps)
    return -mean(sum(multiply(target, log(pred)), axis=1))

def binary_cross_entropy_loss(pred, target, eps=1e-9):
    pred = clip(pred, eps, 1 - eps)
    return -mean(target * log(pred) + (1 - target) * log(1 - pred))

# =============================================================================
# Normalization
# =============================================================================
def batch_norm(x, mean, var, gamma, beta, eps=1e-5):
    return gamma * (x - mean) / sqrt(var + eps) + beta

def layer_norm(x, eps=1e-5):
    m = mean(x, axis=-1, keepdims=True)
    v = var(x, axis=-1, keepdims=True)
    return (x - m) / sqrt(v + eps)

def dropout(x, p=0.5):
    if p <= 0.0: return x
    mask = (_backend.random.rand(*x.shape) > p).astype(_backend.float32)
    return x * mask / (1.0 - p)

# =============================================================================
# AMP-aware Training Loop Utilities
# =============================================================================
def scale_loss(loss):
    """Scale loss for AMP if enabled."""
    if _amp_enabled and _scaler is not None:
        return _scaler.scale(loss)
    return loss

def optimizer_step(optimizer):
    """Perform optimizer step with AMP gradient scaling."""
    if _amp_enabled and _scaler is not None:
        _scaler.step(optimizer)
        _scaler.update()
    else:
        optimizer.step()

def backward(loss):
    """Backward pass, AMP-aware (placeholder for autograd integration)."""
    if hasattr(loss, "backward"):
        loss.backward()

# =============================================================================
# Initialization
# =============================================================================
set_device("cpu")  # default

# =============================================================================
# Public exports
# =============================================================================
__all__ = [
    # Device
    "set_device", "get_device", "get_backend", "get_available_devices", "to_device",
    # AMP
    "amp_enabled", "autocast", "scale_loss", "optimizer_step", "get_scaler",
    # Creation
    "zeros", "ones", "full", "eye", "arange", "linspace", "randn", "rand", "randint",
    # Arithmetic
    "add", "subtract", "multiply", "divide", "power", "negative", "positive",
    # Linear algebra
    "matmul", "dot", "outer", "inner", "einsum",
    # Reductions
    "sum", "mean", "max", "min", "var", "std",
    # Activations / math
    "relu", "sigmoid", "tanh", "softmax", "gelu", "exp", "log", "sqrt", "abs", "sin", "cos",
    # Shape
    "reshape", "transpose", "squeeze", "unsqueeze", "concatenate", "stack", "flatten",
    # Comparison
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    # Advanced
    "clip", "where", "maximum", "minimum",
    # Loss
    "mse_loss", "cross_entropy_loss", "binary_cross_entropy_loss",
    # Normalization
    "batch_norm", "layer_norm", "dropout",
]
