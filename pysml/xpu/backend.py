"""
Intel XPU Backend for PySML — optimized v0.4.8-final
Fast, memory-safe dpnp backend with cached SYCL queues and robust type conversion.
"""

try:
    import dpnp as np
    import dpctl
    AVAILABLE = True
except ImportError:
    AVAILABLE = False
    import numpy as np
    dpctl = None

# Cache default queue for all ops (avoid selecting each call)
if AVAILABLE and dpctl:
    try:
        _DEFAULT_QUEUE = dpctl.SyclQueue()
    except Exception:
        _DEFAULT_QUEUE = None
else:
    _DEFAULT_QUEUE = None

# -----------------------------------------------------------------------------
# Basic creation ops (fp32 default for performance)
# -----------------------------------------------------------------------------
_DEFAULT_DTYPE = np.float32
array = np.array
zeros = lambda shape, dtype=_DEFAULT_DTYPE: np.zeros(shape, dtype=dtype)
ones = lambda shape, dtype=_DEFAULT_DTYPE: np.ones(shape, dtype=dtype)
full = lambda shape, fill_value, dtype=_DEFAULT_DTYPE: np.full(shape, fill_value, dtype=dtype)
eye = np.eye
arange = np.arange
linspace = np.linspace
empty = lambda shape, dtype=_DEFAULT_DTYPE: np.empty(shape, dtype=dtype)

# -----------------------------------------------------------------------------
# Random generation (with fp64 fallbacks)
# -----------------------------------------------------------------------------
class random:
    randn = staticmethod(lambda *shape, **kwargs: np.random.randn(*shape))
    rand = staticmethod(lambda *shape, **kwargs: np.random.rand(*shape))
    randint = staticmethod(np.random.randint)
    uniform = staticmethod(np.random.uniform)
    normal = staticmethod(np.random.normal)

    @staticmethod
    def binomial(n, p, size):
        if not AVAILABLE:
            return np.random.binomial(n, p, size)
        try:
            return np.random.binomial(n, p, size)
        except RuntimeError as e:
            if "fp64" in str(e) or "aspect" in str(e):
                if n == 1:
                    u = np.random.uniform(0.0, 1.0, size)
                    return (u < p).astype(np.float32)
                result = np.zeros(size, dtype=np.float32)
                for _ in range(n):
                    u = np.random.uniform(0.0, 1.0, size)
                    result += (u < p).astype(np.float32)
                return result
            raise

# -----------------------------------------------------------------------------
# Type conversion helpers
# -----------------------------------------------------------------------------
def _is_numpy_array(arr):
    if not AVAILABLE:
        return True
    mod = type(arr).__module__
    return mod == "numpy" or mod.startswith("numpy.")

def _ensure_dpnp_array(arr):
    if not AVAILABLE:
        return arr
    if not _is_numpy_array(arr):
        return arr
    try:
        if _DEFAULT_QUEUE is not None:
            return np.asarray(arr, sycl_queue=_DEFAULT_QUEUE)
        return np.asarray(arr)
    except Exception as e:
        print(f"[PySML:XPU] Warning: conversion to dpnp failed: {e}")
        return arr

# -----------------------------------------------------------------------------
# Arithmetic ops (cached refs, type-safe)
# -----------------------------------------------------------------------------
_add = np.add; _sub = np.subtract; _mul = np.multiply; _div = np.divide; _pow = np.power
def add(a,b): a=_ensure_dpnp_array(a); b=_ensure_dpnp_array(b); return _add(a,b)
def subtract(a,b): a=_ensure_dpnp_array(a); b=_ensure_dpnp_array(b); return _sub(a,b)
def multiply(a,b): a=_ensure_dpnp_array(a); b=_ensure_dpnp_array(b); return _mul(a,b)
def divide(a,b): a=_ensure_dpnp_array(a); b=_ensure_dpnp_array(b); return _div(a,b)
def power(a,b): a=_ensure_dpnp_array(a); b=_ensure_dpnp_array(b); return _pow(a,b)

negative = np.negative
positive = np.positive
abs = absolute = np.abs
sign = np.sign

# In-place variants for autograd
def iadd(a,b): a[...] += b; return a
def isub(a,b): a[...] -= b; return a
def imul(a,b): a[...] *= b; return a
def idiv(a,b): a[...] /= b; return a

# -----------------------------------------------------------------------------
# Linear algebra
# -----------------------------------------------------------------------------
def matmul(a,b): a=_ensure_dpnp_array(a); b=_ensure_dpnp_array(b); return np.matmul(a,b)
def dot(a,b): a=_ensure_dpnp_array(a); b=_ensure_dpnp_array(b); return np.dot(a,b)
def outer(a,b): a=_ensure_dpnp_array(a); b=_ensure_dpnp_array(b); return np.outer(a,b)
def inner(a,b): a=_ensure_dpnp_array(a); b=_ensure_dpnp_array(b); return np.inner(a,b)
einsum = np.einsum
trace = np.trace
diagonal = np.diagonal

# -----------------------------------------------------------------------------
# Shape ops
# -----------------------------------------------------------------------------
def transpose(a,axes=None): a=_ensure_dpnp_array(a); return np.transpose(a,axes=axes)
def reshape(a,shape): a=_ensure_dpnp_array(a); return np.reshape(a,shape)
def swapaxes(a,i,j): a=_ensure_dpnp_array(a); return np.swapaxes(a,i,j)
squeeze=np.squeeze; expand_dims=np.expand_dims
concatenate=np.concatenate; stack=np.stack
flatten=lambda x: x.flatten()
broadcast_to=np.broadcast_to if hasattr(np,"broadcast_to") else None

# -----------------------------------------------------------------------------
# Math & stats
# -----------------------------------------------------------------------------
sin=np.sin; cos=np.cos; tanh=np.tanh; exp=np.exp; log=np.log; sqrt=np.sqrt
sum=np.sum; mean=np.mean; var=np.var; std=np.std; maximum=np.maximum; minimum=np.minimum
clip=np.clip; where=np.where

# -----------------------------------------------------------------------------
# Comparison & logic
# -----------------------------------------------------------------------------
equal=np.equal; not_equal=np.not_equal; less=np.less; greater=np.greater
less_equal=np.less_equal; greater_equal=np.greater_equal
logical_and=np.logical_and; logical_or=np.logical_or; logical_not=np.logical_not

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
zeros_like=np.zeros_like; ones_like=np.ones_like; full_like=np.full_like
asarray=np.asarray; copy=np.copy
asnumpy = getattr(np,"asnumpy",np.asarray)
astype=lambda arr,dtype: arr.astype(dtype) if hasattr(arr,"astype") else np.asarray(arr).astype(dtype)

# -----------------------------------------------------------------------------
# Device management
# -----------------------------------------------------------------------------
def get_device():
    if not AVAILABLE or not dpctl:
        return "cpu"
    try:
        dev=dpctl.select_default_device()
        return f"xpu:{dev.device_id}"
    except Exception:
        return "cpu"

def get_available_devices(backend="level_zero"):
    if not AVAILABLE or not dpctl:
        return []
    try:
        gpus=dpctl.get_devices(device_type="gpu",backend=backend)
        return [f"xpu:{i}" for i,_ in enumerate(gpus)]
    except Exception:
        return []

def synchronize():
    if _DEFAULT_QUEUE is not None:
        try:
            _DEFAULT_QUEUE.wait()
        except Exception:
            pass

# AMP Support ---------------------------------------------------------------
try:
    from . import amp
except Exception:
    amp = None

def autocast(dtype="float16"):
    """Context manager for AMP autocasting."""
    if amp is not None:
        return amp.autocast(dtype=dtype)
    return contextlib.nullcontext()

def autocast_function(fn):
    """Decorator for AMP-enabled functions."""
    if amp is not None:
        return amp.autocast_function(fn)
    return fn


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
BACKEND_NAME="xpu"
DEVICE_TYPE="xpu"
