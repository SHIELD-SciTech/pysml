"""
CUDA Backend for PySML — optimized v0.4.8-final
Efficient CuPy backend with in-place math, cached stream, and safe NumPy fallback.
"""

try:
    import cupy as cp
    AVAILABLE = True
except ImportError:
    AVAILABLE = False
    import numpy as cp

# -----------------------------------------------------------------------------
# Core settings and defaults
# -----------------------------------------------------------------------------
_DEFAULT_DTYPE = cp.float32

# Cached stream for reduced launch overhead
if AVAILABLE:
    try:
        _DEFAULT_STREAM = cp.cuda.Stream(non_blocking=True)
    except Exception:
        _DEFAULT_STREAM = None
else:
    _DEFAULT_STREAM = None

# -----------------------------------------------------------------------------
# Basic tensor creation (fp32 default)
# -----------------------------------------------------------------------------
array = cp.array
zeros = lambda shape, dtype=_DEFAULT_DTYPE: cp.zeros(shape, dtype=dtype)
ones = lambda shape, dtype=_DEFAULT_DTYPE: cp.ones(shape, dtype=dtype)
full = lambda shape, val, dtype=_DEFAULT_DTYPE: cp.full(shape, val, dtype=dtype)
eye = cp.eye
arange = cp.arange
linspace = cp.linspace
empty = lambda shape, dtype=_DEFAULT_DTYPE: cp.empty(shape, dtype=dtype)

# -----------------------------------------------------------------------------
# Random generation
# -----------------------------------------------------------------------------
class random:
    randn = staticmethod(cp.random.randn)
    rand = staticmethod(cp.random.rand)
    randint = staticmethod(cp.random.randint)
    uniform = staticmethod(cp.random.uniform)
    normal = staticmethod(cp.random.normal)
    binomial = staticmethod(cp.random.binomial)

# -----------------------------------------------------------------------------
# Arithmetic (cached functions + in-place ops)
# -----------------------------------------------------------------------------
_add, _sub, _mul, _div, _pow = cp.add, cp.subtract, cp.multiply, cp.divide, cp.power
def add(a,b): return _add(a,b)
def subtract(a,b): return _sub(a,b)
def multiply(a,b): return _mul(a,b)
def divide(a,b): return _div(a,b)
def power(a,b): return _pow(a,b)

negative = cp.negative
positive = cp.positive
abs = absolute = cp.abs
sign = cp.sign

# In-place math ops for autograd efficiency
def iadd(a,b): a[...] += b; return a
def isub(a,b): a[...] -= b; return a
def imul(a,b): a[...] *= b; return a
def idiv(a,b): a[...] /= b; return a

# -----------------------------------------------------------------------------
# Linear algebra
# -----------------------------------------------------------------------------
matmul = cp.matmul
dot = cp.dot
outer = cp.outer
inner = cp.inner
trace = cp.trace
diagonal = cp.diagonal
einsum = cp.einsum

# -----------------------------------------------------------------------------
# Trigonometric / exponential / log
# -----------------------------------------------------------------------------
sin, cos, tanh, exp, log, sqrt = cp.sin, cp.cos, cp.tanh, cp.exp, cp.log, cp.sqrt

# -----------------------------------------------------------------------------
# Shape manipulation
# -----------------------------------------------------------------------------
reshape = cp.reshape
transpose = cp.transpose
swapaxes = cp.swapaxes
squeeze = cp.squeeze
expand_dims = cp.expand_dims
concatenate = cp.concatenate
stack = cp.stack
flatten = lambda x: x.flatten()
broadcast_to = cp.broadcast_to if hasattr(cp, "broadcast_to") else None

# -----------------------------------------------------------------------------
# Reductions & statistics
# -----------------------------------------------------------------------------
sum = cp.sum
mean = cp.mean
var = cp.var
std = cp.std
maximum = cp.maximum
minimum = cp.minimum
clip = cp.clip
where = cp.where

# -----------------------------------------------------------------------------
# Logic & comparison
# -----------------------------------------------------------------------------
equal = cp.equal
not_equal = cp.not_equal
less = cp.less
greater = cp.greater
less_equal = cp.less_equal
greater_equal = cp.greater_equal
logical_and = cp.logical_and
logical_or = cp.logical_or
logical_not = cp.logical_not

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
zeros_like = cp.zeros_like
ones_like = cp.ones_like
full_like = cp.full_like
asarray = cp.asarray
copy = cp.copy
asnumpy = getattr(cp, "asnumpy", lambda x: cp.asarray(x))

astype = lambda arr, dtype: arr.astype(dtype) if hasattr(arr, "astype") else cp.asarray(arr).astype(dtype)

# -----------------------------------------------------------------------------
# Device management
# -----------------------------------------------------------------------------
def get_device():
    """Return current CUDA device string."""
    if AVAILABLE:
        try:
            return f"cuda:{cp.cuda.Device().id}"
        except Exception:
            return "cpu"
    return "cpu"

def get_available_devices():
    """Return list of all CUDA devices."""
    if not AVAILABLE:
        return []
    try:
        n = cp.cuda.runtime.getDeviceCount()
        return [f"cuda:{i}" for i in range(n)]
    except Exception:
        return []

def set_device(device_id: int):
    """Set active CUDA device."""
    if AVAILABLE:
        cp.cuda.Device(device_id).use()

def synchronize():
    """Synchronize current stream/device."""
    if AVAILABLE:
        try:
            if _DEFAULT_STREAM is not None:
                _DEFAULT_STREAM.synchronize()
            else:
                cp.cuda.Stream.null.synchronize()
        except Exception:
            pass

# -----------------------------------------------------------------------------
# Memory management helpers
# -----------------------------------------------------------------------------
def get_memory_info(device_id=0):
    """Return (free,total) memory in bytes."""
    if not AVAILABLE:
        return {}
    try:
        with cp.cuda.Device(device_id):
            free, total = cp.cuda.runtime.memGetInfo()
            used = total - free
            return {
                "total": total,
                "free": free,
                "used": used,
                "used_percent": (used / total * 100) if total > 0 else 0,
                "device_id": device_id,
            }
    except Exception as e:
        return {"error": str(e)}

def empty_cache():
    """Free all cached memory blocks."""
    if not AVAILABLE:
        return
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

def synchronize_all():
    """Synchronize all devices sequentially."""
    if not AVAILABLE:
        return
    try:
        for i in range(cp.cuda.runtime.getDeviceCount()):
            with cp.cuda.Device(i):
                cp.cuda.Stream.null.synchronize()
    except Exception:
        pass

# AMP Support ---------------------------------------------------------------
try:
    from . import amp
except Exception:
    amp = None

def autocast(dtype="float16"):
    """Context manager for AMP autocasting"""
    if amp is not None:
        return amp.autocast(dtype=dtype)
    return contextlib.nullcontext()

def autocast_function(fn):
    """Decorator for AMP-enabled functions"""
    if amp is not None:
        return amp.autocast_function(fn)
    return fn


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
BACKEND_NAME = "cuda"
DEVICE_TYPE = "cuda"
