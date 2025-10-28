"""
Automatic Mixed Precision (AMP) for Intel XPU backend.
Simulates autocast to float16/bfloat16 where supported.
"""

import contextlib
from . import backend

try:
    import dpnp as np
    AVAILABLE = backend.AVAILABLE
except ImportError:
    AVAILABLE = False
    import numpy as np

_current_amp_dtype = None

@contextlib.contextmanager
def autocast(dtype="float16"):
    """
    Context manager for automatic mixed precision on XPU.
    Example:
        >>> from pysml.xpu import amp
        >>> with amp.autocast(dtype="bfloat16"):
        ...     y = matmul(x1, x2)
    """
    global _current_amp_dtype
    prev = _current_amp_dtype
    _current_amp_dtype = dtype
    try:
        yield
    finally:
        _current_amp_dtype = prev

def cast_tensor(t):
    """Safely cast tensor to AMP dtype if enabled and supported."""
    if _current_amp_dtype is None or not hasattr(t, "astype"):
        return t
    target_dtype = (
        np.float16 if _current_amp_dtype.lower() in ("float16", "fp16") else
        np.bfloat16 if hasattr(np, "bfloat16") and _current_amp_dtype.lower() in ("bfloat16", "bf16")
        else None
    )
    if target_dtype is None:
        return t
    try:
        return t.astype(target_dtype)
    except Exception:
        return t

def autocast_function(fn):
    """Decorator that autocasts inputs for XPU functions."""
    def wrapped(*args, **kwargs):
        if _current_amp_dtype is None:
            return fn(*args, **kwargs)
        casted_args = [cast_tensor(a) for a in args]
        return fn(*casted_args, **kwargs)
    return wrapped

def is_enabled(): return _current_amp_dtype is not None
def get_dtype(): return _current_amp_dtype

__all__ = ["autocast", "autocast_function", "cast_tensor", "is_enabled", "get_dtype"]

try:
    from .amp_scaler import GradScaler
except Exception:
    class GradScaler:
        def __init__(self, *a, **kw): raise RuntimeError("GradScaler unavailable")

__all__ += ["GradScaler"]


