

import contextlib
from . import backend

try:
    import cupy as cp
    AVAILABLE = backend.AVAILABLE
except ImportError:
    AVAILABLE = False
    import numpy as cp

_current_amp_dtype = None

@contextlib.contextmanager
def autocast(dtype="float16"):

    global _current_amp_dtype
    prev = _current_amp_dtype
    _current_amp_dtype = dtype
    try:
        yield
    finally:
        _current_amp_dtype = prev

def cast_tensor(t):

    if _current_amp_dtype is None:
        return t
    if not hasattr(t, "astype"):
        return t
    try:
        return t.astype(_current_amp_dtype)
    except Exception:
        return t

def autocast_function(fn):

    def wrapped(*args, **kwargs):
        if _current_amp_dtype is None:
            return fn(*args, **kwargs)
        casted_args = [cast_tensor(a) for a in args]
        return fn(*casted_args, **kwargs)
    return wrapped

def is_enabled():
    return _current_amp_dtype is not None

def get_dtype():
    return _current_amp_dtype

__all__ = ["autocast", "autocast_function", "cast_tensor", "is_enabled", "get_dtype"]

try:
    from .amp_scaler import GradScaler
except Exception:
    class GradScaler:
        def __init__(self, *a, **kw): raise RuntimeError("GradScaler unavailable")

__all__ += ["GradScaler"]
