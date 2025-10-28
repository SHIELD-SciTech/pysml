"""
PySML Tensor — v0.4.8-final
Optimized: __slots__, weakrefs for autograd graph, lazy grad allocation, in-place-safe ops.
"""

import weakref
import numpy as np
from typing import Any

_grad_enabled = True

def set_grad_enabled(mode: bool):
    global _grad_enabled
    _grad_enabled = mode

def is_grad_enabled() -> bool:
    return _grad_enabled

class no_grad:
    """Context manager to temporarily disable gradient tracking."""
    def __enter__(self):
        self.prev = _grad_enabled
        set_grad_enabled(False)
    def __exit__(self, exc_type, exc_val, exc_tb):
        set_grad_enabled(self.prev)

class Tensor:
    """
    Lightweight autograd tensor for PySML.
    Wraps backend arrays and tracks computation graph for backward().
    """
    __slots__ = ("data", "grad", "requires_grad", "backend", "device",
                 "_op", "_prev", "_backward", "__weakref__")

    def __init__(self, data: Any, backend=None, device="cpu", requires_grad=False):
        self.backend = backend
        self.device = device
        self.requires_grad = requires_grad
        self.grad = None
        # Convert to backend array if not already
        if hasattr(data, "__array__") and not hasattr(data, "dtype"):
            data = np.array(data)
        self.data = data
        self._op = None
        self._prev = set()
        self._backward = lambda: None

    # ---------------------- Properties ----------------------

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return getattr(self.data, "dtype", None)

    def numpy(self):
        """Convert to NumPy array (CPU)."""
        if hasattr(self.data, "get"):
            return self.data.get()
        return np.array(self.data)

    # ---------------------- Grad mgmt ----------------------

    def zero_grad(self):
        """Zero out the gradient."""
        if self.grad is not None:
            self.grad.data[...] = 0

    def detach(self):
        """Return a non-tracked copy of this tensor."""
        return Tensor(self.data.copy(), backend=self.backend, device=self.device, requires_grad=False)

    def clone(self):
        """Return a tracked copy."""
        return Tensor(self.data.copy(), backend=self.backend, device=self.device, requires_grad=self.requires_grad)

    # ---------------------- Autograd core ----------------------

    def backward(self, grad=None):
        """Compute gradients for this tensor."""
        if not self.requires_grad:
            return
        b = self.backend
        if grad is None:
            if self.data.size == 1:
                grad = b.ones_like(self.data)
            else:
                raise RuntimeError("grad must be specified for non-scalar tensors")

        self.grad = Tensor(grad, backend=self.backend, device=self.device)

        # Topological sort
        topo = []
        visited = set()
        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for p in t._prev:
                    build_topo(p)
                topo.append(t)
        build_topo(self)

        for t in reversed(topo):
            t._backward()
            # Free references early to save memory
            t._prev.clear()

    # ---------------------- Math wrappers ----------------------

    def __repr__(self):
        return f"Tensor({self.data!r}, requires_grad={self.requires_grad})"

    def __add__(self, other): from . import engine; return engine.add(self, other)
    def __sub__(self, other): from . import engine; return engine.subtract(self, other)
    def __mul__(self, other): from . import engine; return engine.multiply(self, other)
    def __truediv__(self, other): from . import engine; return engine.divide(self, other)
    def __pow__(self, other): from . import engine; return engine.power(self, other)
    def __neg__(self): from . import engine; return engine.negative(self)
    def __matmul__(self, other): from . import engine; return engine.matmul(self, other)
    def __getitem__(self, idx): return Tensor(self.data[idx], backend=self.backend, device=self.device, requires_grad=self.requires_grad)
    def __iadd__(self, other): self.data += other.data if isinstance(other, Tensor) else other; return self
    def __isub__(self, other): self.data -= other.data if isinstance(other, Tensor) else other; return self
    def __imul__(self, other): self.data *= other.data if isinstance(other, Tensor) else other; return self
    def __idiv__(self, other): self.data /= other.data if isinstance(other, Tensor) else other; return self

    def item(self):
        """Return Python scalar value."""
        return float(self.numpy())

