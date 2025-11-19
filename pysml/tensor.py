"""Tensor implementation for PySML.

The previous version of this file provided only a very small subset of the
functionality exposed by frameworks such as PyTorch.  Several parts of the
library – including the official Transformer example – therefore crashed when
invoking convenience methods like :meth:`Tensor.reshape`.  This rewrite refreshes
the class with a modern, PyTorch-inspired API while preserving the existing
autograd engine.
"""

from __future__ import annotations

import gc
import weakref
from collections.abc import Sequence
from typing import Optional, Union, Callable, Any


from .dtype import bf16

DeviceLike = Union[str, "Tensor", None]
ShapeLike = Union[int, Sequence[int]]


class Tensor:
    """Light-weight multidimensional array supporting autograd."""

    __slots__ = (
        "data",
        "_requires_grad",
        "_grad",
        "_grad_fn",
        "_dtype",
        "_backend",
        "device",
        "active_device",
        "_post_backward_hooks",
        "__weakref__",
    )

    def __init__(
        self,
        data,
        dtype: Optional[object] = None,
        device: DeviceLike = "cpu",
        requires_grad: bool = False,
    ) -> None:
        self._requires_grad = requires_grad
        self._grad = None
        self._grad_fn = None

        self._dtype = dtype or DEFAULT_DTYPE
        backend, device_index, device_string = self._resolve_backend(device)
        self._backend = backend
        self.device = device_index
        self.active_device = device_string
        self.data = backend.convert(data, self._dtype, device=device_index)
        self._post_backward_hooks: list[Callable[["Tensor"], None]] = []

    # ------------------------------------------------------------------
    # Gradient handling
    # ------------------------------------------------------------------
    def requires_grad_(self, requires_grad: bool = True) -> "Tensor":
        self._requires_grad = requires_grad
        return self

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value) -> None:
        self._grad = value

    def zero_grad(self) -> None:
        self._grad = None
        if self._post_backward_hooks:
            # Reset hook readiness state if hooks store local metadata.
            # Hooks relying on cached information can use this signal to
            # reschedule reductions.
            for hook in list(self._post_backward_hooks):
                if hasattr(hook, "reset"):
                    try:
                        hook.reset()  # type: ignore[attr-defined]
                    except Exception:
                        # Hooks are user supplied; failure to reset should
                        # not crash gradient zeroing.
                        pass

    # ------------------------------------------------------------------
    # Gradient hooks
    # ------------------------------------------------------------------
    class _PostBackwardHookHandle:
        __slots__ = ("_tensor_ref", "_hook")

        def __init__(self, tensor: "Tensor", hook: callable) -> None:
            self._tensor_ref = weakref.ref(tensor)
            self._hook = hook

        def remove(self) -> None:
            tensor = self._tensor_ref()
            if tensor is not None:
                tensor._remove_post_backward_hook(self._hook)
            self._hook = None

    def register_post_backward_hook(self, hook: callable) -> "Tensor._PostBackwardHookHandle":
        """Register ``hook`` to be invoked after ``backward`` computes grads.

        The hook receives the owning :class:`Tensor` instance as the sole
        argument.  A :class:`handle <Tensor._PostBackwardHookHandle>` is
        returned which can be used to remove the hook.
        """

        if not callable(hook):
            raise TypeError("post-backward hook must be callable")
        self._post_backward_hooks.append(hook)
        return Tensor._PostBackwardHookHandle(self, hook)

    def _remove_post_backward_hook(self, hook: callable) -> None:
        try:
            self._post_backward_hooks.remove(hook)
        except ValueError:
            pass

    def _run_post_backward_hooks(self) -> None:
        if not self._post_backward_hooks:
            return
        # Copy to guard against modifications during iteration
        for hook in list(self._post_backward_hooks):
            try:
                hook(self)
            except Exception:
                # Hooks are user defined; swallow exceptions to avoid
                # destabilising the autograd engine.
                continue

    def backward(self, gradient: Optional["Tensor"] = None, retain_graph: bool = False) -> None:
        if not self._requires_grad:
            raise RuntimeError("Called backward() on tensor that doesn't require grad")

        if gradient is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified for non-scalar tensor")
            gradient = self._new_like(self._backend.asarray([1.0]), requires_grad=False)
        elif not isinstance(gradient, Tensor):
            gradient = self._new_like(self._backend.asarray([gradient]), requires_grad=False)

        topo_order: list[Tensor] = []
        visited: set[int] = set()

        def build_topo(node: Optional[Tensor]) -> None:
            if node is None or id(node) in visited:
                return
            visited.add(id(node))

            if node._grad_fn is not None:
                for input_ref in node._grad_fn.inputs:
                    if input_ref is not None:
                        input_tensor = input_ref()
                        if input_tensor is not None:
                            build_topo(input_tensor)

            topo_order.append(node)

        build_topo(self)

        self._grad = gradient

        for node in reversed(topo_order):
            if node._grad_fn is None:
                continue

            grad_output = node._grad
            if grad_output is None:
                continue

            grad_inputs = node._grad_fn.apply_backward(grad_output)

            for grad_info in grad_inputs:
                if grad_info is None:
                    continue

                input_tensor, grad_value = grad_info
                if grad_value is None or input_tensor is None:
                    continue

                if not isinstance(grad_value, Tensor):
                    grad_value = input_tensor._new_like(grad_value, requires_grad=False)

                if input_tensor._grad is None:
                    input_tensor._grad = grad_value
                else:
                    backend = input_tensor._backend
                    if hasattr(input_tensor._grad, "data") and hasattr(grad_value, "data"):
                        backend.add(
                            input_tensor._grad.data,
                            grad_value.data,
                            out=input_tensor._grad.data,
                        )
                    else:
                        summed = backend.add(
                            getattr(input_tensor._grad, "data", input_tensor._grad),
                            getattr(grad_value, "data", grad_value),
                        )
                        input_tensor._grad = input_tensor._new_like(summed, requires_grad=False)

                input_tensor._run_post_backward_hooks()

        if not retain_graph:
            self._grad_fn = None

    # ------------------------------------------------------------------
    # Device and dtype management
    # ------------------------------------------------------------------
    @property
    def backend_name(self) -> str:
        return getattr(self._backend, 'BACKEND_NAME', 'cpu')

    @property
    def device_type(self) -> str:
        if self.active_device is None:
            return 'cpu'
        return self.active_device.split(':', 1)[0]

    def to(
        self,
        device: DeviceLike = None,
        dtype: Optional[object] = None,
        *,
        copy: bool = False,
    ) -> "Tensor":
        if isinstance(device, Tensor):
            device = device.active_device

        target_device = device or self.active_device
        target_dtype = dtype or self._dtype

        backend, device_index, device_string = self._resolve_backend(target_device)
        needs_backend_switch = backend is not self._backend or copy
        needs_dtype_switch = target_dtype != self._dtype

        if needs_backend_switch or needs_dtype_switch:
            self._backend = backend
            self.device = device_index
            self.active_device = device_string
            self._dtype = target_dtype
            self.data = backend.convert(self.data, target_dtype, device=device_index)
        else:
            self._dtype = target_dtype

        return self

    def cpu(self) -> "Tensor":
        return self.to("cpu")

    def cuda(self, device: Optional[int] = None) -> "Tensor":
        target = "cuda" if device is None else f"cuda:{device}"
        return self.to(target)

    def xpu(self, device: Optional[int] = None) -> "Tensor":
        target = "xpu" if device is None else f"xpu:{device}"
        return self.to(target)

    # ------------------------------------------------------------------
    # Basic information and conversions
    # ------------------------------------------------------------------
    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def dim(self) -> int:
        return self.ndim

    def item(self) -> float:
        backend = self._backend
        if hasattr(backend, "asnumpy"):
            arr = backend.asnumpy(self.data)
        else:
            arr = self.data
        return float(arr.flat[0])

    def numpy(self):
        backend = self._backend
        if hasattr(backend, "asnumpy"):
            return backend.asnumpy(self.data)
        return self.data

    # ------------------------------------------------------------------
    # View & manipulation helpers
    # ------------------------------------------------------------------
    def reshape(self, *shape: ShapeLike) -> "Tensor":
        from . import engine

        if len(shape) == 1 and isinstance(shape[0], (tuple, list, Sequence)):
            target = tuple(shape[0])
        else:
            target = tuple(shape)
        return engine.reshape(self, target)

    def view(self, *shape: ShapeLike) -> "Tensor":
        return self.reshape(*shape)

    def permute(self, *dims: int) -> "Tensor":
        from . import engine

        if len(dims) == 1 and isinstance(dims[0], (tuple, list, Sequence)):
            dims = tuple(dims[0])
        else:
            dims = tuple(dims)
        return engine.permute(self, dims)

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        dims = list(range(self.ndim))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        return self.permute(dims)

    def T(self) -> "Tensor":  # noqa: D401
        """Return the transposed view of the tensor."""

        from . import engine

        return engine.transpose(self)

    def unsqueeze(self, dim: int) -> "Tensor":
        from . import engine

        return engine.unsqueeze(self, dim)

    def squeeze(self, dim: Optional[int] = None) -> "Tensor":
        from . import engine

        if dim is None:
            return engine.squeeze(self)
        return engine.squeeze(self, axis=dim)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
        from math import prod

        total_dims = self.ndim
        if total_dims == 0:
            return self.reshape(1)

        if start_dim < 0:
            start_dim += total_dims
        if end_dim < 0:
            end_dim += total_dims
        if start_dim > end_dim:
            raise ValueError("start_dim must be <= end_dim")

        prefix = self.shape[:start_dim]
        middle = prod(self.shape[start_dim : end_dim + 1])
        suffix = self.shape[end_dim + 1 :]
        return self.reshape(*(prefix + (middle,) + suffix))

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------
    def sum(self, axis=None, keepdims: bool = False):
        from . import engine

        return engine.sum_with_grad(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims: bool = False):
        from . import engine

        return engine.mean_with_grad(self, axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims: bool = False):
        from . import engine

        return engine.max(self, axis=axis, keepdims=keepdims)

    def min(self, axis=None, keepdims: bool = False):
        from . import engine

        return engine.min(self, axis=axis, keepdims=keepdims)

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------
    def detach(self) -> "Tensor":
        detached = Tensor.__new__(Tensor)
        detached._backend = self._backend
        detached._dtype = self._dtype
        detached._requires_grad = False
        detached._grad = None
        detached._grad_fn = None
        detached.device = self.device
        detached.active_device = self.active_device
        detached.data = self.data
        return detached

    def clone(self) -> "Tensor":
        cloned = Tensor.__new__(Tensor)
        cloned._backend = self._backend
        cloned._dtype = self._dtype
        cloned._grad = None
        cloned._requires_grad = self._requires_grad
        cloned._grad_fn = None
        cloned.device = self.device
        cloned.active_device = self.active_device
        cloned.data = self._backend.copy(self.data)
        return cloned

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------
    def free(self) -> None:
        if self.data is not None:
            del self.data
            self.data = None
        if self._grad is not None:
            del self._grad
            self._grad = None
        if self._grad_fn is not None:
            self._grad_fn = None
        gc.collect()

    def __del__(self):
        if hasattr(self, "data") and self.data is not None:
            del self.data
        if hasattr(self, "_grad") and self._grad is not None:
            del self._grad
        if hasattr(self, "_grad_fn"):
            self._grad_fn = None

    # ------------------------------------------------------------------
    # Python protocol helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        grad_fn_str = ""
        if getattr(self, "_grad_fn", None) is not None:
            grad_fn_str = f", grad_fn=<{self._grad_fn.__class__.__name__}>"
        return (
            f"Tensor({self.data!r}, dtype={self._dtype}, requires_grad={self._requires_grad}, "
            f"device='{self.active_device}'{grad_fn_str})"
        )

    __str__ = __repr__

    # ------------------------------------------------------------------
    # Arithmetic operator overloads
    # ------------------------------------------------------------------
    def __add__(self, other):
        from . import engine

        return engine.add(self, other)

    def __radd__(self, other):
        from . import engine

        return engine.add(self, other)

    def __sub__(self, other):
        from . import engine

        return engine.subtract(self, other)

    def __rsub__(self, other):
        from . import engine

        result = engine.subtract(self, other)
        return engine.negative(result)

    def __mul__(self, other):
        from . import engine

        return engine.multiply(self, other)

    def __rmul__(self, other):
        from . import engine

        return engine.multiply(self, other)

    def __truediv__(self, other):
        from . import engine

        return engine.divide(self, other)

    def __rtruediv__(self, other):
        from . import engine

        if not isinstance(other, Tensor):
            other_tensor = Tensor([other], dtype=self._dtype)
            other_tensor.to(self.active_device)
            return engine.divide(other_tensor, self)
        return engine.divide(other, self)

    def __pow__(self, other):
        from . import engine

        return engine.power(self, other)

    def __neg__(self):
        from . import engine

        return engine.negative(self)

    def __matmul__(self, other):
        from . import engine

        return engine.matmul(self, other)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _new_like(self, data, requires_grad: bool = False) -> "Tensor":
        new_tensor = Tensor.__new__(Tensor)
        new_tensor._backend = self._backend
        new_tensor._dtype = self._dtype
        new_tensor._requires_grad = requires_grad
        new_tensor._grad = None
        new_tensor._grad_fn = None
        new_tensor.device = self.device
        new_tensor.active_device = self.active_device
        new_tensor.data = data
        return new_tensor

    @staticmethod
    def _resolve_backend(device: DeviceLike):
        if isinstance(device, Tensor):
            device = device.active_device

        if device is None:
            device = "cpu"

        if isinstance(device, str):
            device = device.lower()
            if device.startswith("xpu"):
                from .xpu import backend as xpu_backend

                index = _parse_device_index(device)
                active = "xpu" if index is None else f"xpu:{index}"
                return xpu_backend, index, active
            if device.startswith("cuda"):
                from .cuda import backend as cuda_backend

                index = _parse_device_index(device)
                active = "cuda" if index is None else f"cuda:{index}"
                return cuda_backend, index, active
            if device == "cpu":
                from .cpu import backend as cpu_backend

                return cpu_backend, None, "cpu"
            raise ValueError(f"Unsupported device specification: {device!r}")

        raise TypeError(f"Expected str, Tensor or None for device, got {type(device)!r}")


def _parse_device_index(device: str) -> Optional[int]:
    if ":" not in device:
        return None
    _, index_str = device.split(":", 1)
    if not index_str:
        return None
    return int(index_str)


from .cpu import backend as cpu_backend

DEFAULT_BACKEND = cpu_backend
DEFAULT_DTYPE = bf16()