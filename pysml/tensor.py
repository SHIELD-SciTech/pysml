"""Tensor implementation for PySML.

The previous version of this file provided only a very small subset of the
functionality exposed by frameworks such as PyTorch. Several parts of the
library – including the official Transformer example – therefore crashed when
invoking convenience methods like :meth:`Tensor.reshape`. This rewrite refreshes
the class with a modern, PyTorch-inspired API while preserving the existing
autograd engine.
"""

from __future__ import annotations

import gc
import threading
import weakref
from functools import lru_cache
from collections.abc import Sequence
from typing import Optional, Union, Callable, Any


from .dtype import bf16
from .memory_pool import get_buffer_pool

DeviceLike = Union[str, "Tensor", None]
ShapeLike = Union[int, Sequence[int]]


class Tensor:
    """Light-weight multidimensional array supporting autograd."""

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._post_backward_hooks = []
        instance._grad_fn = None
        instance._grad = None
        instance._requires_grad = False
        instance._version = 0
        instance._grad_lock = threading.Lock()
        return instance

    __slots__ = (
        "data",
        "_requires_grad",
        "_grad",
        "_grad_fn",
        "_dtype",
        "_backend",
        "device",
        "active_device",
        "_version",
        "_post_backward_hooks",
        "_grad_lock",
        "_freed",
        "_shape",
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
        self._version = 0
        self._grad_lock = threading.Lock()
        self._freed = False

        self._dtype = dtype or DEFAULT_DTYPE
        backend, device_index, device_string = self._resolve_backend(device)
        self._backend = backend
        self.device = device_index
        self.active_device = device_string

        converted = backend.convert(data, self._dtype, device=device_index)
        self._shape = getattr(converted, "shape", getattr(data, "shape", None))
        buffer = _request_buffer(self._shape, self._dtype, backend, device_index)
        if buffer is not None:
            backend.copyto(buffer, converted)
            converted = buffer

        self.data = converted
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

        stack: list[tuple[Optional[Tensor], bool]] = [(self, False)]

        while stack:
            node, processed = stack.pop()
            if node is None:
                continue

            node_id = id(node)
            if processed:
                topo_order.append(node)
                continue

            if node_id in visited:
                continue

            visited.add(node_id)
            stack.append((node, True))

            if node._grad_fn is not None:
                for input_ref in node._grad_fn.inputs:
                    if input_ref is None:
                        continue
                    input_tensor = input_ref()
                    if input_tensor is not None:
                        stack.append((input_tensor, False))

        with self._grad_lock:
            if self._grad is None:
                self._grad = gradient.clone()
            else:
                backend = self._backend
                grad_buffer = getattr(self._grad, "data", self._grad)
                increment = getattr(gradient, "data", gradient)

                if hasattr(backend, "add_"):
                    backend.add_(grad_buffer, increment)
                else:
                    backend.add(
                        grad_buffer,
                        increment,
                        out=grad_buffer,
                    )

                if not hasattr(self._grad, "data"):
                    self._grad = self._new_like(grad_buffer, requires_grad=False)

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

                with input_tensor._grad_lock:
                    if input_tensor._grad is None:
                        input_tensor._grad = grad_value.clone()
                    else:
                        backend = input_tensor._backend
                        grad_buffer = getattr(input_tensor._grad, "data", input_tensor._grad)
                        increment = getattr(grad_value, "data", grad_value)

                        if hasattr(backend, "add_"):
                            backend.add_(grad_buffer, increment)
                        else:
                            backend.add(
                                grad_buffer,
                                increment,
                                out=grad_buffer,
                            )

                        if not hasattr(input_tensor._grad, "data"):
                            # ``_grad`` stored raw backend data; keep the updated buffer wrapped
                            input_tensor._grad = input_tensor._new_like(grad_buffer, requires_grad=False)

                input_tensor._run_post_backward_hooks()

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
        return getattr(self.data, "shape", getattr(self, "_shape", None))

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
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, Sequence)):
            target = tuple(shape[0])
        else:
            target = tuple(shape)
        return engine.reshape(self, target)

    def view(self, *shape: ShapeLike) -> "Tensor":
        return self.reshape(*shape)

    def permute(self, *dims: int) -> "Tensor":
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
        return engine.transpose(self)

    def unsqueeze(self, dim: int) -> "Tensor":
        return engine.unsqueeze(self, dim)

    def squeeze(self, dim: Optional[int] = None) -> "Tensor":
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
        return engine.sum_with_grad(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims: bool = False):
        return engine.mean_with_grad(self, axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims: bool = False):
        return engine.max(self, axis=axis, keepdims=keepdims)

    def min(self, axis=None, keepdims: bool = False):
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
        cloned._version = getattr(self, "_version", 0)

        data_shape = getattr(self.data, "shape", getattr(self, "_shape", None))
        cloned._shape = data_shape

        buffer = _request_buffer(data_shape, self._dtype, self._backend, self.device)
        if buffer is not None:
            self._backend.copyto(buffer, self.data)
            cloned.data = buffer
        else:
            try:
                cloned.data = self._backend.copy(self.data)
            except Exception:
                pointer_clone = _clone_pointer_like(
                    self.data, data_shape, self._dtype, self._backend, self.device
                )
                if pointer_clone is not None:
                    cloned.data = pointer_clone
                else:
                    # Fall back to a dtype/device-aware convert path for raw pointers
                    # (e.g., CuPy MemoryPointer) that lack array semantics.
                    cloned.data = self._backend.convert(self.data, self._dtype, device=self.device)
        return cloned

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------
    def free(self, *, collect: bool = False) -> None:
        if getattr(self, "_freed", False):
            return

        self._freed = True

        if getattr(self, "data", None) is not None:
            buffer_returner = globals().get("_return_buffer")
            try:
                if buffer_returner is not None and self._backend is not None:
                    buffer_returner(self.data, self._dtype, self._backend, self.device)
            finally:
                del self.data
                self.data = None
        if self._grad is not None:
            del self._grad
            self._grad = None
        if self._grad_fn is not None:
            self._grad_fn = None
        if collect:
            gc.collect()

    def __del__(self):
        try:
            self.free(collect=False)
        except Exception:
            # Avoid exceptions during interpreter shutdown
            pass

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

    def __getitem__(self, index):
        return engine.getitem(self, index)

    def __setitem__(self, index, value):
        normalized_index = engine._normalize_index(index)
        payload = value.data if isinstance(value, Tensor) else value
        try:
            payload = self._backend.convert(payload, self._dtype, device=self.device)
        except Exception:
            pass

        self.data[normalized_index] = payload
        self._version += 1

    # ------------------------------------------------------------------
    # Arithmetic operator overloads
    # ------------------------------------------------------------------
    def __add__(self, other):
        return engine.add(self, other)

    def __radd__(self, other):
        return engine.add(self, other)

    def __sub__(self, other):
        return engine.subtract(self, other)

    def __rsub__(self, other):
        result = engine.subtract(self, other)
        return engine.negative(result)

    def __mul__(self, other):
        return engine.multiply(self, other)

    def __rmul__(self, other):
        return engine.multiply(self, other)

    def __truediv__(self, other):
        return engine.divide(self, other)

    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other_tensor = Tensor([other], dtype=self._dtype)
            other_tensor.to(self.active_device)
            return engine.divide(other_tensor, self)
        return engine.divide(other, self)

    def __pow__(self, other):
        return engine.power(self, other)

    def __neg__(self):
        return engine.negative(self)

    def __matmul__(self, other):
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
        new_tensor._version = getattr(self, "_version", 0)

        new_tensor._shape = getattr(data, "shape", getattr(self, "_shape", None))

        buffer = _request_buffer(new_tensor._shape, new_tensor._dtype, new_tensor._backend, new_tensor.device)
        if buffer is not None:
            new_tensor._backend.copyto(buffer, data)
            data = buffer

        new_tensor.data = data
        return new_tensor

    def _bump_version(self) -> None:
        self._version = getattr(self, "_version", 0) + 1

    @staticmethod
    def _resolve_backend(device: DeviceLike):
        if isinstance(device, Tensor):
            device = device.active_device

        if device is None:
            device = "cpu"

        if isinstance(device, str):
            return _cached_resolve_device(device)

        raise TypeError(f"Expected str, Tensor or None for device, got {type(device)!r}")


@lru_cache(maxsize=16)
def _cached_resolve_device(device: str):
    device = device.lower()
    if device.startswith("xpu"):
        try:
            from .xpu import backend as xpu_backend
        except ImportError:
            raise RuntimeError(
                "XPU backend requested but not found. Please install 'dpnp' and 'dpctl'."
            )

        index = _parse_device_index(device)
        active = "xpu" if index is None else f"xpu:{index}"
        return xpu_backend, index, active
    if device.startswith("cuda"):
        try:
            from .cuda import backend as cuda_backend
        except ImportError:
            raise RuntimeError(
                "CUDA backend requested but not found. Please install 'cupy'."
            )

        index = _parse_device_index(device)
        active = "cuda" if index is None else f"cuda:{index}"
        return cuda_backend, index, active
    if device == "cpu":
        from .cpu import backend as cpu_backend

        return cpu_backend, None, "cpu"
    raise ValueError(f"Unsupported device specification: {device!r}")


def _parse_device_index(device: str) -> Optional[int]:
    if ":" not in device:
        return None
    _, index_str = device.split(":", 1)
    if not index_str:
        return None
    return int(index_str)


def _request_buffer(shape, dtype, backend, device):
    if shape is None or backend is None:
        return None
    if getattr(backend, "BACKEND_NAME", None) == "cpu":
        return None
    try:
        return get_buffer_pool().get_buffer(shape, dtype, backend, device)
    except Exception:
        return None


def _return_buffer(buffer, dtype, backend, device):
    if buffer is None or dtype is None or backend is None:
        return
    shape = getattr(buffer, "shape", None)
    if shape is None:
        return
    try:
        get_buffer_pool().return_buffer(buffer, shape, dtype, backend, device)
    except Exception:
        return


def _resolve_backend_dtype(dtype, backend):
    if hasattr(dtype, "precision") or hasattr(dtype, "precission"):
        key = getattr(dtype, "precision", getattr(dtype, "precission", None))
    else:
        key = dtype

    mapping = getattr(backend, "precision_map", None)
    if mapping and key in mapping:
        return mapping[key]
    return key


def _clone_pointer_like(data, shape, dtype, backend, device):
    if shape is None:
        return None

    if getattr(backend, "BACKEND_NAME", None) != "cuda":
        return None

    try:
        import cupy as cp
    except Exception:
        return None

    if not isinstance(data, cp.cuda.memory.MemoryPointer):
        return None

    resolved_dtype = _resolve_backend_dtype(dtype, backend)

    try:
        view = cp.ndarray(shape, dtype=resolved_dtype, memptr=data)
        return backend.copy(view)
    except Exception:
        try:
            view = cp.ndarray(shape, dtype=resolved_dtype, memptr=data)
            return backend.convert(view, dtype, device=device)
        except Exception:
            return None


from .cpu import backend as cpu_backend

DEFAULT_BACKEND = cpu_backend
DEFAULT_DTYPE = bf16()

# Import engine globally to avoid repeated local imports while sidestepping
# initialization cycles.
from . import engine
