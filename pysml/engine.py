from pysml.tensor import Tensor
from pysml.autograd import (
Function, is_grad_enabled,
backward_add, backward_subtract, backward_multiply, backward_divide,
backward_power, backward_matmul, backward_relu, backward_exp,
backward_log, backward_tanh, backward_sum, backward_mean, backward_transpose,
    backward_reshape, backward_sigmoid, backward_sqrt, backward_sin,
    backward_cos,
    # NEW IMPORTS BELOW
    backward_softmax, backward_log_softmax, backward_gelu, backward_silu,
    backward_layer_norm, backward_rms_norm, backward_batch_norm, backward_group_norm,
    backward_dropout, backward_embedding,
        backward_permute, backward_unsqueeze,
        backward_abs, backward_sign, backward_clip, backward_where,
        backward_maximum, backward_minimum,
    backward_max_reduce, backward_min_reduce,
    backward_split, backward_getitem, backward_concatenate,
)
import builtins
import gc

from pysml.memory_pool import get_buffer_pool

backend_priority = ["cpu", "xpu", "cuda"]

def _requires_grad(obj):
        """Safely check if an object requires gradients."""
        return getattr(obj, '_requires_grad', False) if obj is not None else False


def _backend(*tensors):
        backend = tensors[0]._backend
        backend_index = backend_priority.index(backend.BACKEND_NAME)
        for tensor in tensors:
                if backend_priority.index(tensor._backend.BACKEND_NAME) > backend_index:
                        backend = tensor._backend
                        backend_index = backend_priority.index(backend.BACKEND_NAME)
                else:
                        if backend_index == 2:
                                break
        return backend


def _normalize_index(index):
        if isinstance(index, Tensor):
                return index.data
        if isinstance(index, (list, tuple)):
                return tuple(_normalize_index(i) for i in index)
        return index


ENSURE_BACKEND = False


_BUFFER_POOL = get_buffer_pool()


def _maybe_allocate_buffer(shape, dtype, backend, device):
        if shape is None:
                return None
        if getattr(backend, "BACKEND_NAME", None) == "cpu":
                return None
        try:
                return _BUFFER_POOL.get_buffer(shape, dtype, backend, device)
        except Exception:
                return None


def _wrap_result(template, backend, requires_grad, data):
        pooled = data
        buffer = _maybe_allocate_buffer(getattr(data, "shape", None), template._dtype, backend, template.device)
        if buffer is not None and buffer is not data:
                backend.copyto(buffer, data)
                pooled = buffer

        out = Tensor.__new__(Tensor)
        out._requires_grad = requires_grad
        out._grad = None
        out._grad_fn = None
        out._dtype = template._dtype
        out._backend = backend
        out.device = template.device
        out.active_device = template.active_device
        out._version = getattr(template, "_version", 0)
        out._shape = getattr(pooled, "shape", getattr(data, "shape", getattr(template, "_shape", None)))
        out.data = pooled
        return out


def _prepare_inplace_out(out, template, backend, requires_grad):
        out._requires_grad = requires_grad
        out._grad = None
        out._grad_fn = None
        out._dtype = template._dtype
        out._backend = backend
        out.device = template.device
        out.active_device = template.active_device
        out._shape = getattr(out.data, "shape", getattr(template, "_shape", None))
        out._bump_version()
        return out


def _attach_grad_fn(out, backward_fn, inputs, metadata=None):
        if is_grad_enabled() and out._requires_grad:
                out._grad_fn = Function(backward_fn, inputs, metadata=metadata or {})
        else:
                out._grad_fn = None


# ============================================================================
# Arithmetic operations with autograd
# ============================================================================

def add(input, other, alpha=1, out=None):
        backend = input._backend
        if ENSURE_BACKEND and hasattr(other, '_backend'):
                backend = _backend(input, other)
        
        other_data = other.data if hasattr(other, 'data') else other
        
        # Handle alpha scaling
        if alpha != 1:
                other_data = backend.multiply(other_data, alpha)

        if out is None:
                buffer = _maybe_allocate_buffer(input.data.shape, input._dtype, backend, input.device)
                if buffer is not None:
                        result_data = backend.add(input.data, other_data, out=buffer)
                else:
                        result_data = backend.add(input.data, other_data)

                out = _wrap_result(input, backend, input._requires_grad or _requires_grad(other), result_data)
        else:
                backend.add(input.data, other_data, out=out.data)
                out = _prepare_inplace_out(out, input, backend, input._requires_grad or _requires_grad(other))

        _attach_grad_fn(out, backward_add, [input, other if hasattr(other, '_requires_grad') else None], metadata={'alpha': alpha})
        return out


def subtract(input, other, out=None):
        backend = input._backend
        if ENSURE_BACKEND and hasattr(other, '_backend'):
                backend = _backend(input, other)
        
        other_data = other.data if hasattr(other, 'data') else other

        if out is None:
                buffer = _maybe_allocate_buffer(input.data.shape, input._dtype, backend, input.device)
                if buffer is not None:
                        result_data = backend.subtract(input.data, other_data, out=buffer)
                else:
                        result_data = backend.subtract(input.data, other_data)

                out = _wrap_result(input, backend, input._requires_grad or _requires_grad(other), result_data)
        else:
                backend.subtract(input.data, other_data, out=out.data)

                out = _prepare_inplace_out(out, input, backend, input._requires_grad or _requires_grad(other))

        _attach_grad_fn(out, backward_subtract, [input, other if hasattr(other, '_requires_grad') else None])
        return out


def multiply(input, other, out=None):
        backend = input._backend
        if ENSURE_BACKEND and hasattr(other, '_backend'):
                backend = _backend(input, other)

        other_data = other.data if hasattr(other, 'data') else other
        is_scalar = not hasattr(other, 'data')

        if out is None:
                buffer = _maybe_allocate_buffer(input.data.shape, input._dtype, backend, input.device)
                if buffer is not None:
                        result_data = backend.multiply(input.data, other_data, out=buffer)
                else:
                        result_data = backend.multiply(input.data, other_data)

                out = _wrap_result(input, backend, input._requires_grad or _requires_grad(other), result_data)
        else:
                backend.multiply(input.data, other_data, out=out.data)

                out = _prepare_inplace_out(out, input, backend, input._requires_grad or _requires_grad(other))

        metadata = {}
        if is_scalar:
                metadata['other_scalar'] = other_data

        _attach_grad_fn(out, backward_multiply, [input, other if hasattr(other, '_requires_grad') else None], metadata=metadata)
        return out


def divide(input, other, out=None):
        backend = input._backend
        if ENSURE_BACKEND and hasattr(other, '_backend'):
                backend = _backend(input, other)
        
        other_data = other.data if hasattr(other, 'data') else other
        is_scalar = not hasattr(other, 'data')

        if out is None:
                buffer = _maybe_allocate_buffer(input.data.shape, input._dtype, backend, input.device)
                if buffer is not None:
                        result_data = backend.divide(input.data, other_data, out=buffer)
                else:
                        result_data = backend.divide(input.data, other_data)

                out = _wrap_result(input, backend, input._requires_grad or _requires_grad(other), result_data)
        else:
                backend.divide(input.data, other_data, out=out.data)

                out = _prepare_inplace_out(out, input, backend, input._requires_grad or _requires_grad(other))

        metadata = {}
        if is_scalar:
                metadata['other_scalar'] = other_data

        _attach_grad_fn(out, backward_divide, [input, other if hasattr(other, '_requires_grad') else None], metadata=metadata)
        return out


def power(input, exponent, out=None):
        backend = input._backend
        if ENSURE_BACKEND and hasattr(exponent, '_backend'):
                backend = _backend(input, exponent)
        
        exponent_data = exponent.data if hasattr(exponent, 'data') else exponent
        exponent_value = float(exponent_data) if not hasattr(exponent_data, '__len__') else exponent_data

        if out is None:
                buffer = _maybe_allocate_buffer(input.data.shape, input._dtype, backend, input.device)
                if buffer is not None:
                        result_data = backend.power(input.data, exponent_data, out=buffer)
                else:
                        result_data = backend.power(input.data, exponent_data)

                out = _wrap_result(input, backend, input._requires_grad, result_data)
        else:
                backend.power(input.data, exponent_data, out=out.data)

                out = _prepare_inplace_out(out, input, backend, input._requires_grad)

        _attach_grad_fn(out, backward_power, [input], metadata={'exponent': exponent_value})
        return out


def negative(input, out=None):
        backend = input._backend

        if out is None:
                buffer = _maybe_allocate_buffer(input.data.shape, input._dtype, backend, input.device)
                if buffer is not None:
                        result_data = backend.negative(input.data, out=buffer)
                else:
                        result_data = backend.negative(input.data)

                out = _wrap_result(input, backend, input._requires_grad, result_data)
        else:
                backend.negative(input.data, out=out.data)

                out = _prepare_inplace_out(out, input, backend, input._requires_grad)

        _attach_grad_fn(out, backward_multiply, [input, None], metadata={'other_scalar': -1.0})
        return out


# ============================================================================
# Matrix operations with autograd
# ============================================================================

def matmul(input, other, out=None):
        backend = input._backend
        if ENSURE_BACKEND and hasattr(other, '_backend'):
                backend = _backend(input, other)

        if out is None:
                buffer = None
                if getattr(input.data, "ndim", 0) == 2 and getattr(other.data, "ndim", 0) == 2:
                        buffer = _maybe_allocate_buffer(
                                (input.data.shape[0], other.data.shape[1]),
                                input._dtype,
                                backend,
                                input.device,
                        )

                if buffer is not None:
                        result_data = backend.matmul(input.data, other.data, out=buffer)
                else:
                        result_data = backend.matmul(input.data, other.data)

                out = _wrap_result(input, backend, input._requires_grad or _requires_grad(other), result_data)
        else:
                try:
                        backend.matmul(input.data, other.data, out=out.data)
                except:
                        result = backend.matmul(input.data, other.data)
                        backend.copyto(out.data, result)

                out = _prepare_inplace_out(out, input, backend, input._requires_grad or _requires_grad(other))

        _attach_grad_fn(out, backward_matmul, [input, other])
        return out


# ============================================================================
# Activation functions with autograd
# ============================================================================

def relu(input, out=None):
        backend = input._backend

        if out is None:
                buffer = _maybe_allocate_buffer(input.data.shape, input._dtype, backend, input.device)
                if buffer is not None:
                        result_data = backend.maximum(input.data, 0, out=buffer)
                else:
                        result_data = backend.maximum(input.data, 0)

                out = _wrap_result(input, backend, input._requires_grad, result_data)
        else:
                backend.maximum(input.data, 0, out=out.data)

                out = _prepare_inplace_out(out, input, backend, input._requires_grad)

        _attach_grad_fn(out, backward_relu, [input])
        return out


def exp(input, out=None):
        backend = input._backend

        if out is None:
                buffer = _maybe_allocate_buffer(input.data.shape, input._dtype, backend, input.device)
                if buffer is not None:
                        result_data = backend.exp(input.data, out=buffer)
                else:
                        result_data = backend.exp(input.data)

                out = _wrap_result(input, backend, input._requires_grad, result_data)
        else:
                backend.exp(input.data, out=out.data)

                out = _prepare_inplace_out(out, input, backend, input._requires_grad)

        _attach_grad_fn(out, backward_exp, [input])
        return out


def log(input, out=None):
        backend = input._backend

        if out is None:
                buffer = _maybe_allocate_buffer(input.data.shape, input._dtype, backend, input.device)
                if buffer is not None:
                        result_data = backend.log(input.data, out=buffer)
                else:
                        result_data = backend.log(input.data)

                out = _wrap_result(input, backend, input._requires_grad, result_data)
        else:
                backend.log(input.data, out=out.data)

                out = _prepare_inplace_out(out, input, backend, input._requires_grad)

        _attach_grad_fn(out, backward_log, [input])
        return out


def tanh(input, out=None):
        backend = input._backend

        if out is None:
                buffer = _maybe_allocate_buffer(input.data.shape, input._dtype, backend, input.device)
                if buffer is not None:
                        result_data = backend.tanh(input.data, out=buffer)
                else:
                        result_data = backend.tanh(input.data)

                out = _wrap_result(input, backend, input._requires_grad, result_data)
        else:
                backend.tanh(input.data, out=out.data)

                out = _prepare_inplace_out(out, input, backend, input._requires_grad)

        _attach_grad_fn(out, backward_tanh, [input], metadata={'output': out.data})
        return out


def sigmoid(input, out=None):
        backend = input._backend

        if out is None:
                # sigmoid(x) = 1 / (1 + exp(-x))
                temp = backend.add(1.0, backend.exp(backend.negative(input.data)))
                buffer = _maybe_allocate_buffer(input.data.shape, input._dtype, backend, input.device)
                if buffer is not None:
                        result_data = backend.reciprocal(temp, out=buffer)
                else:
                        result_data = backend.reciprocal(temp)

                out = _wrap_result(input, backend, input._requires_grad, result_data)
        else:
                temp = backend.reciprocal(
                        backend.add(1.0, backend.exp(backend.negative(input.data)))
                )
                backend.copyto(out.data, temp)

                out = _prepare_inplace_out(out, input, backend, input._requires_grad)

        _attach_grad_fn(out, backward_sigmoid, [input], metadata={'output': out.data})
        return out


def sqrt(input, out=None):
        backend = input._backend

        if out is None:
                buffer = _maybe_allocate_buffer(input.data.shape, input._dtype, backend, input.device)
                if buffer is not None:
                        result_data = backend.sqrt(input.data, out=buffer)
                else:
                        result_data = backend.sqrt(input.data)

                out = _wrap_result(input, backend, input._requires_grad, result_data)
        else:
                backend.sqrt(input.data, out=out.data)

                out = _prepare_inplace_out(out, input, backend, input._requires_grad)

        _attach_grad_fn(out, backward_sqrt, [input])
        return out


def square(input, out=None):
        # Just use power with exponent=2
        return power(input, 2, out=out)


def sin(input, out=None):
        backend = input._backend
        
        if out is None:
                result_data = backend.sin(input.data)
                out = Tensor.__new__(Tensor)
                out._requires_grad = input._requires_grad
                out._grad = None
                out._dtype = input._dtype
                out._backend = backend
                out.device = input.device
                out.active_device = input.active_device
                out.data = result_data
                
                # Build computational graph
                if is_grad_enabled() and out._requires_grad:
                        out._grad_fn = Function(
                                backward_sin,
                                [input],
                                metadata={}
                        )
        else:
                backend.sin(input.data, out=out.data)
        
        return out


def cos(input, out=None):
        backend = input._backend
        
        if out is None:
                result_data = backend.cos(input.data)
                out = Tensor.__new__(Tensor)
                out._requires_grad = input._requires_grad
                out._grad = None
                out._dtype = input._dtype
                out._backend = backend
                out.device = input.device
                out.active_device = input.active_device
                out.data = result_data
                
                # Build computational graph
                if is_grad_enabled() and out._requires_grad:
                        out._grad_fn = Function(
                                backward_cos,
                                [input],
                                metadata={}
                        )
        else:
                backend.cos(input.data, out=out.data)
        
        return out


# ============================================================================
# Reduction operations with autograd
# ============================================================================

def sum_with_grad(input, axis=None, keepdims=False):
        backend = input._backend
        result_data = backend.sum(input.data, axis=axis, keepdims=keepdims)
        
        # Wrap in Tensor
        out = Tensor.__new__(Tensor)
        out._requires_grad = input._requires_grad
        out._grad = None
        out._dtype = input._dtype
        out._backend = backend
        out.device = input.device
        out.active_device = input.active_device
        out.data = result_data
        
        # Build computational graph
        if is_grad_enabled() and out._requires_grad:
                out._grad_fn = Function(
                        backward_sum,
                        [input],
                        metadata={'axis': axis, 'keepdims': keepdims}
                )
        
        return out


def mean_with_grad(input, axis=None, keepdims=False):
    backend = input._backend
    result_data = backend.mean(input.data, axis=axis, keepdims=keepdims)

    # Wrap in Tensor
    out = Tensor.__new__(Tensor)
    out._requires_grad = input._requires_grad
    out._grad = None
    out._dtype = input._dtype
    out._backend = backend
    out.device = input.device
    out.active_device = input.active_device
    out.data = result_data

    # Build computational graph (mean = sum / n)
    if is_grad_enabled() and out._requires_grad:
        if axis is None:
            n = input.data.size
        else:
            axes = axis if isinstance(axis, (tuple, list)) else (axis,)
            n = 1
            for ax in axes:
                n *= input.data.shape[ax if ax >= 0 else ax + input.data.ndim]

        out._grad_fn = Function(
            backward_mean,
            [input],
            metadata={'axis': axis, 'keepdims': keepdims, 'n': n}
        )

    return out


# ============================================================================
# Differentiable reduction wrappers (alias to *_with_grad for clarity)
# ============================================================================

def sum(input, axis=None, keepdims=False):
        return sum_with_grad(input, axis=axis, keepdims=keepdims)


def mean(input, axis=None, keepdims=False):
        return mean_with_grad(input, axis=axis, keepdims=keepdims)


def max(input, axis=None, keepdims=False):
        backend = input._backend
        result_data = backend.max(input.data, axis=axis, keepdims=keepdims)
        result_data = backend.asarray(result_data)

        out = Tensor.__new__(Tensor)
        out._requires_grad = input._requires_grad
        out._grad = None
        out._dtype = input._dtype
        out._backend = backend
        out.device = input.device
        out.active_device = input.active_device
        out.data = result_data

        if is_grad_enabled() and out._requires_grad:
                if axis is None:
                        indices = backend.argmax(input.data)
                        if hasattr(indices, 'item'):
                                indices = int(indices.item())
                        else:
                                indices = int(indices)
                else:
                        indices = backend.argmax(input.data, axis=axis)
                        indices = backend.expand_dims(indices, axis=axis)

                out._grad_fn = Function(
                        backward_max_reduce,
                        [input],
                        metadata={
                                'axis': axis,
                                'keepdims': keepdims,
                                'max_indices': indices,
                        },
                )

        return out


def min(input, axis=None, keepdims=False):
        backend = input._backend
        result_data = backend.min(input.data, axis=axis, keepdims=keepdims)
        result_data = backend.asarray(result_data)

        out = Tensor.__new__(Tensor)
        out._requires_grad = input._requires_grad
        out._grad = None
        out._dtype = input._dtype
        out._backend = backend
        out.device = input.device
        out.active_device = input.active_device
        out.data = result_data

        if is_grad_enabled() and out._requires_grad:
                if axis is None:
                        indices = backend.argmin(input.data)
                        if hasattr(indices, 'item'):
                                indices = int(indices.item())
                        else:
                                indices = int(indices)
                else:
                        indices = backend.argmin(input.data, axis=axis)
                        indices = backend.expand_dims(indices, axis=axis)

                out._grad_fn = Function(
                        backward_min_reduce,
                        [input],
                        metadata={
                                'axis': axis,
                                'keepdims': keepdims,
                                'min_indices': indices,
                        },
                )

        return out


# ============================================================================
# Operations without autograd (for now)
# ============================================================================

def transpose(input, axes=None):
        backend = input._backend
        if axes is None:
                result_data = backend.transpose(input.data)
        else:
                result_data = backend.transpose(input.data, axes)
        
        out = Tensor.__new__(Tensor)
        out._requires_grad = input._requires_grad
        out._grad = None
        out._dtype = input._dtype
        out._backend = backend
        out.device = input.device
        out.active_device = input.active_device
        out.data = result_data
        
        # Build computational graph
        if is_grad_enabled() and out._requires_grad:
                out._grad_fn = Function(
                        backward_transpose,
                        [input],
                        metadata={'axes': axes}
                )
        
        return out


def reshape(input, shape):
        backend = input._backend
        result_data = backend.reshape(input.data, shape)
        
        out = Tensor.__new__(Tensor)
        out._requires_grad = input._requires_grad
        out._grad = None
        out._dtype = input._dtype
        out._backend = backend
        out.device = input.device
        out.active_device = input.active_device
        out.data = result_data
        
        # Build computational graph
        if is_grad_enabled() and out._requires_grad:
                out._grad_fn = Function(
                        backward_reshape,
                        [input],
                        metadata={'original_shape': input.shape}
                )
        
        return out


# Import remaining operations from original engine (without autograd for now)
def positive(input, out=None):
        backend = input._backend
        if out is None:
                result_data = backend.positive(input.data)
                out = Tensor.__new__(Tensor)
                out._requires_grad = input._requires_grad
                out._grad = None
                out._grad_fn = None  # Gradient is just passed through
                out._dtype = input._dtype
                out._backend = backend
                out.device = input.device
                out.active_device = input.active_device
                out.data = result_data
        else:
                backend.positive(input.data, out=out.data)
        return out


def abs(input, out=None):
        backend = input._backend
        if out is None:
                result_data = backend.abs(input.data)
                out = Tensor.__new__(Tensor)
                out._requires_grad = input._requires_grad
                out._grad = None
                out._grad_fn = None
                out._dtype = input._dtype
                out._backend = backend
                out.device = input.device
                out.active_device = input.active_device
                out.data = result_data
        else:
                backend.abs(input.data, out=out.data)
                out = _prepare_inplace_out(out, input, backend, input._requires_grad)

        if is_grad_enabled() and out._requires_grad:
                out._grad_fn = Function(
                        backward_abs,
                        [input],
                        metadata={},
                )
        return out


absolute = abs  # Alias


def floor_divide(input, other, out=None):
        backend = input._backend
        if ENSURE_BACKEND and hasattr(other, '_backend'):
                backend = _backend(input, other)
        
        other_data = other.data if hasattr(other, 'data') else other
        
        if out is None:
                result_data = backend.floor_divide(input.data, other_data)
                out = Tensor.__new__(Tensor)
                out._requires_grad = False  # Not differentiable
                out._grad = None
                out._grad_fn = None
                out._dtype = input._dtype
                out._backend = backend
                out.device = input.device
                out.active_device = input.active_device
                out.data = result_data
        else:
                backend.floor_divide(input.data, other_data, out=out.data)
        
        return out


def remainder(input, other, out=None):
        backend = input._backend
        if ENSURE_BACKEND and hasattr(other, '_backend'):
                backend = _backend(input, other)
        
        other_data = other.data if hasattr(other, 'data') else other
        
        if out is None:
                result_data = backend.remainder(input.data, other_data)
                out = Tensor.__new__(Tensor)
                out._requires_grad = False  # Not differentiable
                out._grad = None
                out._grad_fn = None
                out._dtype = input._dtype
                out._backend = backend
                out.device = input.device
                out.active_device = input.active_device
                out.data = result_data
        else:
                backend.remainder(input.data, other_data, out=out.data)
        
        return out


def mod(input, other, out=None):
        backend = input._backend
        if ENSURE_BACKEND and hasattr(other, '_backend'):
                backend = _backend(input, other)
        
        other_data = other.data if hasattr(other, 'data') else other
        
        if out is None:
                result_data = backend.mod(input.data, other_data)
                out = Tensor.__new__(Tensor)
                out._requires_grad = False  # Not differentiable
                out._grad = None
                out._grad_fn = None
                out._dtype = input._dtype
                out._backend = backend
                out.device = input.device
                out.active_device = input.active_device
                out.data = result_data
        else:
                backend.mod(input.data, other_data, out=out.data)
        
        return out


def sign(input, out=None):
        backend = input._backend
        if out is None:
                result_data = backend.sign(input.data)
                out = Tensor.__new__(Tensor)
                out._requires_grad = input._requires_grad
                out._grad = None
                out._grad_fn = None
                out._dtype = input._dtype
                out._backend = backend
                out.device = input.device
                out.active_device = input.active_device
                out.data = result_data
        else:
                backend.sign(input.data, out=out.data)
                out = _prepare_inplace_out(out, input, backend, input._requires_grad)

        if is_grad_enabled() and out._requires_grad:
                out._grad_fn = Function(
                        backward_sign,
                        [input],
                        metadata={},
                )
        return out


# ============================================================================
# Additional operations without autograd
# ============================================================================

def log10(input, out=None):
        backend = input._backend
        if out is None:
                result_data = backend.log10(input.data)
                out = Tensor.__new__(Tensor)
                out._requires_grad = input._requires_grad
                out._grad = None
                out._grad_fn = None  # Can be added: grad = grad_out / (x * ln(10))
                out._dtype = input._dtype
                out._backend = backend
                out.device = input.device
                out.active_device = input.active_device
                out.data = result_data
        else:
                backend.log10(input.data, out=out.data)
        return out


def log2(input, out=None):
        backend = input._backend
        if out is None:
                result_data = backend.log2(input.data)
                out = Tensor.__new__(Tensor)
                out._requires_grad = input._requires_grad
                out._grad = None
                out._grad_fn = None  # Can be added: grad = grad_out / (x * ln(2))
                out._dtype = input._dtype
                out._backend = backend
                out.device = input.device
                out.active_device = input.active_device
                out.data = result_data
        else:
                backend.log2(input.data, out=out.data)
        return out


def tan(input, out=None):
        backend = input._backend
        if out is None:
                result_data = backend.tan(input.data)
                out = Tensor.__new__(Tensor)
                out._requires_grad = input._requires_grad
                out._grad = None
                out._grad_fn = None  # Can be added
                out._dtype = input._dtype
                out._backend = backend
                out.device = input.device
                out.active_device = input.active_device
                out.data = result_data
        else:
                backend.tan(input.data, out=out.data)
        return out


def arcsin(input, out=None):
        backend = input._backend
        if out is None:
                result_data = backend.arcsin(input.data)
                out = Tensor.__new__(Tensor)
                out._requires_grad = input._requires_grad
                out._grad = None
                out._grad_fn = None
                out._dtype = input._dtype
                out._backend = backend
                out.device = input.device
                out.active_device = input.active_device
                out.data = result_data
        else:
                backend.arcsin(input.data, out=out.data)
        return out


def arccos(input, out=None):
        backend = input._backend
        if out is None:
                result_data = backend.arccos(input.data)
                out = Tensor.__new__(Tensor)
                out._requires_grad = input._requires_grad
                out._grad = None
                out._grad_fn = None
                out._dtype = input._dtype
                out._backend = backend
                out.device = input.device
                out.active_device = input.active_device
                out.data = result_data
        else:
                backend.arccos(input.data, out=out.data)
        return out


def arctan(input, out=None):
        backend = input._backend
        if out is None:
                result_data = backend.arctan(input.data)
                out = Tensor.__new__(Tensor)
                out._requires_grad = input._requires_grad
                out._grad = None
                out._grad_fn = None
                out._dtype = input._dtype
                out._backend = backend
                out.device = input.device
                out.active_device = input.active_device
                out.data = result_data
        else:
                backend.arctan(input.data, out=out.data)
        return out


def sinh(input, out=None):
        backend = input._backend
        if out is None:
                result_data = backend.sinh(input.data)
                out = Tensor.__new__(Tensor)
                out._requires_grad = input._requires_grad
                out._grad = None
                out._grad_fn = None
                out._dtype = input._dtype
                out._backend = backend
                out.device = input.device
                out.active_device = input.active_device
                out.data = result_data
        else:
                backend.sinh(input.data, out=out.data)
        return out


def cosh(input, out=None):
        backend = input._backend
        if out is None:
                result_data = backend.cosh(input.data)
                out = Tensor.__new__(Tensor)
                out._requires_grad = input._requires_grad
                out._grad = None
                out._grad_fn = None
                out._dtype = input._dtype
                out._backend = backend
                out.device = input.device
                out.active_device = input.active_device
                out.data = result_data
        else:
                backend.cosh(input.data, out=out.data)
        return out


# ============================================================================
# Tensor manipulation operations
# ============================================================================

def squeeze(input, axis=None):
        backend = input._backend
        if axis is None:
                result_data = backend.squeeze(input.data)
        else:
                result_data = backend.squeeze(input.data, axis)
        
        out = Tensor.__new__(Tensor)
        out._requires_grad = input._requires_grad
        out._grad = None
        out._grad_fn = None  # View operation - can add grad
        out._dtype = input._dtype
        out._backend = backend
        out.device = input.device
        out.active_device = input.active_device
        out.data = result_data
        return out


def expand_dims(input, axis):
        backend = input._backend
        result_data = backend.expand_dims(input.data, axis)
        
        out = Tensor.__new__(Tensor)
        out._requires_grad = input._requires_grad
        out._grad = None
        out._grad_fn = None  # View operation - can add grad
        out._dtype = input._dtype
        out._backend = backend
        out.device = input.device
        out.active_device = input.active_device
        out.data = result_data
        return out


def concatenate(tensors, axis=0):
        if not tensors:
                raise ValueError("Need at least one tensor to concatenate")
        
        backend = tensors[0]._backend
        data_list = [t.data for t in tensors]
        result_data = backend.concatenate(data_list, axis=axis)
        
        out = Tensor.__new__(Tensor)
        out._requires_grad = any(t._requires_grad for t in tensors)
        out._grad = None
        out._grad_fn = None
        out._dtype = tensors[0]._dtype
        out._backend = backend
        out.device = tensors[0].device
        out.active_device = tensors[0].active_device
        out.data = result_data

        if is_grad_enabled() and out._requires_grad:
                sizes = [t.shape[axis] for t in tensors]
                out._grad_fn = Function(
                        backward_concatenate,
                        [t if t._requires_grad else None for t in tensors],
                        metadata={'axis': axis, 'sizes': sizes},
                )
        return out


def stack(tensors, axis=0):
        if not tensors:
                raise ValueError("Need at least one tensor to stack")
        
        backend = tensors[0]._backend
        data_list = [t.data for t in tensors]
        result_data = backend.stack(data_list, axis=axis)
        
        out = Tensor.__new__(Tensor)
        out._requires_grad = any(t._requires_grad for t in tensors)
        out._grad = None
        out._grad_fn = None  # Can add grad
        out._dtype = tensors[0]._dtype
        out._backend = backend
        out.device = tensors[0].device
        out.active_device = tensors[0].active_device
        out.data = result_data
        return out


# ============================================================================
# Other matrix operations
# ============================================================================

def dot(input, other):
        backend = input._backend
        if ENSURE_BACKEND and hasattr(other, '_backend'):
                backend = _backend(input, other)
        
        result = backend.dot(input.data, other.data)
        return result


def outer(input, other):
        backend = input._backend
        if ENSURE_BACKEND and hasattr(other, '_backend'):
                backend = _backend(input, other)
        
        result_data = backend.outer(input.data, other.data)
        
        out = Tensor.__new__(Tensor)
        out._requires_grad = input._requires_grad or _requires_grad(other)
        out._grad = None
        out._grad_fn = None  # Can add grad
        out._dtype = input._dtype
        out._backend = backend
        out.device = input.device
        out.active_device = input.active_device
        out.data = result_data
        
        return out


def inner(input, other):
        backend = input._backend
        if ENSURE_BACKEND and hasattr(other, '_backend'):
                backend = _backend(input, other)
        
        result = backend.inner(input.data, other.data)
        return result


# ============================================================================
# Comparison and utility operations
# ============================================================================

def maximum(input, other, out=None):
        backend = input._backend
        if ENSURE_BACKEND and hasattr(other, '_backend'):
                backend = _backend(input, other)
        
        other_data = other.data if hasattr(other, 'data') else other
        
        if out is None:
                result_data = backend.maximum(input.data, other_data)
                out = Tensor.__new__(Tensor)
                out._requires_grad = input._requires_grad or _requires_grad(other)
                out._grad = None
                out._grad_fn = None  # Can add grad based on which input was larger
                out._dtype = input._dtype
                out._backend = backend
                out.device = input.device
                out.active_device = input.active_device
                out.data = result_data
        else:
                backend.maximum(input.data, other_data, out=out.data)
        
        return out


def minimum(input, other, out=None):
        backend = input._backend
        if ENSURE_BACKEND and hasattr(other, '_backend'):
                backend = _backend(input, other)
        
        other_data = other.data if hasattr(other, 'data') else other
        
        if out is None:
                result_data = backend.minimum(input.data, other_data)
                out = Tensor.__new__(Tensor)
                out._requires_grad = input._requires_grad or _requires_grad(other)
                out._grad = None
                out._grad_fn = None  # Can add grad based on which input was smaller
                out._dtype = input._dtype
                out._backend = backend
                out.device = input.device
                out.active_device = input.active_device
                out.data = result_data
        else:
                backend.minimum(input.data, other_data, out=out.data)
        
        return out


def clip(input, min_val, max_val, out=None):
        backend = input._backend
        
        if out is None:
                result_data = backend.clip(input.data, min_val, max_val)
                out = Tensor.__new__(Tensor)
                out._requires_grad = input._requires_grad
                out._grad = None
                out._grad_fn = None  # Can add grad (gradient where not clipped, 0 where clipped)
                out._dtype = input._dtype
                out._backend = backend
                out.device = input.device
                out.active_device = input.active_device
                out.data = result_data
        else:
                backend.clip(input.data, min_val, max_val, out=out.data)
        
        return out


def where(condition, x, y):
        # Get backend from first tensor-like argument
        if hasattr(condition, '_backend'):
                backend = condition._backend
                condition_data = condition.data
        elif hasattr(x, '_backend'):
                backend = x._backend
                condition_data = condition
        else:
                backend = y._backend
                condition_data = condition
        
        x_data = x.data if hasattr(x, 'data') else x
        y_data = y.data if hasattr(y, 'data') else y
        
        result_data = backend.where(condition_data, x_data, y_data)
        
        out = Tensor.__new__(Tensor)
        out._requires_grad = _requires_grad(x) or _requires_grad(y)
        out._grad = None
        out._grad_fn = None
        out._dtype = x._dtype if hasattr(x, '_dtype') else (y._dtype if hasattr(y, '_dtype') else None)
        out._backend = backend
        out.device = x.device if hasattr(x, 'device') else (y.device if hasattr(y, 'device') else None)
        out.active_device = x.active_device if hasattr(x, 'active_device') else (y.active_device if hasattr(y, 'active_device') else 'cpu')
        out.data = result_data

        _attach_grad_fn(
                out,
                backward_where,
                [condition if hasattr(condition, '_requires_grad') else None, x, y],
        )
        return out


def equal(input, other):
        backend = input._backend
        if ENSURE_BACKEND and hasattr(other, '_backend'):
                backend = _backend(input, other)
        
        other_data = other.data if hasattr(other, 'data') else other
        result = backend.equal(input.data, other_data)
        return result


def greater(input, other):
        backend = input._backend
        if ENSURE_BACKEND and hasattr(other, '_backend'):
                backend = _backend(input, other)
        
        other_data = other.data if hasattr(other, 'data') else other
        result = backend.greater(input.data, other_data)
        return result


def less(input, other):
        backend = input._backend
        if ENSURE_BACKEND and hasattr(other, '_backend'):
                backend = _backend(input, other)
        
        other_data = other.data if hasattr(other, 'data') else other
        result = backend.less(input.data, other_data)
        return result



def softmax(input, axis=-1, out=None):
        backend = input._backend
        
        if out is None:
                result_data = backend.softmax(input.data, axis=axis)
                out = Tensor.__new__(Tensor)
                out._requires_grad = input._requires_grad
                out._grad = None
                out._dtype = input._dtype
                out._backend = backend
                out.device = input.device
                out.active_device = input.active_device
                out.data = result_data
                
                if is_grad_enabled() and out._requires_grad:
                        out._grad_fn = Function(
                                backward_softmax,
                                [input],
                                metadata={'axis': axis, 'output': result_data}
                        )
        else:
                backend.softmax(input.data, axis=axis, out=out.data)
        
        return out


def log_softmax(input, axis=-1, out=None):
        backend = input._backend
        
        if out is None:
                result_data = backend.log_softmax(input.data, axis=axis)
                out = Tensor.__new__(Tensor)
                out._requires_grad = input._requires_grad
                out._grad = None
                out._dtype = input._dtype
                out._backend = backend
                out.device = input.device
                out.active_device = input.active_device
                out.data = result_data
                
                if is_grad_enabled() and out._requires_grad:
                        out._grad_fn = Function(backward_log_softmax, [input], metadata={'axis': axis})
        else:
                backend.log_softmax(input.data, axis=axis, out=out.data)
        
        return out


def gelu(input, out=None):
        backend = input._backend
        
        if out is None:
                result_data = backend.gelu(input.data)
                out = Tensor.__new__(Tensor)
                out._requires_grad = input._requires_grad
                out._grad = None
                out._dtype = input._dtype
                out._backend = backend
                out.device = input.device
                out.active_device = input.active_device
                out.data = result_data
                
                if is_grad_enabled() and out._requires_grad:
                        out._grad_fn = Function(backward_gelu, [input], metadata={})
        else:
                backend.gelu(input.data, out=out.data)
        
        return out


def silu(input, out=None):
        backend = input._backend
        
        if out is None:
                result_data = backend.silu(input.data)
                out = Tensor.__new__(Tensor)
                out._requires_grad = input._requires_grad
                out._grad = None
                out._dtype = input._dtype
                out._backend = backend
                out.device = input.device
                out.active_device = input.active_device
                out.data = result_data
                
                if is_grad_enabled() and out._requires_grad:
                        out._grad_fn = Function(backward_silu, [input], metadata={})
        else:
                backend.silu(input.data, out=out.data)
        
        return out


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
        backend = input._backend
        
        if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
        
        weight_data = weight.data if weight is not None else None
        bias_data = bias.data if bias is not None else None
        
        result = backend.layer_norm(
                input.data, normalized_shape, weight_data, bias_data, eps, return_stats=True
        )
        if isinstance(result, tuple):
                result_data, saved_mean, inv_std = result
        else:
                result_data, saved_mean, inv_std = result, None, None
        
        out = Tensor.__new__(Tensor)
        out._requires_grad = input._requires_grad
        out._grad = None
        out._dtype = input._dtype
        out._backend = backend
        out.device = input.device
        out.active_device = input.active_device
        out.data = result_data
        
        if is_grad_enabled() and out._requires_grad:
                out._grad_fn = Function(
                        backward_layer_norm, [input],
                        metadata={
                                'normalized_shape': normalized_shape,
                                'eps': eps,
                                'gamma': weight_data,
                                'mean': saved_mean,
                                'inv_std': inv_std,
                        }
                )
        
        return out


def rms_norm(input, normalized_shape, weight=None, eps=1e-6):
        backend = input._backend
        
        if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
        
        weight_data = weight.data if weight is not None else None
        result = backend.rms_norm(input.data, normalized_shape, weight_data, eps, return_stats=True)
        if isinstance(result, tuple):
                result_data, inv_rms = result
        else:
                result_data, inv_rms = result, None
        
        out = Tensor.__new__(Tensor)
        out._requires_grad = input._requires_grad
        out._grad = None
        out._dtype = input._dtype
        out._backend = backend
        out.device = input.device
        out.active_device = input.active_device
        out.data = result_data
        
        if is_grad_enabled() and out._requires_grad:
                out._grad_fn = Function(
                        backward_rms_norm, [input],
                        metadata={
                                'normalized_shape': normalized_shape,
                                'eps': eps,
                                'gamma': weight_data,
                                'inv_rms': inv_rms,
                        }
                )
        
        return out


def batch_norm(input, running_mean=None, running_var=None, weight=None, bias=None,
                           training=True, momentum=0.1, eps=1e-5):
        backend = input._backend

        weight_data = weight.data if weight is not None else None
        bias_data = bias.data if bias is not None else None

        if len(input.shape) == 2:
                axes = (0,)
        elif len(input.shape) == 4:
                axes = (0, 2, 3)
        else:
                axes = (0,)

        if training:
                mean = backend.mean(input.data, axis=axes, keepdims=True)
                centered = backend.subtract(input.data, mean)
                var = backend.mean(backend.multiply(centered, centered), axis=axes, keepdims=True)
        else:
                shape_4d = (1, -1, 1, 1) if len(input.shape) == 4 else (1, -1)
                mean = backend.reshape(running_mean, shape_4d) if running_mean is not None else backend.mean(input.data, axis=axes, keepdims=True)
                var = backend.reshape(running_var, shape_4d) if running_var is not None else backend.mean(
                        backend.multiply(
                                backend.subtract(input.data, mean),
                                backend.subtract(input.data, mean),
                        ),
                        axis=axes,
                        keepdims=True,
                )

        result_data = backend.batch_norm(
                input.data, running_mean, running_var, weight_data, bias_data, training, momentum, eps
        )
        
        out = Tensor.__new__(Tensor)
        out._requires_grad = input._requires_grad
        out._grad = None
        out._dtype = input._dtype
        out._backend = backend
        out.device = input.device
        out.active_device = input.active_device
        out.data = result_data
        
        if is_grad_enabled() and out._requires_grad and training:
                out._grad_fn = Function(
                        backward_batch_norm, [input],
                        metadata={'eps': eps, 'gamma': weight_data, 'mean': mean, 'var': var, 'axes': axes}
                )

        return out


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
        backend = input._backend
        
        weight_data = weight.data if weight is not None else None
        bias_data = bias.data if bias is not None else None
        
        result_data = backend.group_norm(input.data, num_groups, weight_data, bias_data, eps)
        
        out = Tensor.__new__(Tensor)
        out._requires_grad = input._requires_grad
        out._grad = None
        out._dtype = input._dtype
        out._backend = backend
        out.device = input.device
        out.active_device = input.active_device
        out.data = result_data
        
        if is_grad_enabled() and out._requires_grad:
                out._grad_fn = Function(
                        backward_group_norm, [input],
                        metadata={'num_groups': num_groups, 'eps': eps, 'gamma': weight_data}
                )
        
        return out


def dropout(input, p=0.5, training=True, out=None):
        backend = input._backend

        if out is None:
                if training and p > 0:
                        keep_prob = 1.0 - p
                        rand = backend.rand(input.shape, device=input.device)
                        mask = backend.greater(rand, p)
                        mask = backend.astype(mask, input.data.dtype)
                        scaled_mask = backend.divide(mask, keep_prob)
                        result_data = backend.multiply(input.data, scaled_mask)
                else:
                        mask = None
                        result_data = input.data
                
                out = Tensor.__new__(Tensor)
                out._requires_grad = input._requires_grad
                out._grad = None
                out._dtype = input._dtype
                out._backend = backend
                out.device = input.device
                out.active_device = input.active_device
                out.data = result_data
                
                if is_grad_enabled() and out._requires_grad:
                        out._grad_fn = Function(
                                backward_dropout, [input],
                                metadata={'mask': mask, 'p': p, 'training': training}
                        )
        else:
                if training and p > 0:
                        keep_prob = 1.0 - p
                        rand = backend.rand(input.shape, device=input.device)
                        mask = backend.greater(rand, p)
                        mask = backend.astype(mask, input.data.dtype)
                        scaled_mask = backend.divide(mask, keep_prob)
                        out.data = backend.multiply(input.data, scaled_mask)
                else:
                        out.data = input.data

        return out


def embedding(weight, indices, padding_idx=None):
        backend = weight._backend

        raw_indices = indices.data if hasattr(indices, "data") else indices
        indices_array = backend.asarray(raw_indices)
        if getattr(indices_array.dtype, "kind", "f") not in ("i", "u"):
                indices_array = backend.astype(indices_array, backend.int64)

        result_data = backend.embedding_lookup(weight.data, indices_array, padding_idx)

        out = Tensor.__new__(Tensor)
        out._requires_grad = weight._requires_grad
        out._grad = None
        out._dtype = weight._dtype
        out._backend = backend
        out.device = weight.device
        out.active_device = weight.active_device
        out.data = result_data

        if is_grad_enabled() and out._requires_grad:
                out._grad_fn = Function(
                        backward_embedding,
                        [weight],
                        metadata={"indices": indices_array, "num_embeddings": weight.shape[0]},
                )

        return out


def permute(input, dims):
        backend = input._backend
        result_data = backend.transpose(input.data, dims)
        
        out = Tensor.__new__(Tensor)
        out._requires_grad = input._requires_grad
        out._grad = None
        out._dtype = input._dtype
        out._backend = backend
        out.device = input.device
        out.active_device = input.active_device
        out.data = result_data
        
        if is_grad_enabled() and out._requires_grad:
                out._grad_fn = Function(backward_permute, [input], metadata={'dims': dims})
        
        return out


def unsqueeze(input, dim):
        backend = input._backend
        result_data = backend.unsqueeze(input.data, dim)
        
        out = Tensor.__new__(Tensor)
        out._requires_grad = input._requires_grad
        out._grad = None
        out._dtype = input._dtype
        out._backend = backend
        out.device = input.device
        out.active_device = input.active_device
        out.data = result_data
        
        if is_grad_enabled() and out._requires_grad:
                out._grad_fn = Function(backward_unsqueeze, [input], metadata={'dim': dim})
        
        return out


def split(input, split_size_or_sections, dim=0):
        backend = input._backend
        data = input.data

        if data.ndim == 0:
                raise ValueError("split expects at least a 1D tensor")

        axis = dim % data.ndim

        if isinstance(split_size_or_sections, int):
                if split_size_or_sections <= 0:
                        raise ValueError("split_size must be positive")
                total = data.shape[axis]
                if total == 0:
                        raise ValueError("cannot split tensor with zero size along the given dimension")
                full_chunks, remainder = divmod(total, split_size_or_sections)
                sizes = [split_size_or_sections] * full_chunks
                if remainder:
                        sizes.append(remainder)
        else:
                sizes = list(split_size_or_sections)
                if not sizes:
                        raise ValueError("split expects a non-empty list of sections")
                if any(size <= 0 for size in sizes):
                        raise ValueError("section sizes must be positive")
                if builtins.sum(sizes) != data.shape[axis]:
                        raise ValueError("sum of split sizes must match tensor dimension")

        slices = [slice(None)] * data.ndim
        start = 0
        chunks_data = []
        for size in sizes:
                end = start + size
                slices[axis] = slice(start, end)
                chunk_view = data[tuple(slices)]
                chunks_data.append(chunk_view)
                start = end

        chunks = []
        shared_state = {
                'axis': axis,
                'sizes': sizes,
                'num_chunks': len(chunks_data),
                'buffer': None,
                'completed': 0,
        }
        for chunk_data in chunks_data:
                chunk = Tensor.__new__(Tensor)
                chunk._requires_grad = input._requires_grad
                chunk._grad = None
                chunk._dtype = input._dtype
                chunk._backend = backend
                chunk.device = input.device
                chunk.active_device = input.active_device
                chunk.data = chunk_data
                chunks.append(chunk)

        if is_grad_enabled() and input._requires_grad:
                for idx, chunk in enumerate(chunks):
                        chunk._grad_fn = Function(
                                backward_split,
                                [input],
                                metadata={
                                        'axis': axis,
                                        'sizes': sizes,
                                        'index': idx,
                                        'shared_state': shared_state,
                                },
                        )

        return chunks


def gather(input, dim, index):
        backend = input._backend
        index_data = index.data if hasattr(index, 'data') else index
        result_data = backend.gather(input.data, dim, index_data)
        
        out = Tensor.__new__(Tensor)
        out._requires_grad = input._requires_grad
        out._grad = None
        out._dtype = input._dtype
        out._backend = backend
        out.device = input.device
        out.active_device = input.active_device
        out.data = result_data

        # Note: gather backward requires scatter, simplified
        return out


def getitem(input, index):
        backend = input._backend
        normalized_index = _normalize_index(index)
        result_data = input.data[normalized_index]

        out = Tensor.__new__(Tensor)
        out._requires_grad = input._requires_grad
        out._grad = None
        out._dtype = input._dtype
        out._backend = backend
        out.device = input.device
        out.active_device = input.active_device
        out.data = result_data

        if is_grad_enabled() and input._requires_grad:
                out._grad_fn = Function(
                        backward_getitem,
                        [input],
                        metadata={"index_spec": normalized_index},
                )

        return out


def masked_fill(input, mask, value):
        backend = input._backend
        mask_data = mask.data if hasattr(mask, 'data') else mask
        result_data = backend.masked_fill(input.data, mask_data, value)
        
        out = Tensor.__new__(Tensor)
        out._requires_grad = input._requires_grad
        out._grad = None
        out._dtype = input._dtype
        out._backend = backend
        out.device = input.device
        out.active_device = input.active_device
        out.data = result_data
        
        if is_grad_enabled() and out._requires_grad:
                out._grad_fn = Function(
                        backward_where, [None, input, None],
                        metadata={'condition': backend.logical_not(mask_data)}
                )
        
        return out


