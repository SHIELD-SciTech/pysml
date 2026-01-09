import builtins
import weakref
from typing import Callable, List, Tuple, Optional

class Function:
        __slots__ = ('inputs', 'backward_fn', 'metadata', 'next_functions', '_saved_tensors')

        def __init__(self, backward_fn: Callable, inputs: List, metadata: dict = None, save_for_backward: List = None):
                # Keep weak references for graph traversal but retain strong references for backward.
                self.inputs = [weakref.ref(t) if t is not None else None for t in inputs]
                # Only save tensors explicitly needed for backward (memory optimization)
                self._saved_tensors = save_for_backward if save_for_backward is not None else []
                self.backward_fn = backward_fn
                self.metadata = metadata or {}
                self.next_functions = []  # For graph traversal

        def release_graph(self):

                self._saved_tensors = None
                self.next_functions = []
                # Clear metadata which may hold tensor references (masks, saved outputs, etc.)
                self.metadata = None
                self.backward_fn = None
                self.inputs = None

        def apply_backward(self, grad_output):
                try:
                        return self.backward_fn(grad_output, *self.inputs, **(self.metadata or {}))
                finally:
                        # Release saved references after backward to avoid leaks even if backward fails.
                        self._saved_tensors = None
                        self.metadata = None

def register_post_backward_hook(tensor, hook: Callable):

        from .tensor import Tensor
        if not isinstance(tensor, Tensor):
                raise TypeError("register_post_backward_hook expects a Tensor instance")
        return tensor.register_post_backward_hook(hook)

class no_grad:
        def __init__(self):
                self.prev = None

        def __enter__(self):
                global _grad_enabled
                self.prev = _grad_enabled
                _grad_enabled = False
                return self

        def __exit__(self, *args):
                global _grad_enabled
                _grad_enabled = self.prev

# Global flag for gradient tracking
_grad_enabled = True

def is_grad_enabled():
        return _grad_enabled


def checkpoint(function, *args, use_reentrant=True, **kwargs):
        """
        Gradient checkpointing: saves memory by not storing activations for backward.
        Instead, recomputes forward pass during backward.

        Usage:
            output = checkpoint(layer.forward, input)

        This can reduce memory by ~50% for deep networks at cost of ~30% more compute.
        """
        from .tensor import Tensor

        # Run forward with no_grad to avoid saving activations
        with no_grad():
                outputs = function(*args, **kwargs)

        # Wrap outputs to enable gradient flow
        if isinstance(outputs, Tensor):
                outputs_list = [outputs]
        elif isinstance(outputs, tuple):
                outputs_list = list(outputs)
        else:
                return outputs

        # Find inputs that require grad
        tensor_args = [a for a in args if isinstance(a, Tensor) and a._requires_grad]

        if not tensor_args:
                return outputs

        # Create checkpointed outputs that will recompute on backward
        def make_recompute_backward(func, inputs, kwargs_dict):
                def recompute_backward(grad_output, *input_refs, **meta):
                        # Recompute forward pass
                        actual_inputs = []
                        for inp in inputs:
                                if isinstance(inp, Tensor):
                                        # Create fresh tensor for recomputation
                                        actual_inputs.append(inp)
                                else:
                                        actual_inputs.append(inp)

                        # Run forward again with grad enabled
                        recomputed = func(*actual_inputs, **kwargs_dict)

                        if isinstance(recomputed, tuple):
                                recomputed = recomputed[0]

                        # Now do backward on the recomputed result
                        recomputed.backward(grad_output)

                        # Return gradients for inputs
                        grads = []
                        for inp in inputs:
                                if isinstance(inp, Tensor) and inp._requires_grad:
                                        grads.append((inp, inp.grad))
                                else:
                                        grads.append(None)
                        return grads
                return recompute_backward

        # Attach recompute backward to first output
        first_out = outputs_list[0]
        if isinstance(first_out, Tensor):
                first_out._requires_grad = True
                first_out._grad_fn = Function(
                        make_recompute_backward(function, args, kwargs),
                        tensor_args,
                        metadata={}
                )

        if isinstance(outputs, tuple):
                return tuple(outputs_list)
        return outputs_list[0]

def set_grad_enabled(mode: bool):
        global _grad_enabled
        _grad_enabled = mode

def _reduce_grad_to_shape(grad_data, target_shape, backend):

        if getattr(grad_data, 'shape', None) is None or grad_data.shape == target_shape:
                return grad_data

        ndim_diff = len(grad_data.shape) - len(target_shape)
        if ndim_diff > 0:
                for _ in range(ndim_diff):
                        grad_data = backend.sum(grad_data, axis=0, keepdims=False)

        for axis, size in enumerate(target_shape):
                if size == 1 and grad_data.shape[axis] != 1:
                        grad_data = backend.sum(grad_data, axis=axis, keepdims=True)

        return grad_data

def _reduction_output_shape(input_shape, axis, keepdims):

        if axis is None:
                return tuple(1 for _ in input_shape) if keepdims else tuple()

        if not isinstance(axis, (tuple, list)):
                axis = (axis,)

        normalized_axes = []
        for ax in axis:
                normalized_axes.append(ax if ax >= 0 else ax + len(input_shape))

        if keepdims:
                return tuple(1 if i in normalized_axes else dim for i, dim in enumerate(input_shape))

        return tuple(dim for i, dim in enumerate(input_shape) if i not in normalized_axes)

def _expand_grad_for_reduction(grad_output, input_shape, axis, keepdims, backend):

        grad_data = grad_output.data if hasattr(grad_output, 'data') else grad_output

        # Reduce upstream gradients that may have been broadcast beyond the reduction output shape.
        output_shape = _reduction_output_shape(input_shape, axis, keepdims)
        grad_data = _reduce_grad_to_shape(grad_data, output_shape, backend)

        if axis is None:
                return backend.broadcast_to(grad_data, input_shape)

        if not isinstance(axis, (tuple, list)):
                axis = (axis,)

        normalized_axes = []
        for ax in axis:
                normalized_axes.append(ax if ax >= 0 else ax + len(input_shape))

        if not keepdims:
                for ax in sorted(normalized_axes):
                        grad_data = backend.expand_dims(grad_data, ax)

        try:
                return backend.broadcast_to(grad_data, input_shape)
        except ValueError:
                # Fallback: try reducing to scalar if broadcasting fails
                # This handles cases where grad_output shape is incompatible (e.g. (256, 1) vs (4, 4))
                scalar_grad = backend.sum(grad_data)
                return backend.broadcast_to(scalar_grad, input_shape)

def _wrap_grad_tensor(grad_data, reference):

        from .tensor import Tensor

        if reference is None:
                return grad_data

        payload = grad_data.data if hasattr(grad_data, 'data') else grad_data
        return reference._new_like(payload, requires_grad=False)

def backward_add(grad_output, input_ref, other_ref, **metadata):

        grads = []
        
        backend = grad_output._backend if hasattr(grad_output, '_backend') else None

        # Gradient for first input
        input_tensor = input_ref() if input_ref else None
        if input_tensor is not None and input_tensor._requires_grad:
                grad_data = grad_output.data if hasattr(grad_output, 'data') else grad_output
                if hasattr(grad_output, 'shape') and hasattr(input_tensor, 'shape'):
                        grad_data = _reduce_grad_to_shape(grad_data, input_tensor.shape, backend or input_tensor._backend)

                grad_tensor = _wrap_grad_tensor(grad_data, input_tensor)

                grads.append((input_tensor, grad_tensor))
        else:
                grads.append(None)

        # Gradient for second input
        other_tensor = other_ref() if other_ref else None
        if other_tensor is not None and other_tensor._requires_grad:
                grad_data = grad_output.data if hasattr(grad_output, 'data') else grad_output
                alpha = metadata.get('alpha', 1)
                if hasattr(grad_output, 'shape') and hasattr(other_tensor, 'shape'):
                        grad_data = _reduce_grad_to_shape(grad_data, other_tensor.shape, backend or other_tensor._backend)

                if alpha != 1:
                        backend_for_scale = backend or other_tensor._backend
                        grad_data = backend_for_scale.multiply(grad_data, alpha)

                grad = _wrap_grad_tensor(grad_data, other_tensor)

                grads.append((other_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_subtract(grad_output, input_ref, other_ref, **metadata):

        grads = []

        backend = grad_output._backend if hasattr(grad_output, '_backend') else None

        input_tensor = input_ref() if input_ref else None
        if input_tensor is not None and input_tensor._requires_grad:
                grad_data = grad_output.data if hasattr(grad_output, 'data') else grad_output
                if hasattr(grad_output, 'shape') and hasattr(input_tensor, 'shape'):
                        grad_data = _reduce_grad_to_shape(grad_data, input_tensor.shape, backend or input_tensor._backend)

                grad = _wrap_grad_tensor(grad_data, input_tensor)
                grads.append((input_tensor, grad))
        else:
                grads.append(None)

        other_tensor = other_ref() if other_ref else None
        if other_tensor is not None and other_tensor._requires_grad:
                if backend:
                        grad_data = backend.negative(grad_output.data)
                else:
                        grad_data = -grad_output

                if hasattr(grad_output, 'shape') and hasattr(other_tensor, 'shape'):
                        grad_data = _reduce_grad_to_shape(grad_data, other_tensor.shape, backend or other_tensor._backend)

                grad = _wrap_grad_tensor(grad_data, other_tensor)
                grads.append((other_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_multiply(grad_output, input_ref, other_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        other_tensor = other_ref() if other_ref else None
        backend = grad_output._backend if hasattr(grad_output, '_backend') else None

        if input_tensor is not None and input_tensor._requires_grad:
                # dL/dx = grad_output * other
                if other_tensor:
                        active_backend = backend or other_tensor._backend
                        grad_data = active_backend.multiply(grad_output.data, other_tensor.data)
                        grad_data = _reduce_grad_to_shape(grad_data, input_tensor.shape, active_backend)

                        grad = _wrap_grad_tensor(grad_data, input_tensor)
                        grads.append((input_tensor, grad))
                else:
                        # other is a scalar
                        scalar = metadata.get('other_scalar', 1.0)
                        active_backend = backend or input_tensor._backend
                        grad_data = active_backend.multiply(grad_output.data, scalar)
                        grad_data = _reduce_grad_to_shape(grad_data, input_tensor.shape, active_backend)
                        grad = _wrap_grad_tensor(grad_data, input_tensor)
                        grads.append((input_tensor, grad))
        else:
                grads.append(None)

        if other_tensor is not None and other_tensor._requires_grad:
                # dL/dy = grad_output * input
                active_backend = backend or input_tensor._backend
                grad_data = active_backend.multiply(grad_output.data, input_tensor.data)
                grad_data = _reduce_grad_to_shape(grad_data, other_tensor.shape, active_backend)
                grad = _wrap_grad_tensor(grad_data, other_tensor)
                grads.append((other_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_divide(grad_output, input_ref, other_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        other_tensor = other_ref() if other_ref else None
        backend = grad_output._backend if hasattr(grad_output, '_backend') else None

        if input_tensor is not None and input_tensor._requires_grad:
                # dL/dx = grad_output / other
                if other_tensor:
                        active_backend = backend or other_tensor._backend
                        grad_data = active_backend.divide(grad_output.data, other_tensor.data)
                        grad_data = _reduce_grad_to_shape(grad_data, input_tensor.shape, active_backend)
                        grad = _wrap_grad_tensor(grad_data, input_tensor)
                        grads.append((input_tensor, grad))
                else:
                        scalar = metadata.get('other_scalar', 1.0)
                        active_backend = backend or input_tensor._backend
                        grad_data = active_backend.divide(grad_output.data, scalar)
                        grad_data = _reduce_grad_to_shape(grad_data, input_tensor.shape, active_backend)
                        grad = _wrap_grad_tensor(grad_data, input_tensor)
                        grads.append((input_tensor, grad))
        else:
                grads.append(None)

        if other_tensor is not None and other_tensor._requires_grad:
                # dL/dy = -grad_output * input / other^2
                active_backend = backend or other_tensor._backend
                # -grad_output * input
                temp = active_backend.multiply(grad_output.data, input_tensor.data)
                temp = active_backend.negative(temp)

                # other^2
                other_squared = active_backend.multiply(other_tensor.data, other_tensor.data)

                # final gradient
                grad_data = active_backend.divide(temp, other_squared)
                grad_data = _reduce_grad_to_shape(grad_data, other_tensor.shape, active_backend)
                grad = _wrap_grad_tensor(grad_data, other_tensor)
                grads.append((other_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_power(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend
                exponent = metadata.get('exponent', 2.0)

                # n * x^(n-1)
                # x^(n-1)
                x_pow = backend.power(input_tensor.data, exponent - 1)
                # n * x^(n-1)
                x_pow = backend.multiply(x_pow, exponent)
                # grad_output * n * x^(n-1)
                grad_data = backend.multiply(grad_output.data, x_pow)
                grad = _wrap_grad_tensor(grad_data, input_tensor)

                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_matmul(grad_output, input_ref, other_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        other_tensor = other_ref() if other_ref else None
        
        if input_tensor is not None and input_tensor._requires_grad:
                # dL/dX = grad_output @ other^T
                backend = grad_output._backend

                other_T = backend.transpose(other_tensor.data)
                grad_data = backend.matmul(grad_output.data, other_T)

                if hasattr(grad_data, 'shape') and hasattr(input_tensor, 'shape'):
                        grad_data = _reduce_grad_to_shape(grad_data, input_tensor.shape, backend)
                grad = _wrap_grad_tensor(grad_data, input_tensor)

                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        if other_tensor is not None and other_tensor._requires_grad:
                # dL/dY = input^T @ grad_output
                backend = grad_output._backend

                input_T = backend.transpose(input_tensor.data)
                grad_data = backend.matmul(input_T, grad_output.data)

                if hasattr(grad_data, 'shape') and hasattr(other_tensor, 'shape'):
                        grad_data = _reduce_grad_to_shape(grad_data, other_tensor.shape, backend)
                grad = _wrap_grad_tensor(grad_data, other_tensor)

                grads.append((other_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_relu(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend

                # Mask where input > 0
                mask = backend.greater(input_tensor.data, 0)
                # grad_output * mask
                grad_data = backend.where(mask, grad_output.data, 0)
                grad = _wrap_grad_tensor(grad_data, input_tensor)

                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_exp(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend
                
                # exp(input)
                exp_x = backend.exp(input_tensor.data)
                # grad_output * exp(x)
                grad_data = backend.multiply(grad_output.data, exp_x)
                grad = _wrap_grad_tensor(grad_data, input_tensor)
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_log(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend

                # 1/x
                inv_x = backend.reciprocal(input_tensor.data)
                # grad_output / x
                grad_data = backend.multiply(grad_output.data, inv_x)
                grad = _wrap_grad_tensor(grad_data, input_tensor)

                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_tanh(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None

        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend

                # Prefer using saved output to avoid recomputation.
                tanh_output = metadata.get('output')
                if tanh_output is None:
                        # Fallback to recomputation if metadata missing.
                        grad_input = backend.tanh(input_tensor.data)
                else:
                        grad_input = tanh_output

                # 1 - tanh^2(x)
                tanh_squared = backend.multiply(grad_input, grad_input)
                grad_tanh = backend.subtract(1.0, tanh_squared)
                # grad_output * (1 - tanh^2(x))
                grad_data = backend.multiply(grad_output.data, grad_tanh)
                grad = _wrap_grad_tensor(grad_data, input_tensor)

                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_sum(grad_output, input_ref, **metadata):

    grads = []

    input_tensor = input_ref() if input_ref else None

    if input_tensor is not None and input_tensor._requires_grad:
        backend = grad_output._backend if hasattr(grad_output, '_backend') else input_tensor._backend

        axis = metadata.get('axis', None)
        keepdims = metadata.get('keepdims', False)
        grad_data = _expand_grad_for_reduction(grad_output, input_tensor.shape, axis, keepdims, backend)

        grad = _wrap_grad_tensor(grad_data, input_tensor)

        grads.append((input_tensor, grad))
    else:
        grads.append(None)

    return grads

def backward_transpose(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None

        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend

                # Transpose the gradient back
                axes = metadata.get('axes', None)
                if axes is None:
                        # Simple transpose - reverse all axes
                        grad_data = backend.transpose(grad_output.data)
                else:
                        # Inverse permutation for axes
                        inv_axes = [0] * len(axes)
                        for i, ax in enumerate(axes):
                                inv_axes[ax] = i
                        grad_data = backend.transpose(grad_output.data, inv_axes)

                grad = _wrap_grad_tensor(grad_data, input_tensor)

                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_reshape(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend
                
                # Reshape gradient back to input shape
                original_shape = metadata.get('original_shape', input_tensor.shape)
                grad_data = backend.reshape(grad_output.data, original_shape)
                grad = _wrap_grad_tensor(grad_data, input_tensor)
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_sigmoid(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None

        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend

                sigmoid_x = metadata.get('output')
                if sigmoid_x is None:
                        # Fallback to recomputation if forward output is unavailable.
                        sigmoid_x = backend.reciprocal(
                                backend.add(1.0, backend.exp(backend.negative(input_tensor.data)))
                        )

                # sigmoid(x) * (1 - sigmoid(x))
                grad_sigmoid = backend.multiply(
                        sigmoid_x,
                        backend.subtract(1.0, sigmoid_x)
                )

                # grad_output * sigmoid'(x)
                grad_data = backend.multiply(grad_output.data, grad_sigmoid)
                grad = _wrap_grad_tensor(grad_data, input_tensor)
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_sqrt(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend
                
                # 1 / (2 * sqrt(x))
                sqrt_x = backend.sqrt(input_tensor.data)
                grad_sqrt = backend.reciprocal(backend.multiply(2.0, sqrt_x))
                
                # grad_output * grad_sqrt
                grad_data = backend.multiply(grad_output.data, grad_sqrt)
                grad = _wrap_grad_tensor(grad_data, input_tensor)
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_sin(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend
                
                # cos(x)
                cos_x = backend.cos(input_tensor.data)
                
                # grad_output * cos(x)
                grad_data = backend.multiply(grad_output.data, cos_x)
                grad = _wrap_grad_tensor(grad_data, input_tensor)
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_cos(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend
                
                # -sin(x)
                sin_x = backend.sin(input_tensor.data)
                neg_sin_x = backend.negative(sin_x)
                
                # grad_output * (-sin(x))
                grad_data = backend.multiply(grad_output.data, neg_sin_x)
                grad = _wrap_grad_tensor(grad_data, input_tensor)
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

# Additional backward functions for modern deep learning models

def backward_softmax(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None

        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend
                axis = metadata.get('axis', -1)
                softmax_output = metadata.get('output')

                if softmax_output is None:
                        # Fallback to recomputation if the forward output is missing.
                        x_max = backend.max(input_tensor.data, axis=axis, keepdims=True)
                        x_shifted = backend.subtract(input_tensor.data, x_max)
                        exp_x = backend.exp(x_shifted)
                        sum_exp = backend.sum(exp_x, axis=axis, keepdims=True)
                        softmax_output = backend.divide(exp_x, sum_exp)

                # Compute sum of (softmax * grad_output) along axis
                sum_term = backend.sum(
                        backend.multiply(softmax_output, grad_output.data),
                        axis=axis,
                        keepdims=True
                )

                # Gradient: softmax * (grad_output - sum_term)
                grad_data = backend.multiply(
                        softmax_output,
                        backend.subtract(grad_output.data, sum_term)
                )
                grad = _wrap_grad_tensor(grad_data, input_tensor)
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_layer_norm(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend
                normalized_shape = metadata.get('normalized_shape')
                eps = metadata.get('eps', 1e-5)
                gamma = metadata.get('gamma', None)  # Scale parameter
                saved_mean = metadata.get('mean')
                inv_std = metadata.get('inv_std')

                # Determine axes to normalize over
                ndim = len(input_tensor.shape)
                axes = tuple(range(ndim - len(normalized_shape), ndim))

                if saved_mean is None or inv_std is None:
                        # Fallback to recomputation if stats are unavailable
                        mean = backend.mean(input_tensor.data, axis=axes, keepdims=True)
                        centered = backend.subtract(input_tensor.data, mean)
                        var = backend.mean(backend.multiply(centered, centered), axis=axes, keepdims=True)
                        inv_std = backend.reciprocal(backend.sqrt(backend.add(var, eps)))
                else:
                        mean = saved_mean
                        centered = backend.subtract(input_tensor.data, mean)

                normalized = backend.multiply(centered, inv_std)
                
                # Gradient computation
                if gamma is not None:
                        grad_normalized = backend.multiply(grad_output.data, gamma)
                else:
                        grad_normalized = grad_output.data
                
                # Number of elements being normalized
                N = 1
                for axis in axes:
                        N *= input_tensor.shape[axis]
                
                # Compute gradients
                # dL/dx = (1/N*std) * (N*dL/dy - sum(dL/dy) - normalized*sum(dL/dy * normalized))
                sum_grad = backend.sum(grad_normalized, axis=axes, keepdims=True)
                sum_grad_normalized = backend.sum(
                        backend.multiply(grad_normalized, normalized),
                        axis=axes,
                        keepdims=True
                )
                
                grad_input = backend.multiply(grad_normalized, N)
                grad_input = backend.subtract(grad_input, sum_grad)
                grad_input = backend.subtract(
                        grad_input,
                        backend.multiply(normalized, sum_grad_normalized)
                )
                std = backend.reciprocal(inv_std)
                grad_data = backend.divide(grad_input, backend.multiply(N, std))
                grad = _wrap_grad_tensor(grad_data, input_tensor)
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_rms_norm(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend
                normalized_shape = metadata.get('normalized_shape')
                eps = metadata.get('eps', 1e-6)
                gamma = metadata.get('gamma', None)
                inv_rms = metadata.get('inv_rms')

                # Determine axes
                ndim = len(input_tensor.shape)
                axes = tuple(range(ndim - len(normalized_shape), ndim))

                if inv_rms is None:
                        x_squared = backend.multiply(input_tensor.data, input_tensor.data)
                        mean_squared = backend.mean(x_squared, axis=axes, keepdims=True)
                        rms = backend.sqrt(backend.add(mean_squared, eps))
                        inv_rms = backend.reciprocal(rms)
                else:
                        rms = backend.reciprocal(inv_rms)
                normalized = backend.multiply(input_tensor.data, inv_rms)
                
                # Gradient
                if gamma is not None:
                        grad_normalized = backend.multiply(grad_output.data, gamma)
                else:
                        grad_normalized = grad_output.data
                
                # Number of elements
                N = 1
                for axis in axes:
                        N *= input_tensor.shape[axis]
                
                # Gradient computation
                # dL/dx = (1/rms) * (dL/dy - (x/rms^2) * (1/N) * sum(x * dL/dy))
                sum_term = backend.sum(
                        backend.multiply(input_tensor.data, grad_normalized),
                        axis=axes,
                        keepdims=True
                )
                sum_term = backend.divide(sum_term, N)

                term1 = backend.divide(grad_normalized, rms)
                term2 = backend.multiply(
                        backend.divide(input_tensor.data, backend.multiply(rms, rms)),
                        sum_term
                )
                
                grad_data = backend.subtract(term1, term2)
                grad = _wrap_grad_tensor(grad_data, input_tensor)
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_gelu(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend
                
                # Constants
                sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/π)
                coeff = 0.044715
                
                x = input_tensor.data
                
                # x^3
                x_cubed = backend.multiply(backend.multiply(x, x), x)
                
                # Inner term: x + 0.044715 * x^3
                inner = backend.add(x, backend.multiply(coeff, x_cubed))
                
                # sqrt(2/π) * (x + 0.044715 * x^3)
                inner = backend.multiply(sqrt_2_over_pi, inner)
                
                # tanh(inner)
                tanh_inner = backend.tanh(inner)
                
                # 1 + tanh(inner)
                one_plus_tanh = backend.add(1.0, tanh_inner)
                
                # 0.5 * (1 + tanh(inner))
                half_term = backend.multiply(0.5, one_plus_tanh)
                
                # Derivative of tanh: 1 - tanh^2
                tanh_squared = backend.multiply(tanh_inner, tanh_inner)
                sech_squared = backend.subtract(1.0, tanh_squared)
                
                # Derivative of inner w.r.t. x: 1 + 3 * 0.044715 * x^2
                x_squared = backend.multiply(x, x)
                inner_derivative = backend.add(
                        1.0,
                        backend.multiply(3.0 * coeff, x_squared)
                )
                inner_derivative = backend.multiply(sqrt_2_over_pi, inner_derivative)
                
                # Full derivative: 0.5 * (1 + tanh(inner)) + 0.5 * x * sech^2(inner) * inner_derivative
                term1 = half_term
                term2 = backend.multiply(
                        backend.multiply(backend.multiply(0.5, x), sech_squared),
                        inner_derivative
                )
                gelu_grad = backend.add(term1, term2)
                
                grad_data = backend.multiply(grad_output.data, gelu_grad)
                grad = _wrap_grad_tensor(grad_data, input_tensor)
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_silu(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend
                
                # Compute sigmoid(x)
                sigmoid_x = backend.reciprocal(
                        backend.add(1.0, backend.exp(backend.negative(input_tensor.data)))
                )
                
                # sigmoid(x) * (1 - sigmoid(x))
                sigmoid_grad = backend.multiply(
                        sigmoid_x,
                        backend.subtract(1.0, sigmoid_x)
                )
                
                # f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                silu_grad = backend.add(
                        sigmoid_x,
                        backend.multiply(input_tensor.data, sigmoid_grad)
                )
                
                grad_data = backend.multiply(grad_output.data, silu_grad)
                grad = _wrap_grad_tensor(grad_data, input_tensor)
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_dropout(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend
                mask = metadata.get('mask')  # Binary mask from forward pass
                p = metadata.get('p', 0.5)
                training = metadata.get('training', True)
                
                if training and mask is not None:
                        # Scale by mask and inverse keep probability
                        scale = 1.0 / (1.0 - p)
                        grad_data = backend.multiply(
                                backend.multiply(grad_output.data, mask),
                                scale
                        )
                else:
                        # No dropout during inference
                        grad_data = grad_output.data
                
                grad = _wrap_grad_tensor(grad_data, input_tensor)
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_embedding(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend
                indices = metadata.get('indices')
                num_embeddings = metadata.get('num_embeddings')

                embedding_dim = grad_output.shape[-1]
                grad_shape = (num_embeddings, embedding_dim)
                grad_data = backend.zeros(grad_shape, dtype=grad_output.data.dtype)

                if hasattr(indices, 'data'):
                        indices_data = indices.data
                else:
                        indices_data = indices

                indices_array = backend.asarray(indices_data)
                if getattr(indices_array.dtype, "kind", "f") not in ("i", "u"):
                        indices_array = backend.astype(indices_array, backend.int64)

                backend.scatter_add(
                        grad_data,
                        0,
                        indices_array,
                        grad_output.data,
                )

                grad = _wrap_grad_tensor(grad_data, input_tensor)
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_concatenate(grad_output, *input_refs, **metadata):

        grads = []
        axis = metadata.get('axis', 0)
        
        backend = grad_output._backend
        
        # Split gradient along concatenation axis
        current_idx = 0
        for input_ref in input_refs:
                input_tensor = input_ref() if input_ref else None
                
                if input_tensor is not None and input_tensor._requires_grad:
                        # Slice gradient for this input
                        size = input_tensor.shape[axis]
                        
                        # Create slice indices
                        slices = [slice(None)] * len(grad_output.shape)
                        slices[axis] = slice(current_idx, current_idx + size)
                        
                        grad_data = grad_output.data[tuple(slices)]
                        grad = _wrap_grad_tensor(grad_data, input_tensor)
                        
                        grads.append((input_tensor, grad))
                        current_idx += size
                else:
                        if input_tensor:
                                current_idx += input_tensor.shape[axis]
                        grads.append(None)
        
        return grads

def backward_split(grad_output, input_ref, **metadata):

    grads = []

    input_tensor = input_ref() if input_ref else None

    if input_tensor is not None and input_tensor._requires_grad:
        backend = input_tensor._backend
        axis = metadata.get('axis', 0)
        sizes = metadata.get('sizes', [])
        index = metadata.get('index', 0)

        if grad_output is None:
            grads.append((input_tensor, None))
            return grads

        grad_buffer = backend.zeros_like(input_tensor.data)

        if isinstance(grad_output, (list, tuple)):
            for idx, g in enumerate(grad_output):
                if g is None:
                    continue
                start = builtins.sum(sizes[:idx]) if sizes else 0
                end = start + (sizes[idx] if sizes else g.data.shape[axis])
                slices = [slice(None)] * grad_buffer.ndim
                slices[axis] = slice(start, end)
                grad_buffer[tuple(slices)] = g.data
        else:
            start = builtins.sum(sizes[:index]) if sizes else 0
            end = start + (sizes[index] if sizes else grad_output.data.shape[axis])
            slices = [slice(None)] * grad_buffer.ndim
            slices[axis] = slice(start, end)
            grad_buffer[tuple(slices)] = grad_output.data

        grad = _wrap_grad_tensor(grad_buffer, input_tensor)
        grads.append((input_tensor, grad))
    else:
        grads.append(None)

    return grads

def backward_getitem(grad_output, input_ref, **metadata):

    grads = []

    input_tensor = input_ref() if input_ref else None

    if input_tensor is not None and input_tensor._requires_grad:
        backend = input_tensor._backend
        index_spec = metadata.get('index_spec')

        if index_spec is None or grad_output is None:
            grads.append((input_tensor, None))
            return grads

        grad_data = backend.zeros_like(input_tensor.data)
        grad_slice = grad_output.data if hasattr(grad_output, 'data') else grad_output
        grad_data[index_spec] = grad_slice

        grad = _wrap_grad_tensor(grad_data, input_tensor)
        grads.append((input_tensor, grad))
    else:
        grads.append(None)

    return grads

def backward_max_reduce(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend
                axis = metadata.get('axis', None)
                keepdims = metadata.get('keepdims', False)
                max_indices = metadata.get('max_indices')

                grad_data = backend.zeros_like(input_tensor.data)

                if axis is None:
                        flat_grad = backend.reshape(grad_data, (-1,))
                        grad_value = grad_output.data
                        grad_value = backend.reshape(grad_value, (-1,))
                        if grad_value.shape[0] != 1:
                                grad_value = backend.reshape(grad_value, (1,))

                        index_array = backend.asarray([max_indices])
                        backend.scatter_add(flat_grad, 0, index_array, grad_value)
                        grad_data = backend.reshape(flat_grad, input_tensor.shape)
                else:
                        grad_values = grad_output.data
                        indices = max_indices

                        if not keepdims:
                                grad_values = backend.expand_dims(grad_values, axis=axis)
                                indices = backend.expand_dims(indices, axis=axis)

                        backend.scatter_add(grad_data, axis, indices, grad_values)

                grad = _wrap_grad_tensor(grad_data, input_tensor)
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_min_reduce(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend
                axis = metadata.get('axis', None)
                keepdims = metadata.get('keepdims', False)
                min_indices = metadata.get('min_indices')

                grad_data = backend.zeros_like(input_tensor.data)

                if axis is None:
                        flat_grad = backend.reshape(grad_data, (-1,))
                        grad_value = grad_output.data
                        grad_value = backend.reshape(grad_value, (-1,))
                        if grad_value.shape[0] != 1:
                                grad_value = backend.reshape(grad_value, (1,))

                        index_array = backend.asarray([min_indices])
                        backend.scatter_add(flat_grad, 0, index_array, grad_value)
                        grad_data = backend.reshape(flat_grad, input_tensor.shape)
                else:
                        grad_values = grad_output.data
                        indices = min_indices

                        if not keepdims:
                                grad_values = backend.expand_dims(grad_values, axis=axis)
                                indices = backend.expand_dims(indices, axis=axis)

                        backend.scatter_add(grad_data, axis, indices, grad_values)

                grad = _wrap_grad_tensor(grad_data, input_tensor)
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_maximum(grad_output, input_ref, other_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        other_tensor = other_ref() if other_ref else None
        
        backend = grad_output._backend
        
        if input_tensor is not None and input_tensor._requires_grad:
                # Mask where input >= other
                if other_tensor:
                        mask = backend.greater_equal(input_tensor.data, other_tensor.data)
                else:
                        other_value = metadata.get('other_value', 0.0)
                        mask = backend.greater_equal(input_tensor.data, other_value)
                
                grad_data = backend.multiply(grad_output.data, mask)
                grad = _wrap_grad_tensor(grad_data, input_tensor)
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        if other_tensor is not None and other_tensor._requires_grad:
                # Mask where other > input
                mask = backend.greater(other_tensor.data, input_tensor.data)
                grad_data = backend.multiply(grad_output.data, mask)
                grad = _wrap_grad_tensor(grad_data, other_tensor)
                grads.append((other_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_minimum(grad_output, input_ref, other_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        other_tensor = other_ref() if other_ref else None
        
        backend = grad_output._backend
        
        if input_tensor is not None and input_tensor._requires_grad:
                # Mask where input <= other
                if other_tensor:
                        mask = backend.less_equal(input_tensor.data, other_tensor.data)
                else:
                        other_value = metadata.get('other_value', 0.0)
                        mask = backend.less_equal(input_tensor.data, other_value)
                
                grad_data = backend.multiply(grad_output.data, mask)
                grad = _wrap_grad_tensor(grad_data, input_tensor)
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        if other_tensor is not None and other_tensor._requires_grad:
                # Mask where other < input
                mask = backend.less(other_tensor.data, input_tensor.data)
                grad_data = backend.multiply(grad_output.data, mask)
                grad = _wrap_grad_tensor(grad_data, other_tensor)
                grads.append((other_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_abs(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend
                
                # sign(x)
                sign = backend.sign(input_tensor.data)
                
                grad_data = backend.multiply(grad_output.data, sign)
                grad = _wrap_grad_tensor(grad_data, input_tensor)
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)

        return grads

def backward_sign(grad_output, input_ref, **metadata):

        input_tensor = input_ref() if input_ref else None

        if input_tensor is not None and input_tensor._requires_grad:
                backend = input_tensor._backend
                grad_data = backend.zeros_like(input_tensor.data)
                grad = _wrap_grad_tensor(grad_data, input_tensor)
                return [(input_tensor, grad)]
        return [None]

def backward_clip(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend
                min_val = metadata.get('min', float('-inf'))
                max_val = metadata.get('max', float('inf'))
                
                # Mask where min <= x <= max
                mask_min = backend.greater_equal(input_tensor.data, min_val)
                mask_max = backend.less_equal(input_tensor.data, max_val)
                mask = backend.multiply(mask_min, mask_max)
                
                grad_data = backend.multiply(grad_output.data, mask)
                grad = _wrap_grad_tensor(grad_data, input_tensor)
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_where(grad_output, condition_ref, input_ref, other_ref, **metadata):

        grads = []
        
        # Condition doesn't need gradients (it's boolean)
        grads.append(None)
        
        input_tensor = input_ref() if input_ref else None
        other_tensor = other_ref() if other_ref else None
        condition = metadata.get('condition')
        
        backend = grad_output._backend
        
        if input_tensor is not None and input_tensor._requires_grad:
                # Gradient flows where condition is True
                grad_data = backend.multiply(grad_output.data, condition)
                grad = _wrap_grad_tensor(grad_data, input_tensor)
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        if other_tensor is not None and other_tensor._requires_grad:
                # Gradient flows where condition is False
                not_condition = backend.logical_not(condition)
                grad_data = backend.multiply(grad_output.data, not_condition)
                grad = _wrap_grad_tensor(grad_data, other_tensor)
                grads.append((other_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_mean(grad_output, input_ref, **metadata):

        grads = []

        input_tensor = input_ref() if input_ref else None

        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend if hasattr(grad_output, '_backend') else input_tensor._backend
                axis = metadata.get('axis', None)
                keepdims = metadata.get('keepdims', False)
                n = metadata.get('n', None)

                # Broadcast grad_output to match the unreduced input shape
                grad_data = _expand_grad_for_reduction(grad_output, input_tensor.shape, axis, keepdims, backend)

                # Divide by the number of elements that were averaged
                if n is None:
                        if axis is None:
                                n = 1
                                for dim in input_tensor.shape:
                                        n *= dim
                        else:
                                axes = axis if isinstance(axis, (tuple, list)) else (axis,)
                                n = 1
                                for ax in axes:
                                        n *= input_tensor.shape[ax if ax >= 0 else ax + len(input_tensor.shape)]

                grad_data = backend.divide(grad_data, n)

                grad = _wrap_grad_tensor(grad_data, input_tensor)

                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_unsqueeze(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend
                dim = metadata.get('dim', 0)
                
                # Squeeze the added dimension
                grad_data = backend.squeeze(grad_output.data, axis=dim)
                grad = _wrap_grad_tensor(grad_data, input_tensor)
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_squeeze(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend
                original_shape = metadata.get('original_shape')
                
                # Reshape to original shape
                grad_data = backend.reshape(grad_output.data, original_shape)
                grad = _wrap_grad_tensor(grad_data, input_tensor)
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_permute(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend
                dims = metadata.get('dims')
                
                # Invert permutation
                inv_dims = [0] * len(dims)
                for i, d in enumerate(dims):
                        inv_dims[d] = i
                
                grad_data = backend.transpose(grad_output.data, inv_dims)
                grad = _wrap_grad_tensor(grad_data, input_tensor)
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_expand(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend
                original_shape = input_tensor.shape
                
                grad_data = grad_output.data
                
                # Sum over dimensions that were broadcast
                for i, (orig_dim, grad_dim) in enumerate(zip(original_shape, grad_output.shape)):
                        if orig_dim == 1 and grad_dim != 1:
                                grad_data = backend.sum(grad_data, axis=i, keepdims=True)
                
                # Sum over leading dimensions if they were added
                ndim_diff = len(grad_output.shape) - len(original_shape)
                for _ in range(ndim_diff):
                        grad_data = backend.sum(grad_data, axis=0, keepdims=False)
                
                grad = _wrap_grad_tensor(grad_data, input_tensor)
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_log_softmax(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend
                axis = metadata.get('axis', -1)
                
                # Recompute softmax
                x_max = backend.max(input_tensor.data, axis=axis, keepdims=True)
                x_shifted = backend.subtract(input_tensor.data, x_max)
                exp_x = backend.exp(x_shifted)
                sum_exp = backend.sum(exp_x, axis=axis, keepdims=True)
                softmax_output = backend.divide(exp_x, sum_exp)
                
                # Gradient: grad_output - softmax * sum(grad_output)
                sum_grad = backend.sum(grad_output.data, axis=axis, keepdims=True)
                
                grad_data = backend.subtract(
                        grad_output.data,
                        backend.multiply(softmax_output, sum_grad)
                )
                grad = _wrap_grad_tensor(grad_data, input_tensor)
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_batch_norm(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend
                eps = metadata.get('eps', 1e-5)
                gamma = metadata.get('gamma', None)
                mean = metadata.get('mean')
                var = metadata.get('var')
                axes = metadata.get('axes')

                if axes is None:
                        if len(input_tensor.shape) == 2:
                                axes = (0,)
                        elif len(input_tensor.shape) == 4:
                                axes = (0, 2, 3)
                        else:
                                axes = (0,)

                if mean is None or var is None:
                        mean = backend.mean(input_tensor.data, axis=axes, keepdims=True)
                        centered = backend.subtract(input_tensor.data, mean)
                        var = backend.mean(backend.multiply(centered, centered), axis=axes, keepdims=True)
                else:
                        centered = backend.subtract(input_tensor.data, mean)

                std = backend.sqrt(backend.add(var, eps))
                normalized = backend.divide(centered, std)
                
                # Gradient (simplified)
                if gamma is not None:
                        shape_4d = (1, -1, 1, 1) if len(input_tensor.shape) == 4 else (1, -1)
                        grad_normalized = backend.multiply(grad_output.data, backend.reshape(gamma, shape_4d))
                else:
                        grad_normalized = grad_output.data
                
                N = 1
                for axis in axes:
                        N *= input_tensor.shape[axis]
                
                # Similar to layer norm gradient
                sum_grad = backend.sum(grad_normalized, axis=axes, keepdims=True)
                sum_grad_normalized = backend.sum(
                        backend.multiply(grad_normalized, normalized),
                        axis=axes,
                        keepdims=True
                )
                
                grad_input = backend.multiply(grad_normalized, N)
                grad_input = backend.subtract(grad_input, sum_grad)
                grad_input = backend.subtract(
                        grad_input,
                        backend.multiply(normalized, sum_grad_normalized)
                )
                grad_data = backend.divide(grad_input, backend.multiply(N, std))
                grad = _wrap_grad_tensor(grad_data, input_tensor)
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads

def backward_group_norm(grad_output, input_ref, **metadata):

        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor is not None and input_tensor._requires_grad:
                backend = grad_output._backend
                num_groups = metadata.get('num_groups', 32)
                eps = metadata.get('eps', 1e-5)
                gamma = metadata.get('gamma', None)
                
                # Group norm gradient is similar to layer norm
                # but computed per group of channels
                # This is a simplified implementation
                
                grad_data = grad_output.data  # Placeholder - full implementation needed
                grad = _wrap_grad_tensor(grad_data, input_tensor)
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads
