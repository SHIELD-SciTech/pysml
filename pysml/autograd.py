import builtins
import weakref
from typing import Callable, List, Tuple, Optional

from .tensor import Tensor


class Function:
        __slots__ = ('inputs', 'backward_fn', 'metadata', 'next_functions', '_saved_inputs')

        def __init__(self, backward_fn: Callable, inputs: List, metadata: dict = None):
                # Keep weak references for graph traversal but retain strong references for backward.
                self.inputs = [weakref.ref(t) if t is not None else None for t in inputs]
                self._saved_inputs = [t for t in inputs if t is not None]
                self.backward_fn = backward_fn
                self.metadata = metadata or {}
                self.next_functions = []  # For graph traversal

        def release_graph(self):
                """Release saved inputs to allow tensors to be garbage collected."""

                self._saved_inputs = None
                self.next_functions = []

        def apply_backward(self, grad_output):
                try:
                        return self.backward_fn(grad_output, *self.inputs, **self.metadata)
                finally:
                        # Release saved references after backward to avoid leaks even if backward fails.
                        self._saved_inputs = None


def register_post_backward_hook(tensor: Tensor, hook: Callable[[Tensor], None]):
        """Register a hook that fires after ``tensor`` receives its gradient."""

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


def set_grad_enabled(mode: bool):
        global _grad_enabled
        _grad_enabled = mode


def _reduce_grad_to_shape(grad_data, target_shape, backend):
        """Reduce ``grad_data`` to ``target_shape`` following broadcasting rules."""

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


def _expand_grad_for_reduction(grad_output, input_shape, axis, keepdims, backend):
        """Broadcast ``grad_output`` to ``input_shape`` following sum/mean semantics."""

        grad_data = grad_output.data if hasattr(grad_output, 'data') else grad_output

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

        return backend.broadcast_to(grad_data, input_shape)


def _wrap_grad_tensor(grad_data, reference: Tensor):
        """Create a gradient tensor matching ``reference`` without manual instantiation."""

        if reference is None:
                return grad_data

        payload = grad_data.data if hasattr(grad_data, 'data') else grad_data
        return reference._new_like(payload, requires_grad=False)



def backward_add(grad_output, input_ref, other_ref, **metadata):
        """
        Backward for addition: z = x + y
        
        dL/dx = dL/dz * 1 = dL/dz
        dL/dy = dL/dz * 1 = dL/dz
        """
        grads = []
        
        backend = grad_output._backend if hasattr(grad_output, '_backend') else None

        # Gradient for first input
        input_tensor = input_ref() if input_ref else None
        if input_tensor and input_tensor._requires_grad:
                grad_data = grad_output.data if hasattr(grad_output, 'data') else grad_output
                if hasattr(grad_output, 'shape') and hasattr(input_tensor, 'shape'):
                        grad_data = _reduce_grad_to_shape(grad_data, input_tensor.shape, backend or input_tensor._backend)

                grad_tensor = _wrap_grad_tensor(grad_data, input_tensor)

                grads.append((input_tensor, grad_tensor))
        else:
                grads.append(None)

        # Gradient for second input
        other_tensor = other_ref() if other_ref else None
        if other_tensor and other_tensor._requires_grad:
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
        """
        Backward for subtraction: z = x - y

        dL/dx = dL/dz * 1 = dL/dz
        dL/dy = dL/dz * (-1) = -dL/dz
        """
        grads = []

        backend = grad_output._backend if hasattr(grad_output, '_backend') else None

        input_tensor = input_ref() if input_ref else None
        if input_tensor and input_tensor._requires_grad:
                grad_data = grad_output.data if hasattr(grad_output, 'data') else grad_output
                if hasattr(grad_output, 'shape') and hasattr(input_tensor, 'shape'):
                        grad_data = _reduce_grad_to_shape(grad_data, input_tensor.shape, backend or input_tensor._backend)

                grad = _wrap_grad_tensor(grad_data, input_tensor)
                grads.append((input_tensor, grad))
        else:
                grads.append(None)

        other_tensor = other_ref() if other_ref else None
        if other_tensor and other_tensor._requires_grad:
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
        """
        Backward for multiplication: z = x * y
        
        dL/dx = dL/dz * y
        dL/dy = dL/dz * x
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        other_tensor = other_ref() if other_ref else None
        backend = grad_output._backend if hasattr(grad_output, '_backend') else None

        if input_tensor and input_tensor._requires_grad:
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

        if other_tensor and other_tensor._requires_grad:
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
        """
        Backward for division: z = x / y
        
        dL/dx = dL/dz * (1/y)
        dL/dy = dL/dz * (-x/y^2)
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        other_tensor = other_ref() if other_ref else None
        backend = grad_output._backend if hasattr(grad_output, '_backend') else None

        if input_tensor and input_tensor._requires_grad:
                # dL/dx = grad_output / other
                if other_tensor:
                        active_backend = backend or other_tensor._backend
                        grad = type(grad_output).__new__(type(grad_output))
                        grad._backend = active_backend
                        grad._dtype = grad_output._dtype
                        grad.device = grad_output.device
                        grad.active_device = grad_output.active_device
                        grad.data = active_backend.divide(grad_output.data, other_tensor.data)
                        grad.data = _reduce_grad_to_shape(grad.data, input_tensor.shape, active_backend)
                        grad._requires_grad = False
                        grad._grad = None
                        grad._grad_fn = None
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

        if other_tensor and other_tensor._requires_grad:
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
        """
        Backward for power: z = x^n
        
        dL/dx = dL/dz * n * x^(n-1)
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor and input_tensor._requires_grad:
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
        """
        Backward for matrix multiplication: Z = X @ Y
        
        dL/dX = dL/dZ @ Y^T
        dL/dY = X^T @ dL/dZ
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        other_tensor = other_ref() if other_ref else None
        
        if input_tensor and input_tensor._requires_grad:
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
        
        if other_tensor and other_tensor._requires_grad:
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
        """
        Backward for ReLU: z = max(0, x)
        
        dL/dx = dL/dz * (x > 0)
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor and input_tensor._requires_grad:
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
        """
        Backward for exp: z = exp(x)
        
        dL/dx = dL/dz * exp(x) = dL/dz * z
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor and input_tensor._requires_grad:
                backend = grad_output._backend
                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device
                
                # exp(input)
                exp_x = backend.exp(input_tensor.data)
                # grad_output * exp(x)
                grad.data = backend.multiply(grad_output.data, exp_x)
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads


def backward_log(grad_output, input_ref, **metadata):
        """
        Backward for log: z = log(x)
        
        dL/dx = dL/dz * (1/x)
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor and input_tensor._requires_grad:
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
        """
        Backward for tanh: z = tanh(x)
        
        dL/dx = dL/dz * (1 - tanh^2(x)) = dL/dz * (1 - z^2)
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None

        if input_tensor and input_tensor._requires_grad:
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
    """
    Backward for sum: z = sum(x)

    dL/dx = dL/dz * ones_like(x)
    """
    grads = []

    input_tensor = input_ref() if input_ref else None

    if input_tensor and input_tensor._requires_grad:
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
        """
        Backward for transpose: z = x^T
        
        dL/dx = (dL/dz)^T
        
        Transpose gradient back to match input shape.
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None

        if input_tensor and input_tensor._requires_grad:
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
        """
        Backward for reshape: z = reshape(x, shape)
        
        dL/dx = reshape(dL/dz, x.shape)
        
        Reshape gradient back to match input shape.
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor and input_tensor._requires_grad:
                backend = grad_output._backend
                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device
                
                # Reshape gradient back to input shape
                original_shape = metadata.get('original_shape', input_tensor.shape)
                grad.data = backend.reshape(grad_output.data, original_shape)
                
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads


def backward_sigmoid(grad_output, input_ref, **metadata):
        """
        Backward for sigmoid: z = sigmoid(x) = 1 / (1 + exp(-x))
        
        dL/dx = dL/dz * sigmoid(x) * (1 - sigmoid(x))
                 = dL/dz * z * (1 - z)
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None

        if input_tensor and input_tensor._requires_grad:
                backend = grad_output._backend
                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device

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
                grad.data = backend.multiply(grad_output.data, grad_sigmoid)
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads


def backward_sqrt(grad_output, input_ref, **metadata):
        """
        Backward for sqrt: z = sqrt(x)
        
        dL/dx = dL/dz * 1/(2*sqrt(x))
                 = dL/dz / (2*z)
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor and input_tensor._requires_grad:
                backend = grad_output._backend
                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device
                
                # 1 / (2 * sqrt(x))
                sqrt_x = backend.sqrt(input_tensor.data)
                grad_sqrt = backend.reciprocal(backend.multiply(2.0, sqrt_x))
                
                # grad_output * grad_sqrt
                grad.data = backend.multiply(grad_output.data, grad_sqrt)
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads


def backward_sin(grad_output, input_ref, **metadata):
        """
        Backward for sin: z = sin(x)
        
        dL/dx = dL/dz * cos(x)
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor and input_tensor._requires_grad:
                backend = grad_output._backend
                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device
                
                # cos(x)
                cos_x = backend.cos(input_tensor.data)
                
                # grad_output * cos(x)
                grad.data = backend.multiply(grad_output.data, cos_x)
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads


def backward_cos(grad_output, input_ref, **metadata):
        """
        Backward for cos: z = cos(x)
        
        dL/dx = dL/dz * (-sin(x))
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor and input_tensor._requires_grad:
                backend = grad_output._backend
                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device
                
                # -sin(x)
                sin_x = backend.sin(input_tensor.data)
                neg_sin_x = backend.negative(sin_x)
                
                # grad_output * (-sin(x))
                grad.data = backend.multiply(grad_output.data, neg_sin_x)
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads# Additional backward functions for modern deep learning models
# Append these to the existing autograd.py file


def backward_softmax(grad_output, input_ref, **metadata):
        """
        Backward for softmax: z = softmax(x)
        
        dL/dx_i = z_i * (dL/dz_i - sum_j(z_j * dL/dz_j))
        
        For numerical stability, we use:
        dL/dx = z * (dL/dz - (z · dL/dz))
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None

        if input_tensor and input_tensor._requires_grad:
                backend = grad_output._backend
                axis = metadata.get('axis', -1)
                softmax_output = metadata.get('output')

                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device

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
                grad.data = backend.multiply(
                        softmax_output,
                        backend.subtract(grad_output.data, sum_term)
                )
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads


def backward_layer_norm(grad_output, input_ref, **metadata):
        """
        Backward for layer normalization
        
        y = (x - mean) / sqrt(var + eps) * gamma + beta
        
        This is complex - involves gradients through mean, variance, and affine transform
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor and input_tensor._requires_grad:
                backend = grad_output._backend
                normalized_shape = metadata.get('normalized_shape')
                eps = metadata.get('eps', 1e-5)
                gamma = metadata.get('gamma', None)  # Scale parameter
                saved_mean = metadata.get('mean')
                inv_std = metadata.get('inv_std')

                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device

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
                grad.data = backend.divide(grad_input, backend.multiply(N, std))
                
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads


def backward_rms_norm(grad_output, input_ref, **metadata):
        """
        Backward for RMS normalization (used in LLaMA, Mistral, etc.)
        
        y = x / rms(x) * gamma, where rms(x) = sqrt(mean(x^2) + eps)
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor and input_tensor._requires_grad:
                backend = grad_output._backend
                normalized_shape = metadata.get('normalized_shape')
                eps = metadata.get('eps', 1e-6)
                gamma = metadata.get('gamma', None)
                inv_rms = metadata.get('inv_rms')

                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device

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
                
                grad.data = backend.subtract(term1, term2)
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads


def backward_gelu(grad_output, input_ref, **metadata):
        """
        Backward for GELU activation
        
        GELU(x) = x * Φ(x), where Φ is the CDF of standard normal distribution
        Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor and input_tensor._requires_grad:
                backend = grad_output._backend
                
                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device
                
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
                        backend.multiply(0.5 * x, sech_squared),
                        inner_derivative
                )
                gelu_grad = backend.add(term1, term2)
                
                grad.data = backend.multiply(grad_output.data, gelu_grad)
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads


def backward_silu(grad_output, input_ref, **metadata):
        """
        Backward for SiLU/Swish activation: f(x) = x * sigmoid(x)
        
        f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                  = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor and input_tensor._requires_grad:
                backend = grad_output._backend
                
                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device
                
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
                
                grad.data = backend.multiply(grad_output.data, silu_grad)
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads


def backward_dropout(grad_output, input_ref, **metadata):
        """
        Backward for dropout
        
        During training: y = x * mask / (1 - p)
        During inference: y = x
        
        Gradient: dL/dx = dL/dy * mask / (1 - p)
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor and input_tensor._requires_grad:
                backend = grad_output._backend
                mask = metadata.get('mask')  # Binary mask from forward pass
                p = metadata.get('p', 0.5)
                training = metadata.get('training', True)
                
                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device
                
                if training and mask is not None:
                        # Scale by mask and inverse keep probability
                        scale = 1.0 / (1.0 - p)
                        grad.data = backend.multiply(
                                backend.multiply(grad_output.data, mask),
                                scale
                        )
                else:
                        # No dropout during inference
                        grad.data = grad_output.data
                
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads


def backward_embedding(grad_output, input_ref, **metadata):
        """
        Backward for embedding lookup
        
        Forward: output = embedding_table[indices]
        Backward: Accumulate gradients at the indices used
        
        This returns a sparse gradient update for the embedding table
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor and input_tensor._requires_grad:
                backend = grad_output._backend
                indices = metadata.get('indices')
                num_embeddings = metadata.get('num_embeddings')

                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device

                embedding_dim = grad_output.shape[-1]
                grad_shape = (num_embeddings, embedding_dim)
                grad.data = backend.zeros(grad_shape, dtype=grad_output.data.dtype)

                if hasattr(indices, 'data'):
                        indices_data = indices.data
                else:
                        indices_data = indices

                indices_array = backend.asarray(indices_data)

                backend.scatter_add(
                        grad.data,
                        0,
                        indices_array,
                        grad_output.data,
                )

                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None

                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads


def backward_concatenate(grad_output, *input_refs, **metadata):
        """
        Backward for concatenate
        
        Split the gradient back to original tensor sizes
        """
        grads = []
        axis = metadata.get('axis', 0)
        
        backend = grad_output._backend
        
        # Split gradient along concatenation axis
        current_idx = 0
        for input_ref in input_refs:
                input_tensor = input_ref() if input_ref else None
                
                if input_tensor and input_tensor._requires_grad:
                        grad = type(grad_output).__new__(type(grad_output))
                        grad._backend = backend
                        grad._dtype = grad_output._dtype
                        grad.device = grad_output.device
                        grad.active_device = grad_output.active_device
                        
                        # Slice gradient for this input
                        size = input_tensor.shape[axis]
                        
                        # Create slice indices
                        slices = [slice(None)] * len(grad_output.shape)
                        slices[axis] = slice(current_idx, current_idx + size)
                        
                        grad.data = grad_output.data[tuple(slices)]
                        grad._requires_grad = False
                        grad._grad = None
                        grad._grad_fn = None
                        
                        grads.append((input_tensor, grad))
                        current_idx += size
                else:
                        grads.append(None)
        
        return grads


def backward_split(grad_output, input_ref, **metadata):
        """
        Backward for split/chunk

        Concatenate gradients from all output chunks
        """
        grads = []

        input_tensor = input_ref() if input_ref else None

        if input_tensor and input_tensor._requires_grad:
                backend = input_tensor._backend
                axis = metadata.get('axis', 0)
                sizes = metadata.get('sizes', [])
                index = metadata.get('index', 0)
                shared_state = metadata.get('shared_state', {})

                grad_data = shared_state.get('buffer')
                if grad_data is None:
                        grad_data = backend.zeros_like(input_tensor.data)
                        shared_state['buffer'] = grad_data

                if isinstance(grad_output, (list, tuple)):
                        for idx, g in enumerate(grad_output):
                                if g is None:
                                        continue
                                if grad_data is None:
                                        grad_data = backend.zeros_like(input_tensor.data)
                                        shared_state['buffer'] = grad_data
                                start = builtins.sum(sizes[:idx]) if sizes else 0
                                end = start + (sizes[idx] if sizes else g.data.shape[axis])
                                slices = [slice(None)] * grad_data.ndim
                                slices[axis] = slice(start, end)
                                grad_data[tuple(slices)] = g.data
                elif grad_output is not None:
                        if grad_data is None:
                                grad_data = backend.zeros_like(input_tensor.data)
                                shared_state['buffer'] = grad_data
                        start = builtins.sum(sizes[:index]) if sizes else 0
                        end = start + (sizes[index] if sizes else grad_output.data.shape[axis])
                        slices = [slice(None)] * grad_data.ndim
                        slices[axis] = slice(start, end)
                        grad_data[tuple(slices)] = grad_output.data

                shared_state['completed'] = shared_state.get('completed', 0) + 1

                grad_tensor = None
                if shared_state.get('completed', 0) >= shared_state.get('num_chunks', 1):
                        grad_tensor = type(input_tensor).__new__(type(input_tensor))
                        grad_tensor._backend = backend
                        grad_tensor._dtype = input_tensor._dtype
                        grad_tensor.device = input_tensor.device
                        grad_tensor.active_device = input_tensor.active_device
                        grad_tensor.data = grad_data
                        grad_tensor._requires_grad = False
                        grad_tensor._grad = None
                        grad_tensor._grad_fn = None

                        shared_state['buffer'] = None

                grads.append((input_tensor, grad_tensor))
        else:
                grads.append(None)

        return grads


def backward_max_reduce(grad_output, input_ref, **metadata):
        """
        Backward for max reduction
        
        Gradient flows only to the maximum element(s)
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor and input_tensor._requires_grad:
                backend = grad_output._backend
                axis = metadata.get('axis', None)
                keepdims = metadata.get('keepdims', False)
                max_indices = metadata.get('max_indices')

                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device

                grad.data = backend.zeros_like(input_tensor.data)

                if axis is None:
                        flat_grad = backend.reshape(grad.data, (-1,))
                        grad_value = grad_output.data
                        grad_value = backend.reshape(grad_value, (-1,))
                        if grad_value.shape[0] != 1:
                                grad_value = backend.reshape(grad_value, (1,))

                        index_array = backend.asarray([max_indices])
                        backend.scatter_add(flat_grad, 0, index_array, grad_value)
                else:
                        grad_values = grad_output.data
                        indices = max_indices

                        if not keepdims:
                                grad_values = backend.expand_dims(grad_values, axis=axis)
                                indices = backend.expand_dims(indices, axis=axis)

                        backend.scatter_add(grad.data, axis, indices, grad_values)

                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None

                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads


def backward_min_reduce(grad_output, input_ref, **metadata):
        """
        Backward for min reduction
        
        Gradient flows only to the minimum element(s)
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor and input_tensor._requires_grad:
                backend = grad_output._backend
                axis = metadata.get('axis', None)
                keepdims = metadata.get('keepdims', False)
                min_indices = metadata.get('min_indices')

                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device

                grad.data = backend.zeros_like(input_tensor.data)

                if axis is None:
                        flat_grad = backend.reshape(grad.data, (-1,))
                        grad_value = grad_output.data
                        grad_value = backend.reshape(grad_value, (-1,))
                        if grad_value.shape[0] != 1:
                                grad_value = backend.reshape(grad_value, (1,))

                        index_array = backend.asarray([min_indices])
                        backend.scatter_add(flat_grad, 0, index_array, grad_value)
                else:
                        grad_values = grad_output.data
                        indices = min_indices

                        if not keepdims:
                                grad_values = backend.expand_dims(grad_values, axis=axis)
                                indices = backend.expand_dims(indices, axis=axis)

                        backend.scatter_add(grad.data, axis, indices, grad_values)

                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None

                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads


def backward_maximum(grad_output, input_ref, other_ref, **metadata):
        """
        Backward for element-wise maximum: z = max(x, y)
        
        Gradient flows to the larger input
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        other_tensor = other_ref() if other_ref else None
        
        backend = grad_output._backend
        
        if input_tensor and input_tensor._requires_grad:
                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device
                
                # Mask where input >= other
                if other_tensor:
                        mask = backend.greater_equal(input_tensor.data, other_tensor.data)
                else:
                        other_value = metadata.get('other_value', 0.0)
                        mask = backend.greater_equal(input_tensor.data, other_value)
                
                grad.data = backend.multiply(grad_output.data, mask)
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        if other_tensor and other_tensor._requires_grad:
                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device
                
                # Mask where other > input
                mask = backend.greater(other_tensor.data, input_tensor.data)
                grad.data = backend.multiply(grad_output.data, mask)
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((other_tensor, grad))
        else:
                grads.append(None)
        
        return grads


def backward_minimum(grad_output, input_ref, other_ref, **metadata):
        """
        Backward for element-wise minimum: z = min(x, y)
        
        Gradient flows to the smaller input
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        other_tensor = other_ref() if other_ref else None
        
        backend = grad_output._backend
        
        if input_tensor and input_tensor._requires_grad:
                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device
                
                # Mask where input <= other
                if other_tensor:
                        mask = backend.less_equal(input_tensor.data, other_tensor.data)
                else:
                        other_value = metadata.get('other_value', 0.0)
                        mask = backend.less_equal(input_tensor.data, other_value)
                
                grad.data = backend.multiply(grad_output.data, mask)
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        if other_tensor and other_tensor._requires_grad:
                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device
                
                # Mask where other < input
                mask = backend.less(other_tensor.data, input_tensor.data)
                grad.data = backend.multiply(grad_output.data, mask)
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((other_tensor, grad))
        else:
                grads.append(None)
        
        return grads


def backward_abs(grad_output, input_ref, **metadata):
        """
        Backward for absolute value: z = |x|
        
        dL/dx = dL/dz * sign(x)
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor and input_tensor._requires_grad:
                backend = grad_output._backend
                
                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device
                
                # sign(x)
                sign = backend.sign(input_tensor.data)
                
                grad.data = backend.multiply(grad_output.data, sign)
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)

        return grads


def backward_sign(grad_output, input_ref, **metadata):
        """
        Backward for sign

        The derivative of sign is zero everywhere except at 0 (undefined). We
        propagate a zero gradient to allow the graph to remain connected.
        """
        input_tensor = input_ref() if input_ref else None

        if input_tensor and input_tensor._requires_grad:
                backend = input_tensor._backend
                grad_data = backend.zeros_like(input_tensor.data)
                grad = _wrap_grad_tensor(grad_data, input_tensor)
                return [(input_tensor, grad)]
        return [None]


def backward_clip(grad_output, input_ref, **metadata):
        """
        Backward for clip/clamp: z = clip(x, min, max)
        
        Gradient flows through only where min <= x <= max
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor and input_tensor._requires_grad:
                backend = grad_output._backend
                min_val = metadata.get('min', float('-inf'))
                max_val = metadata.get('max', float('inf'))
                
                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device
                
                # Mask where min <= x <= max
                mask_min = backend.greater_equal(input_tensor.data, min_val)
                mask_max = backend.less_equal(input_tensor.data, max_val)
                mask = backend.multiply(mask_min, mask_max)
                
                grad.data = backend.multiply(grad_output.data, mask)
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads


def backward_where(grad_output, condition_ref, input_ref, other_ref, **metadata):
        """
        Backward for where/select: z = where(condition, x, y)
        
        Gradient flows to x where condition is True, to y where False
        """
        grads = []
        
        # Condition doesn't need gradients (it's boolean)
        grads.append(None)
        
        input_tensor = input_ref() if input_ref else None
        other_tensor = other_ref() if other_ref else None
        condition = metadata.get('condition')
        
        backend = grad_output._backend
        
        if input_tensor and input_tensor._requires_grad:
                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device
                
                # Gradient flows where condition is True
                grad.data = backend.multiply(grad_output.data, condition)
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        if other_tensor and other_tensor._requires_grad:
                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device
                
                # Gradient flows where condition is False
                not_condition = backend.logical_not(condition)
                grad.data = backend.multiply(grad_output.data, not_condition)
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((other_tensor, grad))
        else:
                grads.append(None)
        
        return grads


def backward_mean(grad_output, input_ref, **metadata):
        """
        Backward for mean: z = mean(x)
        
        dL/dx = dL/dz * (1/N) * ones_like(x)
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor and input_tensor._requires_grad:
                backend = grad_output._backend if hasattr(grad_output, '_backend') else input_tensor._backend
                axis = metadata.get('axis', None)
                keepdims = metadata.get('keepdims', False)
                
                grad = type(input_tensor).__new__(type(input_tensor))
                grad._backend = backend
                grad._dtype = input_tensor._dtype
                grad.device = input_tensor.device
                grad.active_device = input_tensor.active_device
                
                # Broadcast grad_output to input shape and divide by count
                grad.data = backend.ones_like(input_tensor.data)
                
                # Calculate number of elements that were averaged
                if axis is None:
                        N = 1
                        for dim in input_tensor.shape:
                                N *= dim
                elif isinstance(axis, int):
                        N = input_tensor.shape[axis]
                else:
                        N = 1
                        for ax in axis:
                                N *= input_tensor.shape[ax]
                
                if hasattr(grad_output, 'data'):
                        grad.data = backend.multiply(grad.data, grad_output.data)
                else:
                        grad.data = backend.multiply(grad.data, grad_output)
                
                grad.data = backend.divide(grad.data, N)
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads


def backward_unsqueeze(grad_output, input_ref, **metadata):
        """
        Backward for unsqueeze: z = unsqueeze(x, dim)
        
        Remove the added dimension
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor and input_tensor._requires_grad:
                backend = grad_output._backend
                dim = metadata.get('dim', 0)
                
                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device
                
                # Squeeze the added dimension
                grad.data = backend.squeeze(grad_output.data, axis=dim)
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads


def backward_squeeze(grad_output, input_ref, **metadata):
        """
        Backward for squeeze: z = squeeze(x, dim)
        
        Re-add the removed dimension(s)
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor and input_tensor._requires_grad:
                backend = grad_output._backend
                original_shape = metadata.get('original_shape')
                
                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device
                
                # Reshape to original shape
                grad.data = backend.reshape(grad_output.data, original_shape)
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads


def backward_permute(grad_output, input_ref, **metadata):
        """
        Backward for permute: z = permute(x, dims)
        
        Invert the permutation
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor and input_tensor._requires_grad:
                backend = grad_output._backend
                dims = metadata.get('dims')
                
                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device
                
                # Invert permutation
                inv_dims = [0] * len(dims)
                for i, d in enumerate(dims):
                        inv_dims[d] = i
                
                grad.data = backend.transpose(grad_output.data, inv_dims)
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads


def backward_expand(grad_output, input_ref, **metadata):
        """
        Backward for expand/broadcast: z = expand(x, shape)
        
        Sum gradients over broadcast dimensions
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor and input_tensor._requires_grad:
                backend = grad_output._backend
                original_shape = input_tensor.shape
                
                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device
                
                grad_data = grad_output.data
                
                # Sum over dimensions that were broadcast
                for i, (orig_dim, grad_dim) in enumerate(zip(original_shape, grad_output.shape)):
                        if orig_dim == 1 and grad_dim != 1:
                                grad_data = backend.sum(grad_data, axis=i, keepdims=True)
                
                # Sum over leading dimensions if they were added
                ndim_diff = len(grad_output.shape) - len(original_shape)
                for _ in range(ndim_diff):
                        grad_data = backend.sum(grad_data, axis=0, keepdims=False)
                
                grad.data = grad_data
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads


def backward_log_softmax(grad_output, input_ref, **metadata):
        """
        Backward for log_softmax: z = log(softmax(x))
        
        More numerically stable than softmax followed by log
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor and input_tensor._requires_grad:
                backend = grad_output._backend
                axis = metadata.get('axis', -1)
                
                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device
                
                # Recompute softmax
                x_max = backend.max(input_tensor.data, axis=axis, keepdims=True)
                x_shifted = backend.subtract(input_tensor.data, x_max)
                exp_x = backend.exp(x_shifted)
                sum_exp = backend.sum(exp_x, axis=axis, keepdims=True)
                softmax_output = backend.divide(exp_x, sum_exp)
                
                # Gradient: grad_output - softmax * sum(grad_output)
                sum_grad = backend.sum(grad_output.data, axis=axis, keepdims=True)
                
                grad.data = backend.subtract(
                        grad_output.data,
                        backend.multiply(softmax_output, sum_grad)
                )
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads


def backward_batch_norm(grad_output, input_ref, **metadata):
        """
        Backward for batch normalization

        This is a simplified version - full implementation needs running statistics
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor and input_tensor._requires_grad:
                backend = grad_output._backend
                eps = metadata.get('eps', 1e-5)
                gamma = metadata.get('gamma', None)

                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device

                # Batch norm normalizes over batch dimension (0)
                # and spatial dimensions (2, 3, ...) for CNNs
                # keeping channel dimension (1) separate

                axes = metadata.get('axes')
                if axes is None:
                        axes = tuple(range(len(input_tensor.shape)))
                        axes = tuple([ax for ax in axes if ax != 1])

                # Prefer cached statistics to avoid recomputation.
                mean = metadata.get('mean')
                var = metadata.get('var')
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
                        grad_normalized = backend.multiply(grad_output.data, gamma)
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
                grad.data = backend.divide(grad_input, backend.multiply(N, std))
                
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads


def backward_group_norm(grad_output, input_ref, **metadata):
        """
        Backward for group normalization (used in diffusion models)
        
        Similar to layer norm but normalizes within groups of channels
        """
        grads = []
        
        input_tensor = input_ref() if input_ref else None
        
        if input_tensor and input_tensor._requires_grad:
                backend = grad_output._backend
                num_groups = metadata.get('num_groups', 32)
                eps = metadata.get('eps', 1e-5)
                gamma = metadata.get('gamma', None)
                
                grad = type(grad_output).__new__(type(grad_output))
                grad._backend = backend
                grad._dtype = grad_output._dtype
                grad.device = grad_output.device
                grad.active_device = grad_output.active_device
                
                # Group norm gradient is similar to layer norm
                # but computed per group of channels
                # This is a simplified implementation
                
                grad.data = grad_output.data  # Placeholder
                grad._requires_grad = False
                grad._grad = None
                grad._grad_fn = None
                
                grads.append((input_tensor, grad))
        else:
                grads.append(None)
        
        return grads
