import numpy as np
from pysml.tensor import Tensor

# Helper function to get appropriate backend
def _get_backend(arr):
    """Detect backend and return appropriate numpy-like module."""
    if hasattr(arr, '__sycl_usm_array_interface__'):
        import dpnp
        return dpnp, 'xpu'
    elif hasattr(arr, '__cuda_array_interface__'):
        import cupy
        return cupy, 'cuda'
    else:
        import numpy
        return numpy, 'cpu'

# Helper function to reduce gradients for broadcasting
def _reduce_gradient(grad, original_shape):
    """Reduce gradient to match original shape after broadcasting."""
    grad_data = grad.data if isinstance(grad, Tensor) else grad
    
    if grad_data.shape == original_shape:
        return grad
    
    # Sum over extra leading dimensions
    ndims_diff = len(grad_data.shape) - len(original_shape)
    for _ in range(ndims_diff):
        grad_data = np.sum(grad_data, axis=0)
    
    # Sum over dimensions that were size 1 in original
    for i in range(len(original_shape)):
        if i < len(grad_data.shape) and original_shape[i] == 1 and grad_data.shape[i] > 1:
            grad_data = np.sum(grad_data, axis=i, keepdims=True)
    
    return Tensor(grad_data)


class Function:
    def __init__(self, *tensors):
        self.parents = [t for t in tensors if isinstance(t, Tensor)]
        self.saved_tensors = []

    def save_for_backward(self, *tensors):
        self.saved_tensors.extend(tensors)

    @classmethod
    def apply(cls, op, *x):
        ctx = op(*x)
        raw_data = []
        for arg in x:
            if isinstance(arg, Tensor):
                raw_data.append(arg.data)
            else:
                raw_data.append(arg)

        result_data = ctx.forward(*raw_data)
        
        # Check if any parent requires grad
        requires_grad = any(isinstance(arg, Tensor) and arg.requires_grad for arg in x)
        result_tensor = Tensor(result_data, _ctx=ctx if requires_grad else None, requires_grad=requires_grad)
        return result_tensor

    def forward(self, *args):
        raise NotImplementedError("You must implement the forward pass for a function.")

    def backward(self, *args):
        raise NotImplementedError("You must implement the backward pass for a function.")

# --- Loss Function ---
class CrossEntropyLoss(Function):
    def forward(self, logits, targets):
        np_backend, backend_name = _get_backend(logits)
        
        max_logits = np_backend.max(logits, axis=-1, keepdims=True)
        exp_logits = np_backend.exp(logits - max_logits)
        sum_exp_logits = np_backend.sum(exp_logits, axis=-1, keepdims=True)
        probs = exp_logits / sum_exp_logits
        
        self.save_for_backward(Tensor(probs), Tensor(targets.data if isinstance(targets, Tensor) else targets))
        
        num_samples = logits.shape[0]
        targets_data = targets.data if isinstance(targets, Tensor) else targets
        
        # Convert targets to appropriate backend with same queue/device
        if backend_name == 'xpu' and not hasattr(targets_data, '__sycl_usm_array_interface__'):
            targets_data = np_backend.array(targets_data, sycl_queue=logits.sycl_queue)
        elif backend_name == 'cuda' and not hasattr(targets_data, '__cuda_array_interface__'):
            targets_data = np_backend.array(targets_data)
        
        targets_int = targets_data.astype(np_backend.int64) if hasattr(targets_data, 'astype') else targets_data.astype(int)
        
        # Create batch indices on same device
        if backend_name == 'xpu':
            batch_indices = np_backend.arange(num_samples, sycl_queue=logits.sycl_queue)
        else:
            batch_indices = np_backend.arange(num_samples)
        
        selected_probs = probs[batch_indices, targets_int]
        selected_log_probs = np_backend.log(selected_probs)
        loss_val = -np_backend.sum(selected_log_probs) / num_samples
        
        # Convert to Python scalar
        if hasattr(loss_val, 'asnumpy'):
            loss_val = float(loss_val.asnumpy())
        elif hasattr(loss_val, 'item'):
            loss_val = loss_val.item()
        else:
            loss_val = float(loss_val)
        
        return loss_val

    def backward(self, grad_output):
        probs, targets = self.saved_tensors
        np_backend, backend_name = _get_backend(probs.data)
        
        num_samples = probs.shape[0]
        grad = probs.data.copy()
        targets_data = targets.data if isinstance(targets, Tensor) else targets
        
        # Convert to appropriate backend
        if backend_name == 'xpu' and not hasattr(targets_data, '__sycl_usm_array_interface__'):
            targets_data = np_backend.array(targets_data, sycl_queue=probs.data.sycl_queue)
        elif backend_name == 'cuda' and not hasattr(targets_data, '__cuda_array_interface__'):
            targets_data = np_backend.array(targets_data)
        
        targets_int = targets_data.astype(np_backend.int64) if hasattr(targets_data, 'astype') else targets_data.astype(int)
        
        # Create batch indices on same device
        if backend_name == 'xpu':
            batch_indices = np_backend.arange(num_samples, sycl_queue=probs.data.sycl_queue)
        else:
            batch_indices = np_backend.arange(num_samples)
        
        grad[batch_indices, targets_int] -= 1
        grad /= num_samples
        
        return Tensor(grad * grad_output.data), None

# --- Activation Functions ---
class ReLU(Function):
    def forward(self, x):
        self.save_for_backward(Tensor(x))
        # Backend-agnostic ReLU: use element-wise operations instead of np.maximum
        # This works with numpy, cupy, and dpnp
        mask = x > 0
        result = x * mask
        return result
    
    def backward(self, grad_output):
        x, = self.saved_tensors
        mask = x.data > 0
        grad = grad_output.data * mask
        return Tensor(grad)

# --- NEWLY ADDED AUTOGRAD FUNCTIONS ---
class Mean(Function):
    def forward(self, x, axis, keepdims):
        self.x_shape = x.shape
        self.axis = axis
        self.keepdims = keepdims
        
        # Get backend-aware numpy module
        np_backend, _ = _get_backend(x)
        return np_backend.mean(x, axis=axis, keepdims=keepdims)
    
    def backward(self, grad_output):
        if grad_output is None:
            return None
        
        # Get backend-aware numpy module
        np_backend, _ = _get_backend(grad_output.data)
        
        # The gradient is distributed equally to all elements that were part of the mean
        num_elements_in_mean = np_backend.prod(np_backend.array(self.x_shape)) / np_backend.prod(np_backend.array(grad_output.shape))
        grad = grad_output.data / num_elements_in_mean
        return Tensor(np_backend.broadcast_to(grad, self.x_shape))

class Pow(Function):
    def forward(self, x, y_val):
        self.save_for_backward(Tensor(x), Tensor(y_val))
        return x ** y_val

    def backward(self, grad_output):
        if grad_output is None:
            return None, None
        x, y = self.saved_tensors
        return grad_output * (y * (x**(y-1))), None

class MaskedFill(Function):
    def forward(self, x, mask, value):
        self.mask = mask
        return np.where(mask, value, x)

    def backward(self, grad_output):
        if grad_output is None:
            return None, None, None
        return Tensor(grad_output.data * (1 - self.mask.data)), None, None

class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, grad_output):
        if grad_output is None:
            return None
        return -grad_output

# --- Existing Autograd Functions ---
class Slice(Function):
    def forward(self, x, key):
        self.key = key
        self.x_shape = x.shape
        return x[key]

    def backward(self, grad_output):
        if grad_output is None:
            return (None,)
        
        np_backend, _ = _get_backend(grad_output.data)
        
        grad_x = np_backend.zeros(self.x_shape, dtype=grad_output.dtype)
        grad_x[self.key] = grad_output.data
        return (Tensor(grad_x),)

class Add(Function):
    def forward(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x + y
    
    def backward(self, grad_output):
        if grad_output is None:
            return None, None
        
        grad_x = _reduce_gradient(grad_output, self.x_shape)
        grad_y = _reduce_gradient(grad_output, self.y_shape)
        
        return grad_x, grad_y

class Sub(Function):
    def forward(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x - y

    def backward(self, grad_output):
        if grad_output is None:
            return None, None
        
        grad_x = _reduce_gradient(grad_output, self.x_shape)
        grad_y = _reduce_gradient(-grad_output, self.y_shape)
        
        return grad_x, grad_y

class Mul(Function):
    def forward(self, x, y):
        self.save_for_backward(Tensor(x), Tensor(y))
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x * y

    def backward(self, grad_output):
        if grad_output is None:
            return None, None
        x, y = self.saved_tensors
        
        grad_x = _reduce_gradient(grad_output * y, self.x_shape)
        grad_y = _reduce_gradient(grad_output * x, self.y_shape)
        
        return grad_x, grad_y
        
class Div(Function):
    def forward(self, x, y):
        self.save_for_backward(Tensor(x), Tensor(y))
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x / y

    def backward(self, grad_output):
        if grad_output is None:
            return None, None
        x, y = self.saved_tensors
        
        grad_x = _reduce_gradient(grad_output / y, self.x_shape)
        grad_y = _reduce_gradient(grad_output * (-x / (y**2)), self.y_shape)
        
        return grad_x, grad_y

class MatMul(Function):
    def forward(self, x, y):
        self.save_for_backward(Tensor(x), Tensor(y))
        return x @ y

    def backward(self, grad_output):
        if grad_output is None:
            return None, None
        x, y = self.saved_tensors
        
        # Handle different dimensionalities
        # grad_output: same shape as output
        # x shape: (..., n, k)
        # y shape: (k, m) or (..., k, m)
        # output shape: (..., n, m)
        
        # Gradient w.r.t. x: grad_output @ y.T
        # Gradient w.r.t. y: x.T @ grad_output
        
        # For x gradient: grad_output @ y.T
        if y.data.ndim == 2:
            # y is 2D (k, m), need to transpose last two dims
            grad_x = grad_output @ y.T
        else:
            # y is multi-dim, transpose last two dimensions
            grad_x = grad_output @ y.transpose(-2, -1)
        
        # For y gradient: x.T @ grad_output
        if x.data.ndim == 2 and grad_output.data.ndim == 2:
            # Simple 2D case
            grad_y = x.T @ grad_output
        elif x.data.ndim > 2 and y.data.ndim == 2:
            # x is batched but y is not (common case: batched input @ weight matrix)
            # Need to sum over batch dimensions
            # x shape: (batch..., n, k), grad_output shape: (batch..., n, m)
            # We want: (k, m)
            x_reshaped = x.data.reshape(-1, x.data.shape[-1])  # (batch*n, k)
            grad_reshaped = grad_output.data.reshape(-1, grad_output.data.shape[-1])  # (batch*n, m)
            grad_y = Tensor(x_reshaped.T @ grad_reshaped)  # (k, m)
        elif x.data.ndim > 2 and y.data.ndim > 2:
            # Both are batched
            grad_y = x.transpose(-2, -1) @ grad_output
        else:
            # Fallback
            grad_y = x.T @ grad_output
        
        return grad_x, grad_y
        
class GetItem(Function):
    def forward(self, x, idx):
        self.save_for_backward(Tensor(x), Tensor(idx))
        self.x_shape = x.shape
        
        # For dpnp/XPU backend compatibility: ensure idx is on same device/queue as x
        if hasattr(x, '__sycl_usm_array_interface__'):
            # We're on XPU - need to handle dpnp indexing carefully
            import dpnp
            
            # Check if idx needs conversion
            needs_conversion = False
            if not hasattr(idx, '__sycl_usm_array_interface__'):
                # idx is not a dpnp array at all (it's numpy)
                needs_conversion = True
            elif not hasattr(idx, 'sycl_queue'):
                # idx is dpnp but doesn't have sycl_queue attribute
                needs_conversion = True
            elif idx.sycl_queue != x.sycl_queue:
                # idx is dpnp but on different queue
                needs_conversion = True
            
            if needs_conversion:
                # Convert idx to dpnp array on the same queue as x
                try:
                    idx = dpnp.array(idx, sycl_queue=x.sycl_queue)
                except:
                    # Fallback: try without specifying queue
                    import dpctl
                    # Get the queue from x
                    if hasattr(x, 'sycl_queue'):
                        queue = x.sycl_queue
                    else:
                        # Get queue from global context
                        from pysml.tensor import SYCL_QUEUE, TensorType
                        device_id = int(TensorType.device.split(':')[1]) if ':' in TensorType.device else 0
                        queue = SYCL_QUEUE.get(device_id)
                    
                    if queue:
                        idx = dpnp.array(idx, sycl_queue=queue)
                    else:
                        idx = dpnp.array(idx)
        
        return x[idx]
    
    def backward(self, grad_output):
        if grad_output is None:
            return None, None
        x, idx = self.saved_tensors
        
        # Get the appropriate numpy module
        if hasattr(grad_output.data, '__sycl_usm_array_interface__'):
            import dpnp as np_backend
        else:
            import numpy as np_backend
        
        grad_x = np_backend.zeros(self.x_shape, dtype=grad_output.data.dtype)
        
        # Ensure indices are integers
        idx_data = idx.data
        if not np_backend.issubdtype(idx_data.dtype, np_backend.integer):
            idx_data = idx_data.astype(np_backend.int64)
        
        # For dpnp, we need to use a different approach since add.at may not work
        if hasattr(grad_output.data, '__sycl_usm_array_interface__'):
            # dpnp scatter operation - move to CPU for backward pass
            import numpy as np_cpu
            grad_x_cpu = np_cpu.zeros(self.x_shape, dtype=np_cpu.float32)
            idx_cpu = idx_data.asnumpy() if hasattr(idx_data, 'asnumpy') else idx_data
            grad_cpu = grad_output.data.asnumpy() if hasattr(grad_output.data, 'asnumpy') else grad_output.data
            np_cpu.add.at(grad_x_cpu, idx_cpu, grad_cpu)
            # Convert back to dpnp
            grad_x = np_backend.array(grad_x_cpu, sycl_queue=grad_output.data.sycl_queue if hasattr(grad_output.data, 'sycl_queue') else None)
        else:
            np_backend.add.at(grad_x, idx_data, grad_output.data)
        
        return Tensor(grad_x), None

class View(Function):
    def forward(self, x, *shape):
        self.x_shape = x.shape
        return x.reshape(*shape)

    def backward(self, grad_output):
        if grad_output is None:
            return (None,)
        return (grad_output.view(*self.x_shape),)

class Transpose(Function):
    def forward(self, x, dim0, dim1):
        self.dim0, self.dim1 = dim0, dim1
        
        np_backend, _ = _get_backend(x)
        
        axes = list(range(x.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return np_backend.transpose(x, axes)

    def backward(self, grad_output):
        if grad_output is None:
            return (None,)
        return (grad_output.transpose(self.dim0, self.dim1),)



class Conv2dFunction(Function):
    def forward(self, x, weight, bias, stride, padding):
        self.save_for_backward(Tensor(x), Tensor(weight))
        self.stride = stride
        self.padding = padding
        self.has_bias = bias is not None
        
        # Get backend
        np_backend, _ = _get_backend(x)
        
        # Apply padding
        if padding[0] > 0 or padding[1] > 0:
            x_padded = np_backend.pad(
                x, 
                ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
                mode='constant'
            )
        else:
            x_padded = x
        
        batch_size, in_channels, in_h, in_w = x_padded.shape
        out_channels, _, k_h, k_w = weight.shape
        
        # Calculate output dimensions
        out_h = (in_h - k_h) // stride[0] + 1
        out_w = (in_w - k_w) // stride[1] + 1
        
        # Initialize output
        output = np_backend.zeros((batch_size, out_channels, out_h, out_w), dtype=x.dtype)
        
        # Perform convolution
        for b in range(batch_size):
            for oc in range(out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * stride[0]
                        w_start = j * stride[1]
                        h_end = h_start + k_h
                        w_end = w_start + k_w
                        
                        # Extract patch
                        patch = x_padded[b, :, h_start:h_end, w_start:w_end]
                        
                        # Convolve
                        output[b, oc, i, j] = np_backend.sum(patch * weight[oc])
        
        # Add bias if present
        if bias is not None:
            output += bias.reshape(1, -1, 1, 1)
        
        return output
    
    def backward(self, grad_output):
        x, weight = self.saved_tensors
        np_backend, _ = _get_backend(grad_output.data)
        
        # Apply padding to input for gradient computation
        if self.padding[0] > 0 or self.padding[1] > 0:
            x_padded = np_backend.pad(
                x.data,
                ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])),
                mode='constant'
            )
        else:
            x_padded = x.data
        
        batch_size, in_channels, in_h, in_w = x_padded.shape
        out_channels, _, k_h, k_w = weight.shape
        _, _, out_h, out_w = grad_output.data.shape
        
        # Gradient w.r.t. input
        grad_x_padded = np_backend.zeros_like(x_padded)
        
        # Gradient w.r.t. weight
        grad_weight = np_backend.zeros_like(weight.data)
        
        # Gradient w.r.t. bias
        if self.has_bias:
            grad_bias = np_backend.sum(grad_output.data, axis=(0, 2, 3))
        else:
            grad_bias = None
        
        # Compute gradients
        for b in range(batch_size):
            for oc in range(out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride[0]
                        w_start = j * self.stride[1]
                        h_end = h_start + k_h
                        w_end = w_start + k_w
                        
                        # Gradient w.r.t. input
                        grad_x_padded[b, :, h_start:h_end, w_start:w_end] += \
                            weight.data[oc] * grad_output.data[b, oc, i, j]
                        
                        # Gradient w.r.t. weight
                        patch = x_padded[b, :, h_start:h_end, w_start:w_end]
                        grad_weight[oc] += patch * grad_output.data[b, oc, i, j]
        
        # Remove padding from gradient
        if self.padding[0] > 0 or self.padding[1] > 0:
            p_h, p_w = self.padding
            grad_x = grad_x_padded[:, :, p_h:-p_h if p_h > 0 else None, p_w:-p_w if p_w > 0 else None]
        else:
            grad_x = grad_x_padded
        
        return Tensor(grad_x), Tensor(grad_weight), Tensor(grad_bias) if grad_bias is not None else None, None, None


class Conv1dFunction(Function):
    def forward(self, x, weight, bias, stride, padding):
        self.save_for_backward(Tensor(x), Tensor(weight))
        self.stride = stride
        self.padding = padding
        self.has_bias = bias is not None
        
        np_backend, _ = _get_backend(x)
        
        # Apply padding
        if padding > 0:
            x_padded = np_backend.pad(x, ((0, 0), (0, 0), (padding, padding)), mode='constant')
        else:
            x_padded = x
        
        batch_size, in_channels, length = x_padded.shape
        out_channels, _, kernel_size = weight.shape
        
        out_length = (length - kernel_size) // stride + 1
        output = np_backend.zeros((batch_size, out_channels, out_length), dtype=x.dtype)
        
        # Perform convolution
        for b in range(batch_size):
            for oc in range(out_channels):
                for i in range(out_length):
                    start = i * stride
                    end = start + kernel_size
                    patch = x_padded[b, :, start:end]
                    output[b, oc, i] = np_backend.sum(patch * weight[oc])
        
        if bias is not None:
            output += bias.reshape(1, -1, 1)
        
        return output
    
    def backward(self, grad_output):
        x, weight = self.saved_tensors
        np_backend, _ = _get_backend(grad_output.data)
        
        if self.padding > 0:
            x_padded = np_backend.pad(x.data, ((0, 0), (0, 0), (self.padding, self.padding)), mode='constant')
        else:
            x_padded = x.data
        
        batch_size, in_channels, length = x_padded.shape
        out_channels, _, kernel_size = weight.shape
        _, _, out_length = grad_output.data.shape
        
        grad_x_padded = np_backend.zeros_like(x_padded)
        grad_weight = np_backend.zeros_like(weight.data)
        
        if self.has_bias:
            grad_bias = np_backend.sum(grad_output.data, axis=(0, 2))
        else:
            grad_bias = None
        
        for b in range(batch_size):
            for oc in range(out_channels):
                for i in range(out_length):
                    start = i * self.stride
                    end = start + kernel_size
                    
                    grad_x_padded[b, :, start:end] += weight.data[oc] * grad_output.data[b, oc, i]
                    patch = x_padded[b, :, start:end]
                    grad_weight[oc] += patch * grad_output.data[b, oc, i]
        
        if self.padding > 0:
            grad_x = grad_x_padded[:, :, self.padding:-self.padding]
        else:
            grad_x = grad_x_padded
        
        return Tensor(grad_x), Tensor(grad_weight), Tensor(grad_bias) if grad_bias is not None else None, None, None


class MaxPool2dFunction(Function):
    def forward(self, x, kernel_size, stride, padding):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        np_backend, _ = _get_backend(x)
        
        # Apply padding
        if padding[0] > 0 or padding[1] > 0:
            x_padded = np_backend.pad(
                x,
                ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
                mode='constant',
                constant_values=-np_backend.inf
            )
        else:
            x_padded = x
        
        batch_size, channels, in_h, in_w = x_padded.shape
        k_h, k_w = kernel_size
        s_h, s_w = stride
        
        out_h = (in_h - k_h) // s_h + 1
        out_w = (in_w - k_w) // s_w + 1
        
        output = np_backend.zeros((batch_size, channels, out_h, out_w), dtype=x.dtype)
        self.max_indices = np_backend.zeros((batch_size, channels, out_h, out_w, 2), dtype=np_backend.int32)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * s_h
                        w_start = j * s_w
                        h_end = h_start + k_h
                        w_end = w_start + k_w
                        
                        patch = x_padded[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, i, j] = np_backend.max(patch)
                        
                        # Store max index for backward
                        max_idx = np_backend.unravel_index(np_backend.argmax(patch), patch.shape)
                        self.max_indices[b, c, i, j] = [max_idx[0], max_idx[1]]
        
        self.input_shape = x.shape
        self.save_for_backward(Tensor(x_padded))
        
        return output
    
    def backward(self, grad_output):
        x_padded, = self.saved_tensors
        np_backend, _ = _get_backend(grad_output.data)
        
        grad_x_padded = np_backend.zeros_like(x_padded.data)
        batch_size, channels, out_h, out_w = grad_output.data.shape
        s_h, s_w = self.stride
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * s_h
                        w_start = j * s_w
                        
                        max_i, max_j = self.max_indices[b, c, i, j]
                        grad_x_padded[b, c, h_start + max_i, w_start + max_j] += grad_output.data[b, c, i, j]
        
        # Remove padding
        if self.padding[0] > 0 or self.padding[1] > 0:
            p_h, p_w = self.padding
            grad_x = grad_x_padded[:, :, p_h:-p_h if p_h > 0 else None, p_w:-p_w if p_w > 0 else None]
        else:
            grad_x = grad_x_padded
        
        return Tensor(grad_x), None, None, None


class AvgPool2dFunction(Function):
    def forward(self, x, kernel_size, stride, padding):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_shape = x.shape
        
        np_backend, _ = _get_backend(x)
        
        if padding[0] > 0 or padding[1] > 0:
            x_padded = np_backend.pad(
                x,
                ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
                mode='constant'
            )
        else:
            x_padded = x
        
        batch_size, channels, in_h, in_w = x_padded.shape
        k_h, k_w = kernel_size
        s_h, s_w = stride
        
        out_h = (in_h - k_h) // s_h + 1
        out_w = (in_w - k_w) // s_w + 1
        
        output = np_backend.zeros((batch_size, channels, out_h, out_w), dtype=x.dtype)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * s_h
                        w_start = j * s_w
                        h_end = h_start + k_h
                        w_end = w_start + k_w
                        
                        patch = x_padded[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, i, j] = np_backend.mean(patch)
        
        return output
    
    def backward(self, grad_output):
        np_backend, _ = _get_backend(grad_output.data)
        
        # Create gradient with padding if needed
        batch_size, channels, in_h, in_w = self.input_shape
        
        if self.padding[0] > 0 or self.padding[1] > 0:
            p_h, p_w = self.padding
            padded_h = in_h + 2 * p_h
            padded_w = in_w + 2 * p_w
            grad_x_padded = np_backend.zeros((batch_size, channels, padded_h, padded_w), dtype=grad_output.data.dtype)
        else:
            grad_x_padded = np_backend.zeros(self.input_shape, dtype=grad_output.data.dtype)
        
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        _, _, out_h, out_w = grad_output.data.shape
        
        # Distribute gradient evenly across the pooling window
        scale = 1.0 / (k_h * k_w)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * s_h
                        w_start = j * s_w
                        h_end = h_start + k_h
                        w_end = w_start + k_w
                        
                        # Distribute gradient evenly to all elements in the window
                        grad_x_padded[b, c, h_start:h_end, w_start:w_end] += \
                            grad_output.data[b, c, i, j] * scale
        
        # Remove padding if present
        if self.padding[0] > 0 or self.padding[1] > 0:
            p_h, p_w = self.padding
            grad_x = grad_x_padded[:, :, p_h:-p_h if p_h > 0 else None, p_w:-p_w if p_w > 0 else None]
        else:
            grad_x = grad_x_padded
        
        return Tensor(grad_x), None, None, None
    
class Sigmoid(Function):
    def forward(self, x):
        np_backend, _ = _get_backend(x)
        sigmoid_x = 1 / (1 + np_backend.exp(-x))
        self.save_for_backward(Tensor(sigmoid_x))
        return sigmoid_x
    
    def backward(self, grad_output):
        if grad_output is None:
            return None
        sigmoid_x, = self.saved_tensors
        # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        grad = grad_output.data * sigmoid_x.data * (1 - sigmoid_x.data)
        return Tensor(grad)


class Tanh(Function):
    def forward(self, x):
        np_backend, _ = _get_backend(x)
        tanh_x = np_backend.tanh(x)
        self.save_for_backward(Tensor(tanh_x))
        return tanh_x
    
    def backward(self, grad_output):
        if grad_output is None:
            return None
        tanh_x, = self.saved_tensors
        # d/dx tanh(x) = 1 - tanh(x)^2
        grad = grad_output.data * (1 - tanh_x.data ** 2)
        return Tensor(grad)



    