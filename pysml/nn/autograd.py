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