"""
PySML Engine - Enhanced Operations with Full Automatic Differentiation
Complete gradient computation for all operations
"""

from typing import Optional, Union, List, Tuple
from .tensor import Tensor

# Backend management (keep existing code)
_current_backend = None
_current_device = "cpu"
_backends = {}

def _import_backends():
    """Import all available backends"""
    global _backends
    
    try:
        from . import cpu
        _backends['cpu'] = cpu.backend
    except ImportError:
        import numpy as np
        _backends['cpu'] = np
    
    try:
        from . import xpu
        if xpu.backend.AVAILABLE:
            _backends['xpu'] = xpu.backend
    except (ImportError, AttributeError):
        pass
    
    try:
        from . import cuda
        if cuda.backend.AVAILABLE:
            _backends['cuda'] = cuda.backend
    except (ImportError, AttributeError):
        pass

_import_backends()
_current_backend = _backends.get('cpu')

def set_device(device: str):
    global _current_backend, _current_device
    device_type = device.split(':')[0]
    if device_type not in _backends:
        raise ValueError(f"Device '{device}' not available")
    _current_backend = _backends[device_type]
    _current_device = device

def get_device():
    return _current_device

def get_backend():
    return _current_backend

def get_default_device():
    return _current_device

def get_available_devices():
    devices = []
    for device_type, backend in _backends.items():
        if hasattr(backend, 'get_available_devices'):
            devices.extend(backend.get_available_devices())
        else:
            devices.append(device_type)
    return devices

def to_device(tensor: Tensor, device: str) -> Tensor:
    """
    Move tensor to different device
    
    Args:
        tensor: Input tensor
        device: Target device
    
    Returns:
        Tensor on new device
    """
    device_type = device.split(':')[0]
    
    if device_type not in _backends:
        raise ValueError(f"Device '{device}' not available")
    
    target_backend = _backends[device_type]
    
    # Convert to numpy first, then to target backend
    numpy_data = tensor.numpy()
    new_data = target_backend.array(numpy_data)
    
    return Tensor(new_data, backend=target_backend, device=device, 
                 requires_grad=tensor.requires_grad)

def _ensure_tensor(x):
    """Convert input to Tensor if it isn't already"""
    if isinstance(x, Tensor):
        return x
    return Tensor(x, backend=_current_backend, device=_current_device)

def _priority_backend(a, b, priority_list=["pysml.cpu.backend", "pysml.xpu.backend", "pysml.cuda.backend"]):
    a_backend = a.backend
    b_backend = b.backend
    a_score = priority_list.index(a_backend.__name__)
    b_score = priority_list.index(b_backend.__name__)
    backends = [a_backend, b_backend]
    return backends[1 if b_score > a_score else 0]

def _tensor_backend(a, b, priority_list=["pysml.cpu.backend", "pysml.xpu.backend", "pysml.cuda.backend"]):
   backend = _priority_backend(a,b,priority_list=priority_list)
   backend_name = backend.__name__.split(".")[1]
   return a, b, backend


# ===== Creation Operations (keep existing) =====

def zeros(*shape, dtype=None, device: Optional[str] = None, requires_grad: bool = False) -> Tensor:
    device = device or _current_device
    backend = _backends[device.split(':')[0]]
    dtype = dtype or backend.float32
    data = backend.zeros(shape if len(shape) > 1 else shape[0], dtype=dtype)
    return Tensor(data, backend=backend, device=device, requires_grad=requires_grad)

def ones(*shape, dtype=None, device: Optional[str] = None, requires_grad: bool = False) -> Tensor:
    device = device or _current_device
    backend = _backends[device.split(':')[0]]
    dtype = dtype or backend.float32
    data = backend.ones(shape if len(shape) > 1 else shape[0], dtype=dtype)
    return Tensor(data, backend=backend, device=device, requires_grad=requires_grad)

def randn(*shape, dtype=None, device: Optional[str] = None, requires_grad: bool = False) -> Tensor:
    device = device or _current_device
    backend = _backends[device.split(':')[0]]
    dtype = dtype or backend.float32
    data = backend.random.randn(*shape).astype(dtype)
    return Tensor(data, backend=backend, device=device, requires_grad=requires_grad)

# ===== Arithmetic Operations with Full Autograd =====

def add(a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
    """Element-wise addition with full autograd support"""
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)
    
    backend = a.backend
    out = Tensor(backend.add(a.data, b.data), backend=backend, device=a.device,
                requires_grad=a.requires_grad or b.requires_grad)
    
    if out.requires_grad:
        out._prev = {a, b} if a.requires_grad and b.requires_grad else ({a} if a.requires_grad else {b})
        out._op = 'add'
        
        def _backward():
            if a.requires_grad:
                grad_a = out.grad.data
                # Handle broadcasting
                if a.shape != out.shape:
                    # Sum over broadcasted dimensions
                    ndims_added = len(out.shape) - len(a.shape)
                    for i in range(ndims_added):
                        grad_a = backend.sum(grad_a, axis=0)
                    for i, (dim_a, dim_out) in enumerate(zip(a.shape, out.shape)):
                        if dim_a == 1 and dim_out > 1:
                            grad_a = backend.sum(grad_a, axis=i, keepdims=True)
                
                if a.grad is None:
                    a.grad = Tensor(grad_a, backend=backend)
                else:
                    a.grad.data = a.grad.data + grad_a
            
            if b.requires_grad:
                grad_b = out.grad.data
                if b.shape != out.shape:
                    ndims_added = len(out.shape) - len(b.shape)
                    for i in range(ndims_added):
                        grad_b = backend.sum(grad_b, axis=0)
                    for i, (dim_b, dim_out) in enumerate(zip(b.shape, out.shape)):
                        if dim_b == 1 and dim_out > 1:
                            grad_b = backend.sum(grad_b, axis=i, keepdims=True)
                
                if b.grad is None:
                    b.grad = Tensor(grad_b, backend=backend)
                else:
                    b.grad.data = b.grad.data + grad_b
        
        out._backward = _backward
    
    return out


def subtract(a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
    """Element-wise subtraction with full autograd"""
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)
    
    backend = a.backend
    out = Tensor(backend.subtract(a.data, b.data), backend=backend, device=a.device,
                requires_grad=a.requires_grad or b.requires_grad)
    
    if out.requires_grad:
        out._prev = {a, b} if a.requires_grad and b.requires_grad else ({a} if a.requires_grad else {b})
        out._op = 'subtract'
        
        def _backward():
            if a.requires_grad:
                grad_a = out.grad.data
                if a.shape != out.shape:
                    ndims_added = len(out.shape) - len(a.shape)
                    for i in range(ndims_added):
                        grad_a = backend.sum(grad_a, axis=0)
                    for i, (dim_a, dim_out) in enumerate(zip(a.shape, out.shape)):
                        if dim_a == 1 and dim_out > 1:
                            grad_a = backend.sum(grad_a, axis=i, keepdims=True)
                
                if a.grad is None:
                    a.grad = Tensor(grad_a, backend=backend)
                else:
                    a.grad.data = a.grad.data + grad_a
            
            if b.requires_grad:
                grad_b = backend.negative(out.grad.data)
                if b.shape != out.shape:
                    ndims_added = len(out.shape) - len(b.shape)
                    for i in range(ndims_added):
                        grad_b = backend.sum(grad_b, axis=0)
                    for i, (dim_b, dim_out) in enumerate(zip(b.shape, out.shape)):
                        if dim_b == 1 and dim_out > 1:
                            grad_b = backend.sum(grad_b, axis=i, keepdims=True)
                
                if b.grad is None:
                    b.grad = Tensor(grad_b, backend=backend)
                else:
                    b.grad.data = b.grad.data + grad_b
        
        out._backward = _backward
    
    return out


def multiply(a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
    """Element-wise multiplication with full autograd"""
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)
    
    a, b, backend = _tensor_backend(a, b)
    out = Tensor(backend.multiply(a.data, b.data), backend=backend, device=a.device,
                requires_grad=a.requires_grad or b.requires_grad)
    
    if out.requires_grad:
        out._prev = {a, b} if a.requires_grad and b.requires_grad else ({a} if a.requires_grad else {b})
        out._op = 'multiply'
        
        def _backward():
            if a.requires_grad:
                grad_a = backend.multiply(out.grad.data, b.data)
                if a.shape != out.shape:
                    ndims_added = len(out.shape) - len(a.shape)
                    for i in range(ndims_added):
                        grad_a = backend.sum(grad_a, axis=0)
                    for i, (dim_a, dim_out) in enumerate(zip(a.shape, out.shape)):
                        if dim_a == 1 and dim_out > 1:
                            grad_a = backend.sum(grad_a, axis=i, keepdims=True)
                
                if a.grad is None:
                    a.grad = Tensor(grad_a, backend=backend)
                else:
                    a.grad.data = a.grad.data + grad_a
            
            if b.requires_grad:
                grad_b = backend.multiply(out.grad.data, a.data)
                if b.shape != out.shape:
                    ndims_added = len(out.shape) - len(b.shape)
                    for i in range(ndims_added):
                        grad_b = backend.sum(grad_b, axis=0)
                    for i, (dim_b, dim_out) in enumerate(zip(b.shape, out.shape)):
                        if dim_b == 1 and dim_out > 1:
                            grad_b = backend.sum(grad_b, axis=i, keepdims=True)
                
                if b.grad is None:
                    b.grad = Tensor(grad_b, backend=backend)
                else:
                    b.grad.data = b.grad.data + grad_b
        
        out._backward = _backward
    
    return out


def divide(a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
    """Element-wise division with full autograd"""
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)
    
    backend = a.backend
    out = Tensor(backend.divide(a.data, b.data), backend=backend, device=a.device,
                requires_grad=a.requires_grad or b.requires_grad)
    
    if out.requires_grad:
        out._prev = {a, b} if a.requires_grad and b.requires_grad else ({a} if a.requires_grad else {b})
        out._op = 'divide'
        
        def _backward():
            if a.requires_grad:
                grad_a = backend.divide(out.grad.data, b.data)
                if a.shape != out.shape:
                    ndims_added = len(out.shape) - len(a.shape)
                    for i in range(ndims_added):
                        grad_a = backend.sum(grad_a, axis=0)
                    for i, (dim_a, dim_out) in enumerate(zip(a.shape, out.shape)):
                        if dim_a == 1 and dim_out > 1:
                            grad_a = backend.sum(grad_a, axis=i, keepdims=True)
                
                if a.grad is None:
                    a.grad = Tensor(grad_a, backend=backend)
                else:
                    a.grad.data = a.grad.data + grad_a
            
            if b.requires_grad:
                grad_b = backend.multiply(
                    backend.negative(out.grad.data),
                    backend.divide(a.data, backend.power(b.data, 2))
                )
                if b.shape != out.shape:
                    ndims_added = len(out.shape) - len(b.shape)
                    for i in range(ndims_added):
                        grad_b = backend.sum(grad_b, axis=0)
                    for i, (dim_b, dim_out) in enumerate(zip(b.shape, out.shape)):
                        if dim_b == 1 and dim_out > 1:
                            grad_b = backend.sum(grad_b, axis=i, keepdims=True)
                
                if b.grad is None:
                    b.grad = Tensor(grad_b, backend=backend)
                else:
                    b.grad.data = b.grad.data + grad_b
        
        out._backward = _backward
    
    return out


def power(a: Union[Tensor, float], exponent: Union[Tensor, float]) -> Tensor:
    """Element-wise power with full autograd"""
    a = _ensure_tensor(a)
    exponent = _ensure_tensor(exponent)
    
    backend = a.backend
    out = Tensor(backend.power(a.data, exponent.data), backend=backend, device=a.device,
                requires_grad=a.requires_grad or exponent.requires_grad)
    
    if out.requires_grad:
        out._prev = {a, exponent} if a.requires_grad and exponent.requires_grad else ({a} if a.requires_grad else {exponent})
        out._op = 'power'
        
        def _backward():
            if a.requires_grad:
                # d/da (a^b) = b * a^(b-1)
                grad_a = backend.multiply(
                    out.grad.data,
                    backend.multiply(
                        exponent.data,
                        backend.power(a.data, backend.subtract(exponent.data, 1))
                    )
                )
                
                if a.grad is None:
                    a.grad = Tensor(grad_a, backend=backend)
                else:
                    a.grad.data = a.grad.data + grad_a
            
            if exponent.requires_grad:
                # d/db (a^b) = a^b * ln(a)
                grad_exp = backend.multiply(
                    out.grad.data,
                    backend.multiply(
                        backend.power(a.data, exponent.data),
                        backend.log(backend.add(a.data, 1e-10))
                    )
                )
                
                if exponent.grad is None:
                    exponent.grad = Tensor(grad_exp, backend=backend)
                else:
                    exponent.grad.data = exponent.grad.data + grad_exp
        
        out._backward = _backward
    
    return out


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication with full autograd"""
    backend = a.backend
    out = Tensor(backend.matmul(a.data, b.data), backend=backend, device=a.device,
                requires_grad=a.requires_grad or b.requires_grad)
    
    if out.requires_grad:
        out._prev = {a, b} if a.requires_grad and b.requires_grad else ({a} if a.requires_grad else {b})
        out._op = 'matmul'
        
        def _backward():
            if a.requires_grad:
                # grad_a = grad_out @ b.T
                grad_a = backend.matmul(out.grad.data, backend.swapaxes(b.data, -2, -1))
                
                if a.grad is None:
                    a.grad = Tensor(grad_a, backend=backend)
                else:
                    a.grad.data = a.grad.data + grad_a
            
            if b.requires_grad:
                # grad_b = a.T @ grad_out
                grad_b = backend.matmul(backend.swapaxes(a.data, -2, -1), out.grad.data)
                
                if b.grad is None:
                    b.grad = Tensor(grad_b, backend=backend)
                else:
                    b.grad.data = b.grad.data + grad_b
        
        out._backward = _backward
    
    return out


def negative(a: Tensor) -> Tensor:
    """Numerical negative with autograd"""
    backend = a.backend
    out = Tensor(backend.negative(a.data), backend=backend, device=a.device, 
                requires_grad=a.requires_grad)
    
    if out.requires_grad:
        out._prev = {a}
        out._op = 'negative'
        
        def _backward():
            if a.requires_grad:
                grad_a = backend.negative(out.grad.data)
                if a.grad is None:
                    a.grad = Tensor(grad_a, backend=backend)
                else:
                    a.grad.data = a.grad.data + grad_a
        
        out._backward = _backward
    
    return out


def positive(a: Tensor) -> Tensor:
    """Numerical positive with autograd"""
    backend = a.backend
    out = Tensor(backend.positive(a.data), backend=backend, device=a.device, 
                requires_grad=a.requires_grad)
    
    if out.requires_grad:
        out._prev = {a}
        out._op = 'positive'
        
        def _backward():
            if a.requires_grad:
                if a.grad is None:
                    a.grad = out.grad
                else:
                    a.grad.data = a.grad.data + out.grad.data
        
        out._backward = _backward
    
    return out


# ===== Reduction Operations with Autograd =====

def sum(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    """Sum along axis with full autograd"""
    backend = a.backend
    out = Tensor(backend.sum(a.data, axis=axis, keepdims=keepdims), backend=backend, 
                device=a.device, requires_grad=a.requires_grad)
    
    if out.requires_grad:
        out._prev = {a}
        out._op = 'sum'
        
        def _backward():
            if a.requires_grad:
                grad_data = out.grad.data
                
                # MEMORY FIX: Don't create huge zero arrays for broadcasting!
                # Instead, use backend's broadcast_to or repeat/tile operations
                if not keepdims and axis is not None:
                    # Expand dimensions that were reduced
                    if isinstance(axis, int):
                        grad_data = backend.expand_dims(grad_data, axis=axis)
                    else:
                        for ax in sorted(axis):
                            grad_data = backend.expand_dims(grad_data, axis=ax)
                
                # CRITICAL FIX: Use broadcast_to if available, otherwise tile/repeat
                # This avoids allocating a full zero array
                if hasattr(backend, 'broadcast_to'):
                    # Most efficient - just creates a view
                    grad_data = backend.broadcast_to(grad_data, a.shape)
                elif hasattr(backend, 'tile'):
                    # Calculate tile repeats needed for each dimension
                    repeats = []
                    for dim_a, dim_grad in zip(a.shape, grad_data.shape):
                        repeats.append(dim_a // dim_grad if dim_grad != 0 else 1)
                    grad_data = backend.tile(grad_data, repeats)
                else:
                    # Fallback: use ones and multiply (still better than add with zeros)
                    ones = backend.ones(a.shape, dtype=grad_data.dtype)
                    grad_data = backend.multiply(grad_data, ones)
                
                if a.grad is None:
                    a.grad = Tensor(grad_data, backend=backend)
                else:
                    a.grad.data = a.grad.data + grad_data
        
        out._backward = _backward
    
    return out


def mean(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    """Mean along axis with full autograd"""
    backend = a.backend
    out = Tensor(backend.mean(a.data, axis=axis, keepdims=keepdims), backend=backend, 
                device=a.device, requires_grad=a.requires_grad)
    
    if out.requires_grad:
        out._prev = {a}
        out._op = 'mean'
        
        def _backward():
            if a.requires_grad:
                grad_data = out.grad.data
                if not keepdims and axis is not None:
                    if isinstance(axis, int):
                        grad_data = backend.expand_dims(grad_data, axis=axis)
                    else:
                        for ax in sorted(axis):
                            grad_data = backend.expand_dims(grad_data, axis=ax)
                
                # Calculate number of elements that were averaged
                if axis is None:
                    n = a.size
                elif isinstance(axis, int):
                    n = a.shape[axis]
                else:
                    n = 1
                    for ax in axis:
                        n *= a.shape[ax]
                
                # MEMORY FIX: Broadcast without creating huge zero arrays
                if hasattr(backend, 'broadcast_to'):
                    grad_data = backend.broadcast_to(grad_data, a.shape)
                elif hasattr(backend, 'tile'):
                    repeats = []
                    for dim_a, dim_grad in zip(a.shape, grad_data.shape):
                        repeats.append(dim_a // dim_grad if dim_grad != 0 else 1)
                    grad_data = backend.tile(grad_data, repeats)
                else:
                    ones = backend.ones(a.shape, dtype=grad_data.dtype)
                    grad_data = backend.multiply(grad_data, ones)
                
                grad_data = backend.divide(grad_data, n)
                
                if a.grad is None:
                    a.grad = Tensor(grad_data, backend=backend)
                else:
                    a.grad.data = a.grad.data + grad_data
        
        out._backward = _backward
    
    return out


# ===== Activation Functions with Full Autograd =====

def relu(a: Tensor) -> Tensor:
    """ReLU activation with full autograd"""
    backend = a.backend
    out = Tensor(backend.maximum(a.data, 0), backend=backend, device=a.device, 
                requires_grad=a.requires_grad)
    
    if out.requires_grad:
        out._prev = {a}
        out._op = 'relu'
        
        def _backward():
            if a.requires_grad:
                # Gradient is 1 where input > 0, else 0
                grad_a = backend.multiply(
                    out.grad.data,
                    backend.greater(a.data, 0).astype(out.grad.data.dtype)
                )
                
                if a.grad is None:
                    a.grad = Tensor(grad_a, backend=backend)
                else:
                    a.grad.data = a.grad.data + grad_a
        
        out._backward = _backward
    
    return out


def sigmoid(a: Tensor) -> Tensor:
    """Sigmoid activation with full autograd"""
    backend = a.backend
    sig = backend.divide(1, backend.add(1, backend.exp(backend.negative(a.data))))
    out = Tensor(sig, backend=backend, device=a.device, requires_grad=a.requires_grad)
    
    if out.requires_grad:
        out._prev = {a}
        out._op = 'sigmoid'
        
        def _backward():
            if a.requires_grad:
                # Gradient: sigmoid(x) * (1 - sigmoid(x))
                grad_a = backend.multiply(
                    out.grad.data,
                    backend.multiply(sig, backend.subtract(1, sig))
                )
                
                if a.grad is None:
                    a.grad = Tensor(grad_a, backend=backend)
                else:
                    a.grad.data = a.grad.data + grad_a
        
        out._backward = _backward
    
    return out


def tanh(a: Tensor) -> Tensor:
    """Tanh activation with full autograd"""
    backend = a.backend
    tanh_val = backend.tanh(a.data)
    out = Tensor(tanh_val, backend=backend, device=a.device, requires_grad=a.requires_grad)
    
    if out.requires_grad:
        out._prev = {a}
        out._op = 'tanh'
        
        def _backward():
            if a.requires_grad:
                # Gradient: 1 - tanh(x)^2
                grad_a = backend.multiply(
                    out.grad.data,
                    backend.subtract(1, backend.power(tanh_val, 2))
                )
                
                if a.grad is None:
                    a.grad = Tensor(grad_a, backend=backend)
                else:
                    a.grad.data = a.grad.data + grad_a
        
        out._backward = _backward
    
    return out


def softmax(a: Tensor, axis: int = -1) -> Tensor:
    """Softmax along axis with full autograd"""
    backend = a.backend
    exp_a = backend.exp(backend.subtract(a.data, backend.max(a.data, axis=axis, keepdims=True)))
    softmax_val = backend.divide(exp_a, backend.sum(exp_a, axis=axis, keepdims=True))
    out = Tensor(softmax_val, backend=backend, device=a.device, requires_grad=a.requires_grad)
    
    if out.requires_grad:
        out._prev = {a}
        out._op = 'softmax'
        
        def _backward():
            if a.requires_grad:
                # Gradient: softmax(x) * (grad - sum(grad * softmax(x)))
                s = softmax_val
                sum_term = backend.sum(
                    backend.multiply(out.grad.data, s),
                    axis=axis,
                    keepdims=True
                )
                grad_a = backend.multiply(
                    s,
                    backend.subtract(out.grad.data, sum_term)
                )
                
                if a.grad is None:
                    a.grad = Tensor(grad_a, backend=backend)
                else:
                    a.grad.data = a.grad.data + grad_a
        
        out._backward = _backward
    
    return out


def gelu(a: Tensor) -> Tensor:
    """GELU activation with full autograd"""
    backend = a.backend
    import math
    
    sqrt_2_pi = math.sqrt(2 / math.pi)
    x = a.data
    
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x_cubed = backend.power(x, 3)
    tanh_arg = backend.multiply(sqrt_2_pi, backend.add(x, backend.multiply(0.044715, x_cubed)))
    tanh_val = backend.tanh(tanh_arg)
    gelu_val = backend.multiply(0.5, backend.multiply(x, backend.add(1, tanh_val)))
    
    out = Tensor(gelu_val, backend=backend, device=a.device, requires_grad=a.requires_grad)
    
    if out.requires_grad:
        out._prev = {a}
        out._op = 'gelu'
        
        def _backward():
            if a.requires_grad:
                # Approximate gradient
                sech2 = backend.subtract(1, backend.power(tanh_val, 2))
                term1 = backend.multiply(0.5, backend.add(1, tanh_val))
                term2 = backend.multiply(
                    0.5 * sqrt_2_pi,
                    backend.multiply(
                        x,
                        backend.multiply(
                            sech2,
                            backend.add(1, backend.multiply(3 * 0.044715, backend.power(x, 2)))
                        )
                    )
                )
                grad_a = backend.multiply(out.grad.data, backend.add(term1, term2))
                
                if a.grad is None:
                    a.grad = Tensor(grad_a, backend=backend)
                else:
                    a.grad.data = a.grad.data + grad_a
        
        out._backward = _backward
    
    return out


# ===== Mathematical Functions with Full Autograd =====

def exp(a: Tensor) -> Tensor:
    """Exponential with full autograd"""
    backend = a.backend
    exp_val = backend.exp(a.data)
    out = Tensor(exp_val, backend=backend, device=a.device, requires_grad=a.requires_grad)
    
    if out.requires_grad:
        out._prev = {a}
        out._op = 'exp'
        
        def _backward():
            if a.requires_grad:
                # Gradient: exp(x)
                grad_a = backend.multiply(out.grad.data, exp_val)
                
                if a.grad is None:
                    a.grad = Tensor(grad_a, backend=backend)
                else:
                    a.grad.data = a.grad.data + grad_a
        
        out._backward = _backward
    
    return out


def log(a: Tensor) -> Tensor:
    """Natural logarithm with full autograd"""
    backend = a.backend
    out = Tensor(backend.log(a.data), backend=backend, device=a.device, requires_grad=a.requires_grad)
    
    if out.requires_grad:
        out._prev = {a}
        out._op = 'log'
        
        def _backward():
            if a.requires_grad:
                # Gradient: 1/x
                grad_a = backend.divide(out.grad.data, a.data)
                
                if a.grad is None:
                    a.grad = Tensor(grad_a, backend=backend)
                else:
                    a.grad.data = a.grad.data + grad_a
        
        out._backward = _backward
    
    return out


def sqrt(a: Tensor) -> Tensor:
    """Square root with full autograd"""
    backend = a.backend
    sqrt_val = backend.sqrt(a.data)
    out = Tensor(sqrt_val, backend=backend, device=a.device, requires_grad=a.requires_grad)
    
    if out.requires_grad:
        out._prev = {a}
        out._op = 'sqrt'
        
        def _backward():
            if a.requires_grad:
                # Gradient: 1 / (2 * sqrt(x))
                grad_a = backend.divide(out.grad.data, backend.multiply(2, sqrt_val))
                
                if a.grad is None:
                    a.grad = Tensor(grad_a, backend=backend)
                else:
                    a.grad.data = a.grad.data + grad_a
        
        out._backward = _backward
    
    return out


def abs(a: Tensor) -> Tensor:
    """Absolute value with full autograd"""
    backend = a.backend
    out = Tensor(backend.abs(a.data), backend=backend, device=a.device, requires_grad=a.requires_grad)
    
    if out.requires_grad:
        out._prev = {a}
        out._op = 'abs'
        
        def _backward():
            if a.requires_grad:
                # Gradient: sign(x)
                grad_a = backend.multiply(out.grad.data, backend.sign(a.data))
                
                if a.grad is None:
                    a.grad = Tensor(grad_a, backend=backend)
                else:
                    a.grad.data = a.grad.data + grad_a
        
        out._backward = _backward
    
    return out


def sin(a: Tensor) -> Tensor:
    """Sine with full autograd"""
    backend = a.backend
    out = Tensor(backend.sin(a.data), backend=backend, device=a.device, requires_grad=a.requires_grad)
    
    if out.requires_grad:
        out._prev = {a}
        out._op = 'sin'
        
        def _backward():
            if a.requires_grad:
                # Gradient: cos(x)
                grad_a = backend.multiply(out.grad.data, backend.cos(a.data))
                
                if a.grad is None:
                    a.grad = Tensor(grad_a, backend=backend)
                else:
                    a.grad.data = a.grad.data + grad_a
        
        out._backward = _backward
    
    return out


def cos(a: Tensor) -> Tensor:
    """Cosine with full autograd"""
    backend = a.backend
    out = Tensor(backend.cos(a.data), backend=backend, device=a.device, requires_grad=a.requires_grad)
    
    if out.requires_grad:
        out._prev = {a}
        out._op = 'cos'
        
        def _backward():
            if a.requires_grad:
                # Gradient: -sin(x)
                grad_a = backend.multiply(out.grad.data, backend.negative(backend.sin(a.data)))
                
                if a.grad is None:
                    a.grad = Tensor(grad_a, backend=backend)
                else:
                    a.grad.data = a.grad.data + grad_a
        
        out._backward = _backward
    
    return out


# ===== Shape Operations with Autograd =====

def reshape(a: Tensor, shape: Tuple[int, ...]) -> Tensor:
    """Reshape tensor with full autograd"""
    backend = a.backend
    out = Tensor(backend.reshape(a.data, shape), backend=backend, device=a.device, 
                requires_grad=a.requires_grad)
    
    if out.requires_grad:
        out._prev = {a}
        out._op = 'reshape'
        
        def _backward():
            if a.requires_grad:
                grad_a = backend.reshape(out.grad.data, a.shape)
                
                if a.grad is None:
                    a.grad = Tensor(grad_a, backend=backend)
                else:
                    a.grad.data = a.grad.data + grad_a
        
        out._backward = _backward
    
    return out


def transpose(a: Tensor, axes: Optional[Tuple[int, ...]] = None) -> Tensor:
    """Transpose tensor with full autograd"""
    backend = a.backend
    out = Tensor(backend.transpose(a.data, axes=axes), backend=backend, device=a.device, 
                requires_grad=a.requires_grad)
    
    if out.requires_grad:
        out._prev = {a}
        out._op = 'transpose'
        
        def _backward():
            if a.requires_grad:
                if axes is None:
                    grad_a = backend.transpose(out.grad.data)
                else:
                    # Compute inverse permutation
                    inv_axes = [0] * len(axes)
                    for i, ax in enumerate(axes):
                        inv_axes[ax] = i
                    grad_a = backend.transpose(out.grad.data, tuple(inv_axes))
                
                if a.grad is None:
                    a.grad = Tensor(grad_a, backend=backend)
                else:
                    a.grad.data = a.grad.data + grad_a
        
        out._backward = _backward
    
    return out


def squeeze(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Tensor:
    """Remove single-dimensional entries with autograd"""
    backend = a.backend
    out = Tensor(backend.squeeze(a.data, axis=axis), backend=backend, device=a.device, 
                requires_grad=a.requires_grad)
    
    if out.requires_grad:
        out._prev = {a}
        out._op = 'squeeze'
        
        def _backward():
            if a.requires_grad:
                grad_a = backend.reshape(out.grad.data, a.shape)
                
                if a.grad is None:
                    a.grad = Tensor(grad_a, backend=backend)
                else:
                    a.grad.data = a.grad.data + grad_a
        
        out._backward = _backward
    
    return out


def unsqueeze(a: Tensor, axis: int) -> Tensor:
    """Add a dimension with autograd"""
    backend = a.backend
    out = Tensor(backend.expand_dims(a.data, axis=axis), backend=backend, device=a.device, 
                requires_grad=a.requires_grad)
    
    if out.requires_grad:
        out._prev = {a}
        out._op = 'unsqueeze'
        
        def _backward():
            if a.requires_grad:
                grad_a = backend.squeeze(out.grad.data, axis=axis)
                
                if a.grad is None:
                    a.grad = Tensor(grad_a, backend=backend)
                else:
                    a.grad.data = a.grad.data + grad_a
        
        out._backward = _backward
    
    return out


def concatenate(tensors: List[Tensor], axis: int = 0) -> Tensor:
    """Concatenate tensors along axis with autograd"""
    backend = tensors[0].backend
    arrays = [t.data for t in tensors]
    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(backend.concatenate(arrays, axis=axis), backend=backend, 
                device=tensors[0].device, requires_grad=requires_grad)
    
    if out.requires_grad:
        out._prev = set(t for t in tensors if t.requires_grad)
        out._op = 'concatenate'
        
        def _backward():
            grad_data = out.grad.data
            
            # Split gradient along concatenation axis
            split_indices = []
            current = 0
            for t in tensors[:-1]:
                current += t.shape[axis]
                split_indices.append(current)
            
            if len(split_indices) > 0:
                grad_splits = backend.split(grad_data, split_indices, axis=axis)
            else:
                grad_splits = [grad_data]
            
            for t, grad_split in zip(tensors, grad_splits):
                if t.requires_grad:
                    if t.grad is None:
                        t.grad = Tensor(grad_split, backend=backend)
                    else:
                        t.grad.data = t.grad.data + grad_split
        
        out._backward = _backward
    
    return out


def stack(tensors: List[Tensor], axis: int = 0) -> Tensor:
    """Stack tensors along new axis with autograd"""
    backend = tensors[0].backend
    arrays = [t.data for t in tensors]
    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(backend.stack(arrays, axis=axis), backend=backend, 
                device=tensors[0].device, requires_grad=requires_grad)
    
    if out.requires_grad:
        out._prev = set(t for t in tensors if t.requires_grad)
        out._op = 'stack'
        
        def _backward():
            grad_splits = backend.split(out.grad.data, len(tensors), axis=axis)
            
            for t, grad_split in zip(tensors, grad_splits):
                if t.requires_grad:
                    grad_split = backend.squeeze(grad_split, axis=axis)
                    if t.grad is None:
                        t.grad = Tensor(grad_split, backend=backend)
                    else:
                        t.grad.data = t.grad.data + grad_split
        
        out._backward = _backward
    
    return out


def flatten(a: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    """Flatten dimensions with autograd"""
    if end_dim < 0:
        end_dim = len(a.shape) + end_dim
    
    new_shape = (
        a.shape[:start_dim] + 
        (int(a.backend.prod(a.backend.array(a.shape[start_dim:end_dim+1]))),) + 
        a.shape[end_dim+1:]
    )
    
    return reshape(a, new_shape)


# ===== Loss Functions =====

def mse_loss(predictions: Tensor, targets: Tensor, reduction: str = 'mean') -> Tensor:
    """Mean Squared Error loss with autograd"""
    diff = subtract(predictions, targets)
    squared = multiply(diff, diff)
    
    if reduction == 'mean':
        return mean(squared)
    elif reduction == 'sum':
        return sum(squared)
    else:
        return squared


def cross_entropy_loss(predictions: Tensor, targets: Tensor, reduction: str = 'mean') -> Tensor:
    """Cross entropy loss with autograd"""
    probs = softmax(predictions, axis=-1)
    
    # Clip for numerical stability
    backend = probs.backend
    probs_clipped = Tensor(
        backend.clip(probs.data, 1e-10, 1.0),
        backend=backend,
        device=probs.device,
        requires_grad=probs.requires_grad
    )
    probs_clipped._prev = probs._prev
    probs_clipped._op = probs._op
    probs_clipped._backward = probs._backward
    
    log_probs = log(probs_clipped)
    loss = negative(sum(multiply(targets, log_probs), axis=-1))
    
    if reduction == 'mean':
        return mean(loss)
    elif reduction == 'sum':
        return sum(loss)
    else:
        return loss


# ===== Utility Functions =====

def clip(a: Tensor, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Tensor:
    """Clip values to range (no gradient through clipping)"""
    backend = a.backend
    return Tensor(backend.clip(a.data, min_val, max_val), backend=backend, device=a.device)


def where(condition: Tensor, x: Union[Tensor, float], y: Union[Tensor, float]) -> Tensor:
    """Select elements from x or y based on condition"""
    x = _ensure_tensor(x)
    y = _ensure_tensor(y)
    backend = x.backend
    return Tensor(backend.where(condition.data, x.data, y.data), backend=backend, device=x.device)


def maximum(a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
    """Element-wise maximum"""
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)
    backend = a.backend
    return Tensor(backend.maximum(a.data, b.data), backend=backend, device=a.device)


def minimum(a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
    """Element-wise minimum"""
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)
    backend = a.backend
    return Tensor(backend.minimum(a.data, b.data), backend=backend, device=a.device)


def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Dropout regularization"""
    if not training or p == 0:
        return x
    
    backend = x.backend
    mask = backend.random.binomial(1, 1 - p, x.shape)
    mask = backend.divide(mask, 1 - p)
    return multiply(x, Tensor(mask, backend=backend, device=x.device))


# ===== Additional Creation Functions =====

def full(shape: Tuple[int, ...], fill_value: float, dtype=None, device: Optional[str] = None, 
         requires_grad: bool = False) -> Tensor:
    """Create tensor filled with value"""
    device = device or _current_device
    backend = _backends[device.split(':')[0]]
    dtype = dtype or backend.float32
    
    data = backend.full(shape, fill_value, dtype=dtype)
    return Tensor(data, backend=backend, device=device, requires_grad=requires_grad)


def eye(n: int, m: Optional[int] = None, dtype=None, device: Optional[str] = None, 
        requires_grad: bool = False) -> Tensor:
    """Create identity matrix"""
    device = device or _current_device
    backend = _backends[device.split(':')[0]]
    dtype = dtype or backend.float32
    
    data = backend.eye(n, m, dtype=dtype)
    return Tensor(data, backend=backend, device=device, requires_grad=requires_grad)


def arange(start, stop=None, step=1, dtype=None, device: Optional[str] = None, 
          requires_grad: bool = False) -> Tensor:
    """Create tensor with evenly spaced values"""
    device = device or _current_device
    backend = _backends[device.split(':')[0]]
    
    if stop is None:
        stop = start
        start = 0
    
    data = backend.arange(start, stop, step, dtype=dtype)
    return Tensor(data, backend=backend, device=device, requires_grad=requires_grad)


def linspace(start: float, stop: float, num: int = 50, dtype=None, device: Optional[str] = None, 
            requires_grad: bool = False) -> Tensor:
    """Create tensor with evenly spaced values over interval"""
    device = device or _current_device
    backend = _backends[device.split(':')[0]]
    dtype = dtype or backend.float32
    
    data = backend.linspace(start, stop, num, dtype=dtype)
    return Tensor(data, backend=backend, device=device, requires_grad=requires_grad)


def rand(*shape, dtype=None, device: Optional[str] = None, requires_grad: bool = False) -> Tensor:
    """Create tensor with random uniform distribution [0, 1)"""
    device = device or _current_device
    backend = _backends[device.split(':')[0]]
    dtype = dtype or backend.float32
    
    data = backend.random.rand(*shape).astype(dtype)
    return Tensor(data, backend=backend, device=device, requires_grad=requires_grad)


def randint(low: int, high: int, shape: Tuple[int, ...], dtype=None, device: Optional[str] = None, 
           requires_grad: bool = False) -> Tensor:
    """Create tensor with random integers"""
    device = device or _current_device
    backend = _backends[device.split(':')[0]]
    dtype = dtype or backend.int32
    
    data = backend.random.randint(low, high, shape, dtype=dtype)
    return Tensor(data, backend=backend, device=device, requires_grad=requires_grad)


# ===== Additional Operations =====

def max(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    """Max along axis"""
    backend = a.backend
    return Tensor(backend.max(a.data, axis=axis, keepdims=keepdims), backend=backend, device=a.device)


def min(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    """Min along axis"""
    backend = a.backend
    return Tensor(backend.min(a.data, axis=axis, keepdims=keepdims), backend=backend, device=a.device)


def var(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    """Variance along axis"""
    backend = a.backend
    return Tensor(backend.var(a.data, axis=axis, keepdims=keepdims), backend=backend, device=a.device)


def std(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    """Standard deviation along axis"""
    backend = a.backend
    return Tensor(backend.std(a.data, axis=axis, keepdims=keepdims), backend=backend, device=a.device)


def dot(a: Tensor, b: Tensor) -> Tensor:
    """Dot product"""
    backend = a.backend
    return Tensor(backend.dot(a.data, b.data), backend=backend, device=a.device)


def outer(a: Tensor, b: Tensor) -> Tensor:
    """Outer product"""
    backend = a.backend
    return Tensor(backend.outer(a.data, b.data), backend=backend, device=a.device)


def inner(a: Tensor, b: Tensor) -> Tensor:
    """Inner product"""
    backend = a.backend
    return Tensor(backend.inner(a.data, b.data), backend=backend, device=a.device)


def einsum(subscripts: str, *operands) -> Tensor:
    """Einstein summation convention"""
    tensors = [_ensure_tensor(op) for op in operands]
    backend = tensors[0].backend
    arrays = [t.data for t in tensors]
    return Tensor(backend.einsum(subscripts, *arrays), backend=backend, device=tensors[0].device)


def equal(a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
    """Element-wise equality"""
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)
    backend = a.backend
    return Tensor(backend.equal(a.data, b.data), backend=backend, device=a.device)


def not_equal(a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
    """Element-wise inequality"""
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)
    backend = a.backend
    return Tensor(backend.not_equal(a.data, b.data), backend=backend, device=a.device)


def less(a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
    """Element-wise less than"""
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)
    backend = a.backend
    return Tensor(backend.less(a.data, b.data), backend=backend, device=a.device)


def less_equal(a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
    """Element-wise less than or equal"""
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)
    backend = a.backend
    return Tensor(backend.less_equal(a.data, b.data), backend=backend, device=a.device)


def greater(a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
    """Element-wise greater than"""
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)
    backend = a.backend
    return Tensor(backend.greater(a.data, b.data), backend=backend, device=a.device)


def greater_equal(a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
    """Element-wise greater than or equal"""
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)
    backend = a.backend
    return Tensor(backend.greater_equal(a.data, b.data), backend=backend, device=a.device)


def binary_cross_entropy_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Binary cross entropy loss"""
    pred_clipped = Tensor(
        predictions.backend.clip(predictions.data, 1e-10, 1 - 1e-10),
        backend=predictions.backend,
        device=predictions.device
    )
    loss = negative(add(
        multiply(targets, log(pred_clipped)),
        multiply(subtract(1, targets), log(subtract(1, pred_clipped)))
    ))
    return mean(loss)


def batch_norm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    """Batch normalization"""
    mean_val = mean(x, axis=0, keepdims=True)
    var_val = var(x, axis=0, keepdims=False)
    var_tensor = Tensor(var_val.data, backend=x.backend, device=x.device)
    x_norm = divide(subtract(x, mean_val), sqrt(add(var_tensor, eps)))
    return add(multiply(gamma, x_norm), beta)


def layer_norm(x: Tensor, normalized_shape: Tuple[int, ...], eps: float = 1e-5) -> Tensor:
    """Layer normalization"""
    axes = tuple(range(-len(normalized_shape), 0))
    mean_val = mean(x, axis=axes, keepdims=True)
    var_val = var(x, axis=axes, keepdims=False)
    var_tensor = Tensor(var_val.data, backend=x.backend, device=x.device)
    x_norm = divide(subtract(x, mean_val), sqrt(add(var_tensor, eps)))
    return x_norm