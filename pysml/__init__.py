"""
PySML - Python Simple Machine Learning Framework
Multi-backend tensor computation framework with automatic differentiation

Supports:
- CPU (NumPy)
- Intel XPU (dpnp)
- NVIDIA CUDA (CuPy)
"""

__version__ = "0.1.0"

# Import core components
from .tensor import Tensor
from . import engine

# Re-export commonly used functions from engine
from .engine import (
    # Device management
    set_device,
    get_device,
    get_backend,
    get_available_devices,
    to_device,
    
    # Creation operations
    zeros,
    ones,
    full,
    eye,
    arange,
    linspace,
    randn,
    rand,
    randint,
    
    # Arithmetic operations
    add,
    subtract,
    multiply,
    divide,
    power,
    negative,
    positive,
    matmul,
    
    # Reduction operations
    sum,
    mean,
    max,
    min,
    var,
    std,
    
    # Activation functions
    relu,
    sigmoid,
    tanh,
    softmax,
    gelu,
    
    # Mathematical functions
    exp,
    log,
    sqrt,
    abs,
    sin,
    cos,
    
    # Shape operations
    reshape,
    transpose,
    squeeze,
    unsqueeze,
    concatenate,
    stack,
    flatten,
    
    # Comparison operations
    equal,
    not_equal,
    less,
    less_equal,
    greater,
    greater_equal,
    
    # Advanced operations
    clip,
    where,
    maximum,
    minimum,
    
    # Linear algebra
    dot,
    outer,
    inner,
    einsum,
    
    # Loss functions
    mse_loss,
    cross_entropy_loss,
    binary_cross_entropy_loss,
    
    # Normalization
    batch_norm,
    layer_norm,
    dropout,
)

__all__ = [
    # Core
    'Tensor',
    'engine',
    
    # Device management
    'set_device',
    'get_device',
    'get_backend',
    'get_available_devices',
    'to_device',
    
    # Creation
    'zeros',
    'ones',
    'full',
    'eye',
    'arange',
    'linspace',
    'randn',
    'rand',
    'randint',
    
    # Arithmetic
    'add',
    'subtract',
    'multiply',
    'divide',
    'power',
    'negative',
    'positive',
    'matmul',
    
    # Reductions
    'sum',
    'mean',
    'max',
    'min',
    'var',
    'std',
    
    # Activations
    'relu',
    'sigmoid',
    'tanh',
    'softmax',
    'gelu',
    
    # Math
    'exp',
    'log',
    'sqrt',
    'abs',
    'sin',
    'cos',
    
    # Shape
    'reshape',
    'transpose',
    'squeeze',
    'unsqueeze',
    'concatenate',
    'stack',
    'flatten',
    
    # Comparison
    'equal',
    'not_equal',
    'less',
    'less_equal',
    'greater',
    'greater_equal',
    
    # Advanced
    'clip',
    'where',
    'maximum',
    'minimum',
    
    # Linear algebra
    'dot',
    'outer',
    'inner',
    'einsum',
    
    # Loss
    'mse_loss',
    'cross_entropy_loss',
    'binary_cross_entropy_loss',
    
    # Normalization
    'batch_norm',
    'layer_norm',
    'dropout',
]


# Example usage and quick start
def demo():
    """Run a quick demo of PySML capabilities"""
    print("=== PySML Quick Demo ===\n")
    
    print(f"PySML version: {__version__}")
    print(f"Available devices: {get_available_devices()}")
    print(f"Current device: {get_device()}\n")
    
    # Create tensors
    print("Creating tensors...")
    x = randn(3, 3, requires_grad=True)
    y = randn(3, 3, requires_grad=True)
    
    print(f"x:\n{x}")
    print(f"y:\n{y}\n")
    
    # Forward pass
    print("Forward pass...")
    z = matmul(x, y)
    loss = sum(relu(z))
    
    print(f"z = x @ y:\n{z}")
    print(f"loss = sum(relu(z)): {loss.item()}\n")
    
    # Backward pass
    print("Backward pass...")
    loss.backward()
    
    print(f"Gradient of x:\n{x.grad}")
    print(f"Gradient of y:\n{y.grad}\n")
    
    print("Demo complete!")


if __name__ == "__main__":
    demo()