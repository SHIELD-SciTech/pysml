"""
PySML Tensor Class with Full Autograd Support - MEMORY OPTIMIZED
Enhanced tensor implementation with complete gradient computation and aggressive memory management
"""

from typing import Optional, Union, Tuple, List, Set


class Tensor:
    """
    Universal Tensor class with full automatic differentiation support
    
    Attributes:
        data: Underlying array data from backend
        backend: Backend module being used
        device: Device string ('cpu', 'xpu', 'cuda')
        requires_grad: Whether to track gradients
        grad: Gradient tensor
        grad_fn: Function for backward pass
        _prev: Set of parent tensors in computation graph
        _op: Operation that created this tensor
    """
    
    def __init__(self, data, backend=None, device: str = None, requires_grad: bool = False):
        """
        Initialize Tensor
        
        Args:
            data: Array-like data or backend array
            backend: Backend module to use (auto-detected if None)
            device: Device to place tensor on
            requires_grad: Whether to compute gradients
        """
        from . import engine
        
        # Auto-detect backend if not provided
        if backend is None:
            backend = engine.get_backend()
        
        self.backend = backend
        
        # Convert data to backend array
        if hasattr(data, '__array__'):
            self.data = data
        else:
            self.data = self.backend.array(data)
        
        self.device = device or engine.get_default_device()
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
        
        # For autograd graph - use set for efficient lookups
        self._prev: Set['Tensor'] = set()
        self._op: str = ''
        self._backward = lambda: None
    
    def __hash__(self):
        """Make tensor hashable using id"""
        return id(self)
    
    def __eq__(self, other):
        """Equality for hashing (identity-based) or element-wise comparison"""
        # For hashing and identity comparison
        if not isinstance(other, Tensor):
            # Element-wise comparison with scalar
            from . import engine
            return engine.equal(self, other)
        # For sets and dicts, use identity
        return id(self) == id(other)
    
    def equals(self, other):
        """Element-wise equality comparison (returns boolean Tensor)"""
        from . import engine
        return engine.equal(self, other)
    
    @property
    def shape(self):
        """Return shape of tensor"""
        return self.data.shape
    
    @property
    def dtype(self):
        """Return data type of tensor"""
        return self.data.dtype
    
    @property
    def ndim(self):
        """Return number of dimensions"""
        return self.data.ndim
    
    @property
    def size(self):
        """Return total number of elements"""
        return self.data.size
    
    @property
    def T(self):
        """Transpose - backend-aware version"""
        from . import engine
        
        # Use backend's transpose to maintain backend type
        transposed_data = self.backend.transpose(self.data)
        
        # Create new tensor with transposed data
        result = Tensor(transposed_data, backend=self.backend, device=self.device,
                       requires_grad=self.requires_grad)
        
        if self.requires_grad:
            result._prev = {self}
            result._op = 'transpose'
            
            def _backward():
                if self.grad is None:
                    self.grad = Tensor(self.backend.transpose(result.grad.data), 
                                      backend=self.backend)
                else:
                    self.grad.data = self.grad.data + self.backend.transpose(result.grad.data)
            
            result._backward = _backward
        
        return result
    
    def numpy(self):
        """Convert to NumPy array"""
        if hasattr(self.backend, 'asnumpy'):
            return self.backend.asnumpy(self.data)
        elif hasattr(self.data, 'get'):
            return self.data.get()
        else:
            import numpy as np
            return np.array(self.data)
    
    def item(self):
        """Return scalar value (only for single-element tensors)"""
        if self.size != 1:
            raise ValueError("Only single-element tensors can be converted to scalar")
        return self.numpy().item()
    
    def to(self, device: str):
        """Move tensor to different device"""
        from . import engine
        return engine.to_device(self, device)
    
    def zero_grad(self, set_to_none=True):
        """
        Zero out gradients and clear computation graph
        
        Args:
            set_to_none: If True, set grad to None (better for memory). 
                        If False, zero out existing gradient array.
        """
        if set_to_none:
            self.grad = None
        else:
            if self.grad is not None:
                # Zero out in-place to avoid allocation
                self.grad.data.fill(0)
        
        # CRITICAL: Clear computation graph to free memory
        self._prev.clear()
        self._backward = lambda: None
    
    def backward(self, grad=None, retain_graph=False):
        """
        Compute gradients via backpropagation using topological sort
        
        Args:
            grad: Gradient from upstream (defaults to ones for scalar)
            retain_graph: If False (default), free computation graph after backward pass
        """
        if not self.requires_grad:
            return
        
        # Initialize gradient
        if grad is None:
            if self.size == 1:
                grad = Tensor(self.backend.ones_like(self.data), backend=self.backend)
            else:
                raise RuntimeError("Gradient must be specified for non-scalar tensors")
        
        # Build topological order of computation graph
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Initialize gradient for this tensor
        self.grad = grad
        
        # Propagate gradients in reverse topological order
        for node in reversed(topo):
            if node.grad is not None:
                node._backward()
                
                # CRITICAL FIX: Clear computation graph after backward to free memory
                if not retain_graph:
                    node._prev.clear()
                    node._backward = lambda: None
                    # Keep grad but clear the graph
        
        # Clear visited set to free memory
        visited.clear()
        del topo
    
    def detach(self):
        """Return a new tensor detached from the computation graph"""
        result = Tensor(self.data.copy(), backend=self.backend, 
                       device=self.device, requires_grad=False)
        return result
    
    def clone(self):
        """Return a copy of the tensor that retains gradient tracking"""
        result = Tensor(self.data.copy(), backend=self.backend,
                       device=self.device, requires_grad=self.requires_grad)
        if self.requires_grad:
            result._prev = {self}
            result._op = 'clone'
            
            def _backward():
                if self.grad is None:
                    self.grad = result.grad
                else:
                    self.grad.data = self.grad.data + result.grad.data
            
            result._backward = _backward
        
        return result
    
    def __repr__(self):
        grad_str = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        device_str = f", device='{self.device}'" if self.device else ""
        return f"Tensor({self.data}{grad_str}{device_str})"
    
    def __str__(self):
        return str(self.data)
    
    # ===== Arithmetic Operators with Full Autograd =====
    
    def __add__(self, other):
        from . import engine
        return engine.add(self, other)
    
    def __radd__(self, other):
        from . import engine
        return engine.add(other, self)
    
    def __sub__(self, other):
        from . import engine
        return engine.subtract(self, other)
    
    def __rsub__(self, other):
        from . import engine
        return engine.subtract(other, self)
    
    def __mul__(self, other):
        from . import engine
        return engine.multiply(self, other)
    
    def __rmul__(self, other):
        from . import engine
        return engine.multiply(other, self)
    
    def __truediv__(self, other):
        from . import engine
        return engine.divide(self, other)
    
    def __rtruediv__(self, other):
        from . import engine
        return engine.divide(other, self)
    
    def __pow__(self, other):
        from . import engine
        return engine.power(self, other)
    
    def __neg__(self):
        from . import engine
        return engine.negative(self)
    
    def __pos__(self):
        from . import engine
        return engine.positive(self)
    
    def __matmul__(self, other):
        from . import engine
        return engine.matmul(self, other)
    
    # ===== Comparison Operators =====
    # Note: __eq__ is handled specially for hashing, use .equals() for element-wise comparison
    
    def __ne__(self, other):
        from . import engine
        return engine.not_equal(self, other)
    
    def __lt__(self, other):
        from . import engine
        return engine.less(self, other)
    
    def __le__(self, other):
        from . import engine
        return engine.less_equal(self, other)
    
    def __gt__(self, other):
        from . import engine
        return engine.greater(self, other)
    
    def __ge__(self, other):
        from . import engine
        return engine.greater_equal(self, other)
    
    # ===== Indexing =====
    
    def __getitem__(self, key):
        result = Tensor(self.data[key], backend=self.backend, device=self.device, 
                       requires_grad=self.requires_grad)
        
        if self.requires_grad:
            result._prev = {self}
            result._op = 'getitem'
            
            def _backward():
                if self.grad is None:
                    self.grad = Tensor(self.backend.zeros_like(self.data), backend=self.backend)
                
                # Add gradient at the indexed location
                grad_data = self.grad.data
                grad_data[key] = grad_data[key] + result.grad.data
            
            result._backward = _backward
        
        return result
    
    def __setitem__(self, key, value):
        if isinstance(value, Tensor):
            self.data[key] = value.data
        else:
            self.data[key] = value
    
    # ===== Tensor Methods =====
    
    def reshape(self, *shape):
        from . import engine
        return engine.reshape(self, shape if len(shape) > 1 else shape[0])
    
    def transpose(self, axes=None):
        from . import engine
        return engine.transpose(self, axes)
    
    def permute(self, *axes):
        """
        Permute the dimensions of the tensor.
        More intuitive than transpose for reordering dimensions.
        
        Args:
            *axes: New order of dimensions (can be passed as tuple or unpacked)
        
        Example:
            >>> x = Tensor(data)  # shape: (2, 3, 4, 5)
            >>> y = x.permute(0, 2, 1, 3)  # shape: (2, 4, 3, 5)
            >>> # or
            >>> y = x.permute((0, 2, 1, 3))
        """
        from . import engine
        # Handle both x.permute(0, 2, 1, 3) and x.permute((0, 2, 1, 3))
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = axes[0]
        return engine.transpose(self, axes)
    
    def flatten(self, start_dim=0, end_dim=-1):
        from . import engine
        return engine.flatten(self, start_dim, end_dim)
    
    def squeeze(self, axis=None):
        from . import engine
        return engine.squeeze(self, axis)
    
    def unsqueeze(self, axis):
        from . import engine
        return engine.unsqueeze(self, axis)
    
    def sum(self, axis=None, keepdims=False):
        from . import engine
        return engine.sum(self, axis, keepdims)
    
    def mean(self, axis=None, keepdims=False):
        from . import engine
        return engine.mean(self, axis, keepdims)
    
    def max(self, axis=None, keepdims=False):
        from . import engine
        return engine.max(self, axis, keepdims)
    
    def min(self, axis=None, keepdims=False):
        from . import engine
        return engine.min(self, axis, keepdims)
    
    def var(self, axis=None, keepdims=False):
        from . import engine
        return engine.var(self, axis, keepdims)
    
    def std(self, axis=None, keepdims=False):
        from . import engine
        return engine.std(self, axis, keepdims)
    
    def exp(self):
        from . import engine
        return engine.exp(self)
    
    def log(self):
        from . import engine
        return engine.log(self)
    
    def sqrt(self):
        from . import engine
        return engine.sqrt(self)
    
    def abs(self):
        from . import engine
        return engine.abs(self)
    
    def sin(self):
        from . import engine
        return engine.sin(self)
    
    def cos(self):
        from . import engine
        return engine.cos(self)
    
    def relu(self):
        from . import engine
        return engine.relu(self)
    
    def sigmoid(self):
        from . import engine
        return engine.sigmoid(self)
    
    def tanh(self):
        from . import engine
        return engine.tanh(self)
    
    def softmax(self, axis=-1):
        from . import engine
        return engine.softmax(self, axis)
    
    def clip(self, min_val=None, max_val=None):
        from . import engine
        return engine.clip(self, min_val, max_val)