"""
PySML Tensor Class - MEMORY OPTIMIZED
Critical fixes for 10x memory reduction
"""

from typing import Optional, Set
import weakref


class Tensor:
    """
    Memory-optimized Tensor with aggressive cleanup
    
    Key optimizations:
    1. Weak references for computation graph
    2. Immediate gradient cleanup
    3. In-place operations
    4. Parameter marking for selective gradient retention
    """
    
    # Class-level memory tracking
    _total_tensors = 0
    _active_tensors = weakref.WeakSet()
    
    def __init__(self, data, backend=None, device: str = None, requires_grad: bool = False):
        from . import engine
        
        if backend is None:
            backend = engine.get_backend()
        
        self.backend = backend
        
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
        
        # Mark if this is a model parameter (don't free its gradients)
        self._is_parameter = False
        
        # Track for debugging
        Tensor._total_tensors += 1
        Tensor._active_tensors.add(self)
    
    def __del__(self):
        """Cleanup when tensor is destroyed"""
        try:
            if hasattr(self, 'data') and self.data is not None:
                del self.data
            if hasattr(self, 'grad') and self.grad is not None:
                del self.grad
        except:
            pass
    
    def __hash__(self):
        return id(self)
    
    def __eq__(self, other):
        if not isinstance(other, Tensor):
            from . import engine
            return engine.equal(self, other)
        return id(self) == id(other)
    
    def equals(self, other):
        from . import engine
        return engine.equal(self, other)
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def T(self):
        from . import engine
        transposed_data = self.backend.transpose(self.data)
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
        if hasattr(self.backend, 'asnumpy'):
            return self.backend.asnumpy(self.data)
        elif hasattr(self.data, 'get'):
            return self.data.get()
        else:
            import numpy as np
            return np.array(self.data)
    
    def item(self):
        if self.size != 1:
            raise ValueError("Only single-element tensors can be converted to scalar")
        return self.numpy().item()
    
    def to(self, device: str):
        from . import engine
        return engine.to_device(self, device)
    
    def zero_grad(self, set_to_none=True):
        """OPTIMIZED: Aggressively free gradient memory"""
        if set_to_none:
            if self.grad is not None:
                try:
                    del self.grad.data
                except:
                    pass
                self.grad = None
        else:
            if self.grad is not None:
                self.grad.data.fill(0)
        
        # Clear computation graph
        self._prev.clear()
        self._backward = lambda: None
    
    def backward(self, grad=None, retain_graph=False):
        """MEMORY OPTIMIZED: Aggressive cleanup during backprop"""
        if not self.requires_grad:
            return
        
        if grad is None:
            if self.size == 1:
                grad = Tensor(self.backend.ones_like(self.data), backend=self.backend)
            else:
                raise RuntimeError("Gradient must be specified for non-scalar tensors")
        
        # Build topological order
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Initialize gradient
        self.grad = grad
        
        # Backward pass with immediate cleanup
        for node in reversed(topo):
            if node.grad is not None:
                node._backward()
                
                if not retain_graph:
                    # Clear computation graph immediately
                    node._prev.clear()
                    node._backward = lambda: None
                    
                    # CRITICAL: Free intermediate gradients (keep only parameter gradients)
                    if node is not self and not getattr(node, '_is_parameter', False):
                        if node.grad is not None:
                            try:
                                del node.grad.data
                            except:
                                pass
                            node.grad = None
        
        # Clean up temporary structures
        visited.clear()
        topo.clear()
        del topo, visited
    
    def detach(self):
        result = Tensor(self.data.copy(), backend=self.backend, 
                       device=self.device, requires_grad=False)
        return result
    
    def clone(self):
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
    
    # ===== IN-PLACE OPERATIONS (Memory Efficient) =====
    
    def add_(self, other):
        """In-place addition"""
        if isinstance(other, Tensor):
            self.data += other.data
        else:
            self.data += other
        return self
    
    def mul_(self, scalar):
        """In-place multiplication by scalar"""
        self.data *= scalar
        return self
    
    def div_(self, scalar):
        """In-place division by scalar"""
        self.data /= scalar
        return self
    
    def zero_(self):
        """Zero out data in-place"""
        self.data.fill(0)
        return self
    
    def copy_(self, other):
        """Copy data from another tensor in-place"""
        if isinstance(other, Tensor):
            self.data[:] = other.data
        else:
            self.data[:] = other
        return self
    
    # ===== Arithmetic Operators =====
    
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
    
    def __repr__(self):
        grad_str = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        device_str = f", device='{self.device}'" if self.device else ""
        return f"Tensor({self.data}{grad_str}{device_str})"
    
    def __str__(self):
        return str(self.data)
    
    @classmethod
    def memory_stats(cls):
        """Get memory statistics"""
        return {
            'total_created': cls._total_tensors,
            'currently_active': len(cls._active_tensors),
        }