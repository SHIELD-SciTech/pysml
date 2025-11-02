from .dtype import *
from .autograd import Function, is_grad_enabled, no_grad
import gc


class Tensor:
	def __init__(self, data, dtype=None, requires_grad=False):
		self._requires_grad = requires_grad
		self._grad = None
		self._grad_fn = None  # Computational graph node
		
		if dtype == None:
			dtype = DEFAULT_DTYPE
		self._dtype = dtype
		self._backend = DEFAULT_BACKEND
		self.device = None
		self.active_device = "cpu"
		self.data = self.convert(data, dtype)
	
	def requires_grad_(self, requires_grad=True):
		self._requires_grad = requires_grad
		return self
	
	@property
	def grad(self):
		return self._grad
	
	@grad.setter
	def grad(self, value):
		self._grad = value
	
	def zero_grad(self):
		self._grad = None
	
	def backward(self, gradient=None, retain_graph=False):
		if not self._requires_grad:
			raise RuntimeError("Called backward() on tensor that doesn't require grad")
		
		# If no gradient provided, assume scalar output with gradient 1
		if gradient is None:
			if self.data.size == 1:
				# Scalar output
				gradient = Tensor.__new__(Tensor)
				gradient._backend = self._backend
				gradient._dtype = self._dtype
				gradient.device = self.device
				gradient.active_device = self.active_device
				gradient.data = self._backend.asarray([1.0])
				gradient._requires_grad = False
				gradient._grad = None
				gradient._grad_fn = None
			else:
				raise RuntimeError(
					"grad must be specified for non-scalar tensor"
				)
		
		# Build topological order of computational graph
		topo_order = []
		visited = set()
		
		def build_topo(node):
			if node is None or id(node) in visited:
				return
			visited.add(id(node))
			
			if node._grad_fn is not None:
				for input_ref in node._grad_fn.inputs:
					if input_ref is not None:
						input_tensor = input_ref()
						if input_tensor is not None:
							build_topo(input_tensor)

			topo_order.append(node)
		
		build_topo(self)
		
		# Initialize gradient for output
		self._grad = gradient
		
		# Backward pass in reverse topological order
		for node in reversed(topo_order):
			if node._grad_fn is None:
				continue
			
			# Get gradient flowing into this node
			grad_output = node._grad
			if grad_output is None:
				continue
			
			# Compute gradients for inputs
			grad_inputs = node._grad_fn.apply_backward(grad_output)
			
			# Accumulate gradients for each input
			for grad_info in grad_inputs:
				if grad_info is None:
					continue
				
				input_tensor, grad = grad_info
				
				if input_tensor._grad is None:
					input_tensor._grad = grad
				else:
					# Accumulate gradient
					backend = input_tensor._backend
					input_tensor._grad.data = backend.add(
						input_tensor._grad.data, 
						grad.data
					)
		
		# Clean up computational graph if not retaining
		if not retain_graph:
			self._grad_fn = None

	def __add__(self, other):
		from . import engine
		return engine.add(self, other)
	
	def __radd__(self, other):
		from . import engine
		return engine.add(self, other)
	
	def __sub__(self, other):
		from . import engine
		return engine.subtract(self, other)
	
	def __rsub__(self, other):
		from . import engine
		# other - self = -(self - other)
		result = engine.subtract(self, other)
		return engine.negative(result)
	
	def __mul__(self, other):
		from . import engine
		return engine.multiply(self, other)
	
	def __rmul__(self, other):
		from . import engine
		return engine.multiply(self, other)
	
	def __truediv__(self, other):
		from . import engine
		return engine.divide(self, other)
	
	def __rtruediv__(self, other):
		from . import engine
		# Create tensor from other if it's a scalar
		if not isinstance(other, Tensor):
			other_tensor = Tensor([other], dtype=self._dtype)
			other_tensor.to(self.active_device)
			return engine.divide(other_tensor, self)
		return engine.divide(other, self)
	
	def __pow__(self, other):
		from . import engine
		return engine.power(self, other)
	
	def __neg__(self):
		from . import engine
		return engine.negative(self)
	
	def __matmul__(self, other):
		from . import engine
		return engine.matmul(self, other)

	@property
	def shape(self):
		return self.data.shape
	
	def to(self, device):
		if "xpu" in device:
			from .xpu import backend
			self._backend = backend
			if ":" in device:
				self.device = int(device.split(":")[1])
			self.data = backend.convert(self.data, self._dtype, self.device)
			self.active_device = device
			return self
		if "cuda" in device:
			from .cuda import backend
			self._backend = backend
			if ":" in device:
				self.device = int(device.split(":")[1])
			self.data = backend.convert(self.data, self._dtype, self.device)
			self.active_device = device
			return self
		from .cpu import backend
		self._backend = backend
		self.device = None
		self.data = backend.convert(self.data, self._dtype, self.device)
		self.active_device = device
		return self
	
	def convert(self, data, dtype, _device=None):
		self.data = self._backend.convert(data, dtype, device=_device)
		return self.data
	
	def detach(self):
		detached = Tensor.__new__(Tensor)
		detached._backend = self._backend
		detached._dtype = self._dtype
		detached._requires_grad = False
		detached._grad = None
		detached._grad_fn = None
		detached.device = self.device
		detached.active_device = self.active_device
		detached.data = self.data  # Share data
		return detached
	
	def clone(self):
		cloned_tensor = Tensor.__new__(Tensor)
		cloned_tensor._backend = self._backend
		cloned_tensor._dtype = self._dtype
		cloned_tensor._grad = self._grad
		cloned_tensor._requires_grad = self._requires_grad
		cloned_tensor._grad_fn = None  # Don't copy graph
		cloned_tensor.device = self.device
		cloned_tensor.active_device = self.active_device
		cloned_tensor.data = self._backend.copy(self.data)
		return cloned_tensor
	
	def T(self):
		"""Transpose (creates view with autograd support)"""
		from . import engine
		return engine.transpose(self)
	
	def sum(self, axis=None, keepdims=False):
		from . import engine
		return engine.sum_with_grad(self, axis=axis, keepdims=keepdims)
	
	def mean(self, axis=None, keepdims=False):
		from . import engine
		return engine.mean_with_grad(self, axis=axis, keepdims=keepdims)
	
	def item(self):
		backend = self._backend
		if hasattr(backend, 'asnumpy'):
			arr = backend.asnumpy(self.data)
		else:
			arr = self.data
		return float(arr.flat[0])
	
	def numpy(self):
		backend = self._backend
		if hasattr(backend, 'asnumpy'):
			return backend.asnumpy(self.data)
		return self.data
	
	def free(self):
		if self.data is not None:
			del self.data
			self.data = None
		if self._grad is not None:
			del self._grad
			self._grad = None
		if self._grad_fn is not None:
			self._grad_fn = None
		gc.collect()
	
	def __del__(self):
		if hasattr(self, 'data') and self.data is not None:
			del self.data
		if hasattr(self, '_grad') and self._grad is not None:
			del self._grad
		if hasattr(self, '_grad_fn'):
			self._grad_fn = None
	
	def __repr__(self):
		grad_fn_str = f", grad_fn=<{self._grad_fn.__class__.__name__}>" if hasattr(self, '_grad_fn') and self._grad_fn else ""
		return (f"Tensor({self.data}, dtype={self._dtype}, "
				f"requires_grad={self._requires_grad}{grad_fn_str})")
	
	def __str__(self):
		return self.__repr__()


# Default backend and dtype
from .cpu import backend as cpu_backend
DEFAULT_BACKEND = cpu_backend
DEFAULT_DTYPE = bf16()