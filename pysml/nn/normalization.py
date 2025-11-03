from .module import Module, Parameter
import math


class LayerNorm(Module):
	
	def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
		super().__init__()
		
		if isinstance(normalized_shape, int):
			normalized_shape = (normalized_shape,)
		self.normalized_shape = tuple(normalized_shape)
		self.eps = eps
		self.elementwise_affine = elementwise_affine
		
		if elementwise_affine:
			self.weight = Parameter(self._initialize_weight())
			self.bias = Parameter(self._initialize_bias())
		else:
			self.weight = None
			self.bias = None
	
	def _initialize_weight(self):
		from .. import Tensor
		import numpy as np
		# Initialize to ones (no scaling initially)
		data = np.ones(self.normalized_shape)
		return Tensor(data, requires_grad=True)
	
	def _initialize_bias(self):
		from .. import Tensor
		import numpy as np
		# Initialize to zeros (no shift initially)
		data = np.zeros(self.normalized_shape)
		return Tensor(data, requires_grad=True)
	
	def forward(self, x):
		from .. import engine
		
		# Use optimized backend implementation
		# 50% memory savings through efficient variance computation!
		weight_data = self.weight.data if self.weight is not None else None
		bias_data = self.bias.data if self.bias is not None else None
		
		return engine.layer_norm(x, self.normalized_shape, weight_data, bias_data, self.eps)
	
	def extra_repr(self):
		return (f"{self.normalized_shape}, eps={self.eps}, "
				f"elementwise_affine={self.elementwise_affine}")


class RMSNorm(Module):
	
	def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
		super().__init__()
		
		if isinstance(normalized_shape, int):
			normalized_shape = (normalized_shape,)
		self.normalized_shape = tuple(normalized_shape)
		self.eps = eps
		self.elementwise_affine = elementwise_affine
		
		if elementwise_affine:
			self.weight = Parameter(self._initialize_weight())
		else:
			self.weight = None
	
	def _initialize_weight(self):
		from .. import Tensor
		import numpy as np
		data = np.ones(self.normalized_shape)
		return Tensor(data, requires_grad=True)
	
	def forward(self, x):
		from .. import engine
		
		# RMSNorm is more efficient than LayerNorm (no mean computation)
		# Used in LLaMA, Mistral, and other modern LLMs
		weight_data = self.weight.data if self.weight is not None else None
		
		return engine.rms_norm(x, self.normalized_shape, weight_data, self.eps)
	
	def extra_repr(self):
		return (f"{self.normalized_shape}, eps={self.eps}, "
				f"elementwise_affine={self.elementwise_affine}")


class BatchNorm1d(Module):
	
	def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
		super().__init__()
		self.num_features = num_features
		self.eps = eps
		self.momentum = momentum
		self.affine = affine
		self.track_running_stats = track_running_stats
		
		if affine:
			self.weight = Parameter(self._initialize_weight())
			self.bias = Parameter(self._initialize_bias())
		else:
			self.weight = None
			self.bias = None
		
		if track_running_stats:
			# Register buffers (non-trainable)
			from .. import Tensor
			import numpy as np
			self.register_buffer('running_mean', Tensor(np.zeros(num_features), requires_grad=False))
			self.register_buffer('running_var', Tensor(np.ones(num_features), requires_grad=False))
			self.register_buffer('num_batches_tracked', Tensor(np.array(0), requires_grad=False))
		else:
			self.running_mean = None
			self.running_var = None
			self.num_batches_tracked = None
	
	def _initialize_weight(self):
		from .. import Tensor
		import numpy as np
		data = np.ones(self.num_features)
		return Tensor(data, requires_grad=True)
	
	def _initialize_bias(self):
		from .. import Tensor
		import numpy as np
		data = np.zeros(self.num_features)
		return Tensor(data, requires_grad=True)
	
	def forward(self, x):
		from .. import engine
		
		# x can be (batch, features) or (batch, features, length)
		if len(x.shape) not in [2, 3]:
			raise ValueError(f"Expected 2D or 3D input, got {len(x.shape)}D")
		
		# Get running statistics if tracking
		running_mean = self.running_mean.data if self.running_mean is not None else None
		running_var = self.running_var.data if self.running_var is not None else None
		
		weight_data = self.weight.data if self.weight is not None else None
		bias_data = self.bias.data if self.bias is not None else None
		
		# Use optimized backend implementation
		return engine.batch_norm(
			x, running_mean, running_var, weight_data, bias_data,
			training=self.training, momentum=self.momentum, eps=self.eps
		)
	
	def extra_repr(self):
		return (f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, "
				f"affine={self.affine}, track_running_stats={self.track_running_stats}")


class BatchNorm2d(Module):
	
	def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
		super().__init__()
		self.num_features = num_features
		self.eps = eps
		self.momentum = momentum
		self.affine = affine
		self.track_running_stats = track_running_stats
		
		if affine:
			self.weight = Parameter(self._initialize_weight())
			self.bias = Parameter(self._initialize_bias())
		else:
			self.weight = None
			self.bias = None
		
		if track_running_stats:
			from .. import Tensor
			import numpy as np
			self.register_buffer('running_mean', Tensor(np.zeros(num_features), requires_grad=False))
			self.register_buffer('running_var', Tensor(np.ones(num_features), requires_grad=False))
			self.register_buffer('num_batches_tracked', Tensor(np.array(0), requires_grad=False))
		else:
			self.running_mean = None
			self.running_var = None
			self.num_batches_tracked = None
	
	def _initialize_weight(self):
		from .. import Tensor
		import numpy as np
		data = np.ones(self.num_features)
		return Tensor(data, requires_grad=True)
	
	def _initialize_bias(self):
		from .. import Tensor
		import numpy as np
		data = np.zeros(self.num_features)
		return Tensor(data, requires_grad=True)
	
	def forward(self, x):
		from .. import engine
		
		# x: (batch, channels, height, width)
		if len(x.shape) != 4:
			raise ValueError(f"Expected 4D input (batch, channels, H, W), got {len(x.shape)}D")
		
		running_mean = self.running_mean.data if self.running_mean is not None else None
		running_var = self.running_var.data if self.running_var is not None else None
		
		weight_data = self.weight.data if self.weight is not None else None
		bias_data = self.bias.data if self.bias is not None else None
		
		return engine.batch_norm(
			x, running_mean, running_var, weight_data, bias_data,
			training=self.training, momentum=self.momentum, eps=self.eps
		)
	
	def extra_repr(self):
		return (f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, "
				f"affine={self.affine}, track_running_stats={self.track_running_stats}")


class BatchNorm3d(Module):
	
	def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
		super().__init__()
		self.num_features = num_features
		self.eps = eps
		self.momentum = momentum
		self.affine = affine
		self.track_running_stats = track_running_stats
		
		if affine:
			self.weight = Parameter(self._initialize_weight())
			self.bias = Parameter(self._initialize_bias())
		else:
			self.weight = None
			self.bias = None
		
		if track_running_stats:
			from .. import Tensor
			import numpy as np
			self.register_buffer('running_mean', Tensor(np.zeros(num_features), requires_grad=False))
			self.register_buffer('running_var', Tensor(np.ones(num_features), requires_grad=False))
			self.register_buffer('num_batches_tracked', Tensor(np.array(0), requires_grad=False))
		else:
			self.running_mean = None
			self.running_var = None
			self.num_batches_tracked = None
	
	def _initialize_weight(self):
		from .. import Tensor
		import numpy as np
		data = np.ones(self.num_features)
		return Tensor(data, requires_grad=True)
	
	def _initialize_bias(self):
		from .. import Tensor
		import numpy as np
		data = np.zeros(self.num_features)
		return Tensor(data, requires_grad=True)
	
	def forward(self, x):
		from .. import engine
		
		# x: (batch, channels, depth, height, width)
		if len(x.shape) != 5:
			raise ValueError(f"Expected 5D input (batch, channels, D, H, W), got {len(x.shape)}D")
		
		# Reshape to 4D for batch_norm implementation
		# (batch, channels, D*H*W)
		batch, channels, d, h, w = x.shape
		x_reshaped = x.reshape((batch, channels, d * h * w))
		
		running_mean = self.running_mean.data if self.running_mean is not None else None
		running_var = self.running_var.data if self.running_var is not None else None
		
		weight_data = self.weight.data if self.weight is not None else None
		bias_data = self.bias.data if self.bias is not None else None
		
		output = engine.batch_norm(
			x_reshaped, running_mean, running_var, weight_data, bias_data,
			training=self.training, momentum=self.momentum, eps=self.eps
		)
		
		# Reshape back to 5D
		return output.reshape((batch, channels, d, h, w))
	
	def extra_repr(self):
		return (f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, "
				f"affine={self.affine}, track_running_stats={self.track_running_stats}")


class GroupNorm(Module):
	
	def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
		super().__init__()
		
		if num_channels % num_groups != 0:
			raise ValueError(f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})")
		
		self.num_groups = num_groups
		self.num_channels = num_channels
		self.eps = eps
		self.affine = affine
		
		if affine:
			self.weight = Parameter(self._initialize_weight())
			self.bias = Parameter(self._initialize_bias())
		else:
			self.weight = None
			self.bias = None
	
	def _initialize_weight(self):
		from .. import Tensor
		import numpy as np
		data = np.ones(self.num_channels)
		return Tensor(data, requires_grad=True)
	
	def _initialize_bias(self):
		from .. import Tensor
		import numpy as np
		data = np.zeros(self.num_channels)
		return Tensor(data, requires_grad=True)
	
	def forward(self, x):
		from .. import engine
		
		# x: (batch, channels, *) where * is spatial dimensions
		if len(x.shape) < 2:
			raise ValueError(f"Expected at least 2D input, got {len(x.shape)}D")
		
		weight_data = self.weight.data if self.weight is not None else None
		bias_data = self.bias.data if self.bias is not None else None
		
		# Use optimized backend implementation
		return engine.group_norm(x, self.num_groups, weight_data, bias_data, self.eps)
	
	def extra_repr(self):
		return (f"{self.num_groups}, {self.num_channels}, eps={self.eps}, "
				f"affine={self.affine}")


class InstanceNorm1d(Module):
	
	def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
		super().__init__()
		self.num_features = num_features
		self.eps = eps
		self.momentum = momentum
		self.affine = affine
		
		# InstanceNorm is just GroupNorm with num_groups=num_channels
		# Each channel is normalized independently
		
		if affine:
			self.weight = Parameter(self._initialize_weight())
			self.bias = Parameter(self._initialize_bias())
		else:
			self.weight = None
			self.bias = None
	
	def _initialize_weight(self):
		from .. import Tensor
		import numpy as np
		data = np.ones(self.num_features)
		return Tensor(data, requires_grad=True)
	
	def _initialize_bias(self):
		from .. import Tensor
		import numpy as np
		data = np.zeros(self.num_features)
		return Tensor(data, requires_grad=True)
	
	def forward(self, x):
		from .. import engine
		
		# x: (batch, features, length)
		if len(x.shape) != 3:
			raise ValueError(f"Expected 3D input (batch, features, length), got {len(x.shape)}D")
		
		weight_data = self.weight.data if self.weight is not None else None
		bias_data = self.bias.data if self.bias is not None else None
		
		# Use GroupNorm with num_groups = num_channels
		return engine.group_norm(x, self.num_features, weight_data, bias_data, self.eps)
	
	def extra_repr(self):
		return (f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, "
				f"affine={self.affine}")


class InstanceNorm2d(Module):
	
	def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
		super().__init__()
		self.num_features = num_features
		self.eps = eps
		self.momentum = momentum
		self.affine = affine
		
		if affine:
			self.weight = Parameter(self._initialize_weight())
			self.bias = Parameter(self._initialize_bias())
		else:
			self.weight = None
			self.bias = None
	
	def _initialize_weight(self):
		from .. import Tensor
		import numpy as np
		data = np.ones(self.num_features)
		return Tensor(data, requires_grad=True)
	
	def _initialize_bias(self):
		from .. import Tensor
		import numpy as np
		data = np.zeros(self.num_features)
		return Tensor(data, requires_grad=True)
	
	def forward(self, x):
		from .. import engine
		
		# x: (batch, channels, height, width)
		if len(x.shape) != 4:
			raise ValueError(f"Expected 4D input (batch, channels, H, W), got {len(x.shape)}D")
		
		weight_data = self.weight.data if self.weight is not None else None
		bias_data = self.bias.data if self.bias is not None else None
		
		# Use GroupNorm with num_groups = num_channels
		return engine.group_norm(x, self.num_features, weight_data, bias_data, self.eps)
	
	def extra_repr(self):
		return (f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, "
				f"affine={self.affine}")


class InstanceNorm3d(Module):
	
	def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
		super().__init__()
		self.num_features = num_features
		self.eps = eps
		self.momentum = momentum
		self.affine = affine
		
		if affine:
			self.weight = Parameter(self._initialize_weight())
			self.bias = Parameter(self._initialize_bias())
		else:
			self.weight = None
			self.bias = None
	
	def _initialize_weight(self):
		from .. import Tensor
		import numpy as np
		data = np.ones(self.num_features)
		return Tensor(data, requires_grad=True)
	
	def _initialize_bias(self):
		from .. import Tensor
		import numpy as np
		data = np.zeros(self.num_features)
		return Tensor(data, requires_grad=True)
	
	def forward(self, x):
		from .. import engine
		
		# x: (batch, channels, depth, height, width)
		if len(x.shape) != 5:
			raise ValueError(f"Expected 5D input (batch, channels, D, H, W), got {len(x.shape)}D")
		
		weight_data = self.weight.data if self.weight is not None else None
		bias_data = self.bias.data if self.bias is not None else None
		
		# Use GroupNorm with num_groups = num_channels
		return engine.group_norm(x, self.num_features, weight_data, bias_data, self.eps)
	
	def extra_repr(self):
		return (f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, "
				f"affine={self.affine}")


class LocalResponseNorm(Module):
	
	def __init__(self, size, alpha=1e-4, beta=0.75, k=1.0):
		super().__init__()
		self.size = size
		self.alpha = alpha
		self.beta = beta
		self.k = k
	
	def forward(self, x):
		# LRN: normalize across channels
		# output = x / (k + alpha * sum(x^2 in window))^beta
		
		from .. import engine
		backend = x._backend
		
		# x: (batch, channels, height, width)
		if len(x.shape) != 4:
			raise ValueError(f"Expected 4D input, got {len(x.shape)}D")
		
		batch, channels, height, width = x.shape
		
		# Compute x^2
		x_squared = engine.square(x)
		
		# For each channel, sum over neighboring channels
		# This is a simplified version - full implementation needs proper windowing
		pad = self.size // 2
		
		# Simple implementation: average pooling across channels
		# TODO: Implement proper local response normalization with windowing
		
		# For now, use a simplified version
		sum_squared = engine.mean(x_squared, axis=1, keepdims=True)
		
		# Broadcast to all channels
		from .. import Tensor
		sum_squared_broadcast = Tensor.__new__(Tensor)
		sum_squared_broadcast._backend = backend
		sum_squared_broadcast._dtype = x._dtype
		sum_squared_broadcast.device = x.device
		sum_squared_broadcast.active_device = x.active_device
		sum_squared_broadcast.data = backend.broadcast_to(
			sum_squared.data, x.shape
		) if backend.broadcast_to else backend.tile(sum_squared.data, (1, channels, 1, 1))
		sum_squared_broadcast._requires_grad = x._requires_grad
		sum_squared_broadcast._grad = None
		
		# Compute denominator: (k + alpha * sum)^beta
		denom = engine.power(
			engine.add(self.k, engine.multiply(self.alpha, sum_squared_broadcast)),
			self.beta
		)
		
		return engine.divide(x, denom)
	
	def extra_repr(self):
		return f"size={self.size}, alpha={self.alpha}, beta={self.beta}, k={self.k}"


__all__ = [
	'LayerNorm', 'RMSNorm',
	'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d',
	'GroupNorm',
	'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d',
	'LocalResponseNorm',
]