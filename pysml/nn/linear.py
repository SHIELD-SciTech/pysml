import math
from .module import Module, Parameter


class Linear(Module):
	
	def __init__(self, in_features, out_features, bias=True):
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		
		# Initialize weight parameter
		self.weight = Parameter(self._initialize_weight())
		
		# Initialize bias parameter if needed
		if bias:
			self.bias = Parameter(self._initialize_bias())
		else:
			self.bias = None
	
	def _initialize_weight(self):
		# Kaiming/He initialization for better gradient flow
		# Variance = 2 / fan_in for ReLU-based networks
		from .. import Tensor
		import numpy as np
		
		limit = math.sqrt(1.0 / self.in_features)
		data = np.random.uniform(-limit, limit, (self.out_features, self.in_features))
		return Tensor(data, requires_grad=True)
	
	def _initialize_bias(self):
		from .. import Tensor
		import numpy as np
		
		limit = math.sqrt(1.0 / self.in_features)
		data = np.random.uniform(-limit, limit, (self.out_features,))
		return Tensor(data, requires_grad=True)
	
	def forward(self, x):
		# x: (batch, in_features) or (*, in_features)
		# weight: (out_features, in_features)
		# output: (batch, out_features) or (*, out_features)
		
		from .. import engine
		
		# Ensure input has correct shape
		input_shape = x.shape
		if len(input_shape) == 1:
			# Single sample: (in_features,)
			x = x.reshape((1, -1))
			single_sample = True
		else:
			single_sample = False
		
		# Check dimension compatibility
		if x.shape[-1] != self.in_features:
			raise ValueError(
				f"Input dimension mismatch: expected {self.in_features}, "
				f"got {x.shape[-1]}"
			)
		
		# Reshape for batched operation if needed
		original_shape = x.shape
		if len(original_shape) > 2:
			# Flatten all but last dimension: (batch, seq, features) -> (batch*seq, features)
			batch_size = 1
			for dim in original_shape[:-1]:
				batch_size *= dim
			x = x.reshape((batch_size, self.in_features))
		
		# Matrix multiplication: (batch, in) @ (in, out).T = (batch, out)
		output = engine.matmul(x, self.weight.data.T())
		
		# Add bias if present
		if self.bias is not None:
			output = engine.add(output, self.bias.data)
		
		# Reshape back to original structure
		if len(original_shape) > 2:
			output_shape = list(original_shape[:-1]) + [self.out_features]
			output = output.reshape(tuple(output_shape))
		
		# Remove batch dimension if input was single sample
		if single_sample:
			output = output.reshape((self.out_features,))
		
		return output
	
	def extra_repr(self):
		return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class Bilinear(Module):
	
	def __init__(self, in1_features, in2_features, out_features, bias=True):
		super().__init__()
		self.in1_features = in1_features
		self.in2_features = in2_features
		self.out_features = out_features
		
		# Weight tensor: (out_features, in1_features, in2_features)
		self.weight = Parameter(self._initialize_weight())
		
		if bias:
			self.bias = Parameter(self._initialize_bias())
		else:
			self.bias = None
	
	def _initialize_weight(self):
		from .. import Tensor
		import numpy as np
		
		# Xavier/Glorot initialization
		limit = math.sqrt(6.0 / (self.in1_features + self.in2_features + self.out_features))
		data = np.random.uniform(-limit, limit, 
								  (self.out_features, self.in1_features, self.in2_features))
		return Tensor(data, requires_grad=True)
	
	def _initialize_bias(self):
		from .. import Tensor
		import numpy as np
		
		data = np.zeros((self.out_features,))
		return Tensor(data, requires_grad=True)
	
	def forward(self, x1, x2):
		# Bilinear transformation: y = x1^T W x2 + b
		# x1: (batch, in1_features)
		# x2: (batch, in2_features)
		# weight: (out_features, in1_features, in2_features)
		# output: (batch, out_features)
		
		from .. import engine
		
		batch_size = x1.shape[0]
		
		# Check dimensions
		if x1.shape[-1] != self.in1_features:
			raise ValueError(
				f"x1 dimension mismatch: expected {self.in1_features}, got {x1.shape[-1]}"
			)
		if x2.shape[-1] != self.in2_features:
			raise ValueError(
				f"x2 dimension mismatch: expected {self.in2_features}, got {x2.shape[-1]}"
			)
		
		# Efficient computation: (batch, out) = sum_i,j (x1[:, i] * W[:, i, j] * x2[:, j])
		# Reshape weight for batch operations
		# weight: (out, in1, in2) -> (out, in1*in2)
		weight_flat = self.weight.data.reshape((self.out_features, -1))
		
		# Outer product of x1 and x2 for each batch
		# x1: (batch, in1) -> (batch, in1, 1)
		# x2: (batch, in2) -> (batch, 1, in2)
		# outer: (batch, in1, in2)
		backend = x1._backend
		x1_expanded = backend.expand_dims(x1.data, axis=-1)
		x2_expanded = backend.expand_dims(x2.data, axis=1)
		outer = backend.multiply(x1_expanded, x2_expanded)
		
		# Flatten outer product: (batch, in1*in2)
		from .. import Tensor
		outer_tensor = Tensor.__new__(Tensor)
		outer_tensor._backend = backend
		outer_tensor._dtype = x1._dtype
		outer_tensor.device = x1.device
		outer_tensor.active_device = x1.active_device
		outer_tensor.data = backend.reshape(outer, (batch_size, -1))
		outer_tensor._requires_grad = False
		outer_tensor._grad = None
		
		# Matrix multiply with weight: (batch, in1*in2) @ (in1*in2, out).T
		output = engine.matmul(outer_tensor, weight_flat.T())
		
		# Add bias
		if self.bias is not None:
			output = engine.add(output, self.bias.data)
		
		return output
	
	def extra_repr(self):
		return (f"in1_features={self.in1_features}, in2_features={self.in2_features}, "
				f"out_features={self.out_features}, bias={self.bias is not None}")


class LazyLinear(Module):
	
	def __init__(self, out_features, bias=True):
		super().__init__()
		self.out_features = out_features
		self.in_features = None
		self._bias_enabled = bias
		
		# Will be initialized on first forward pass
		self.weight = None
		self.bias = None
	
	def _initialize_parameters(self, in_features):
		# Initialize weight and bias once we know input size
		self.in_features = in_features
		
		from .. import Tensor
		import numpy as np
		
		# Kaiming initialization
		limit = math.sqrt(1.0 / in_features)
		weight_data = np.random.uniform(-limit, limit, (self.out_features, in_features))
		self.weight = Parameter(Tensor(weight_data, requires_grad=True))
		
		if self._bias_enabled:
			bias_data = np.random.uniform(-limit, limit, (self.out_features,))
			self.bias = Parameter(Tensor(bias_data, requires_grad=True))
	
	def forward(self, x):
		# Lazy initialization on first forward pass
		if self.weight is None:
			self._initialize_parameters(x.shape[-1])
		
		# Use same logic as Linear.forward
		from .. import engine
		
		input_shape = x.shape
		if len(input_shape) == 1:
			x = x.reshape((1, -1))
			single_sample = True
		else:
			single_sample = False
		
		if x.shape[-1] != self.in_features:
			raise ValueError(
				f"Input dimension changed: expected {self.in_features}, got {x.shape[-1]}"
			)
		
		original_shape = x.shape
		if len(original_shape) > 2:
			batch_size = 1
			for dim in original_shape[:-1]:
				batch_size *= dim
			x = x.reshape((batch_size, self.in_features))
		
		output = engine.matmul(x, self.weight.data.T())
		
		if self.bias is not None:
			output = engine.add(output, self.bias.data)
		
		if len(original_shape) > 2:
			output_shape = list(original_shape[:-1]) + [self.out_features]
			output = output.reshape(tuple(output_shape))
		
		if single_sample:
			output = output.reshape((self.out_features,))
		
		return output
	
	def extra_repr(self):
		in_features_str = self.in_features if self.in_features is not None else 'uninitialized'
		return f"in_features={in_features_str}, out_features={self.out_features}, bias={self._bias_enabled}"


# Utility Functions

def init_xavier_uniform_(tensor, gain=1.0):
	# Xavier/Glorot uniform initialization
	# Good for sigmoid/tanh activations
	from .. import Tensor
	import numpy as np
	
	fan_in, fan_out = tensor.shape[-2], tensor.shape[-1]
	std = gain * math.sqrt(2.0 / (fan_in + fan_out))
	limit = math.sqrt(3.0) * std
	
	data = np.random.uniform(-limit, limit, tensor.shape)
	tensor.data.data = tensor.data._backend.asarray(data)


def init_xavier_normal_(tensor, gain=1.0):
	# Xavier/Glorot normal initialization
	from .. import Tensor
	import numpy as np
	
	fan_in, fan_out = tensor.shape[-2], tensor.shape[-1]
	std = gain * math.sqrt(2.0 / (fan_in + fan_out))
	
	data = np.random.normal(0, std, tensor.shape)
	tensor.data.data = tensor.data._backend.asarray(data)


def init_kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
	# Kaiming/He uniform initialization
	# Good for ReLU activations
	from .. import Tensor
	import numpy as np
	
	fan = _calculate_fan(tensor.shape, mode)
	gain = _calculate_gain(nonlinearity, a)
	std = gain / math.sqrt(fan)
	limit = math.sqrt(3.0) * std
	
	data = np.random.uniform(-limit, limit, tensor.shape)
	tensor.data.data = tensor.data._backend.asarray(data)


def init_kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
	# Kaiming/He normal initialization
	from .. import Tensor
	import numpy as np
	
	fan = _calculate_fan(tensor.shape, mode)
	gain = _calculate_gain(nonlinearity, a)
	std = gain / math.sqrt(fan)
	
	data = np.random.normal(0, std, tensor.shape)
	tensor.data.data = tensor.data._backend.asarray(data)


def _calculate_fan(shape, mode='fan_in'):
	# Calculate fan_in or fan_out for initialization
	if len(shape) < 2:
		raise ValueError("Shape must have at least 2 dimensions")
	
	fan_in = shape[-2]
	fan_out = shape[-1]
	
	if mode == 'fan_in':
		return fan_in
	elif mode == 'fan_out':
		return fan_out
	elif mode == 'fan_avg':
		return (fan_in + fan_out) / 2.0
	else:
		raise ValueError(f"Invalid mode: {mode}")


def _calculate_gain(nonlinearity, param=None):
	# Calculate gain factor for different activations
	gains = {
		'linear': 1.0,
		'sigmoid': 1.0,
		'tanh': 5.0 / 3.0,
		'relu': math.sqrt(2.0),
		'leaky_relu': math.sqrt(2.0 / (1 + (param or 0.01) ** 2)),
		'selu': 3.0 / 4.0,
	}
	
	if nonlinearity in gains:
		return gains[nonlinearity]
	else:
		return 1.0


__all__ = [
	'Linear',
	'Bilinear',
	'LazyLinear',
	'init_xavier_uniform_',
	'init_xavier_normal_',
	'init_kaiming_uniform_',
	'init_kaiming_normal_',
]