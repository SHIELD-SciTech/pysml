from .module import Module, Parameter
import math


class Conv1d(Module):
	
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
				 padding=0, dilation=1, groups=1, bias=True):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
		self.stride = stride if isinstance(stride, tuple) else (stride,)
		self.padding = padding if isinstance(padding, tuple) else (padding,)
		self.dilation = dilation if isinstance(dilation, tuple) else (dilation,)
		self.groups = groups
		
		if in_channels % groups != 0:
			raise ValueError("in_channels must be divisible by groups")
		if out_channels % groups != 0:
			raise ValueError("out_channels must be divisible by groups")
		
		# Initialize weights
		self.weight = Parameter(self._initialize_weight())
		
		if bias:
			self.bias = Parameter(self._initialize_bias())
		else:
			self.bias = None
	
	def _initialize_weight(self):
		from .. import Tensor
		import numpy as np
		
		# Kaiming initialization
		k = self.kernel_size[0]
		fan_in = self.in_channels * k
		limit = math.sqrt(1.0 / fan_in)
		
		shape = (self.out_channels, self.in_channels // self.groups, k)
		data = np.random.uniform(-limit, limit, shape)
		return Tensor(data, requires_grad=True)
	
	def _initialize_bias(self):
		from .. import Tensor
		import numpy as np
		
		k = self.kernel_size[0]
		fan_in = self.in_channels * k
		limit = math.sqrt(1.0 / fan_in)
		
		data = np.random.uniform(-limit, limit, (self.out_channels,))
		return Tensor(data, requires_grad=True)
	
	def forward(self, x):
		# x: (batch, in_channels, length)
		# output: (batch, out_channels, length_out)
		
		# Use im2col convolution (efficient on CPU/GPU)
		# This is a simplified implementation
		# Production version would use optimized backend kernels
		
		import numpy as np
		backend = x._backend
		
		batch_size, in_channels, length = x.shape
		k = self.kernel_size[0]
		s = self.stride[0]
		p = self.padding[0]
		d = self.dilation[0]
		
		# Calculate output length
		length_out = (length + 2 * p - d * (k - 1) - 1) // s + 1
		
		# Convert to numpy for convolution (temporary)
		x_np = x.numpy()
		weight_np = self.weight.data.numpy()
		
		# Pad input
		if p > 0:
			x_np = np.pad(x_np, ((0, 0), (0, 0), (p, p)), mode='constant')
		
		# im2col transformation
		output_np = np.zeros((batch_size, self.out_channels, length_out))
		
		for b in range(batch_size):
			for oc in range(self.out_channels):
				for i in range(length_out):
					start = i * s
					receptive_field = []
					for ic in range(self.in_channels // self.groups):
						field = x_np[b, ic, start:start + k * d:d]
						if len(field) < k:
							field = np.pad(field, (0, k - len(field)))
						receptive_field.append(field)
					
					receptive_field = np.array(receptive_field).flatten()
					weight_flat = weight_np[oc].flatten()
					
					# Ensure same size
					min_len = min(len(receptive_field), len(weight_flat))
					output_np[b, oc, i] = np.dot(receptive_field[:min_len], weight_flat[:min_len])
		
		# Add bias
		if self.bias is not None:
			bias_np = self.bias.data.numpy()
			output_np += bias_np.reshape(1, -1, 1)
		
		# Convert back to tensor
		from .. import Tensor
		output = Tensor.__new__(Tensor)
		output._backend = backend
		output._dtype = x._dtype
		output.device = x.device
		output.active_device = x.active_device
		output.data = backend.asarray(output_np)
		output._requires_grad = x._requires_grad
		output._grad = None
		
		return output
	
	def extra_repr(self):
		s = (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size[0]}, "
			 f"stride={self.stride[0]}")
		if self.padding[0] != 0:
			s += f", padding={self.padding[0]}"
		if self.dilation[0] != 1:
			s += f", dilation={self.dilation[0]}"
		if self.groups != 1:
			s += f", groups={self.groups}"
		if self.bias is None:
			s += ", bias=False"
		return s


class Conv2d(Module):
	
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
				 padding=0, dilation=1, groups=1, bias=True):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		
		# Convert to tuple if needed
		self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
		self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
		self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
		self.dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
		self.groups = groups
		
		if in_channels % groups != 0:
			raise ValueError("in_channels must be divisible by groups")
		if out_channels % groups != 0:
			raise ValueError("out_channels must be divisible by groups")
		
		# Initialize weights
		self.weight = Parameter(self._initialize_weight())
		
		if bias:
			self.bias = Parameter(self._initialize_bias())
		else:
			self.bias = None
	
	def _initialize_weight(self):
		from .. import Tensor
		import numpy as np
		
		# Kaiming initialization
		kh, kw = self.kernel_size
		fan_in = self.in_channels * kh * kw
		limit = math.sqrt(1.0 / fan_in)
		
		shape = (self.out_channels, self.in_channels // self.groups, kh, kw)
		data = np.random.uniform(-limit, limit, shape)
		return Tensor(data, requires_grad=True)
	
	def _initialize_bias(self):
		from .. import Tensor
		import numpy as np
		
		kh, kw = self.kernel_size
		fan_in = self.in_channels * kh * kw
		limit = math.sqrt(1.0 / fan_in)
		
		data = np.random.uniform(-limit, limit, (self.out_channels,))
		return Tensor(data, requires_grad=True)
	
	def forward(self, x):
		# x: (batch, in_channels, height, width)
		# output: (batch, out_channels, height_out, width_out)
		
		import numpy as np
		backend = x._backend
		
		batch_size, in_channels, height, width = x.shape
		kh, kw = self.kernel_size
		sh, sw = self.stride
		ph, pw = self.padding
		dh, dw = self.dilation
		
		# Calculate output dimensions
		height_out = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
		width_out = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
		
		# Convert to numpy (temporary - would use optimized kernels in production)
		x_np = x.numpy()
		weight_np = self.weight.data.numpy()
		
		# Pad input
		if ph > 0 or pw > 0:
			x_np = np.pad(x_np, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode='constant')
		
		# Convolution (naive implementation - would use im2col + gemm in production)
		output_np = np.zeros((batch_size, self.out_channels, height_out, width_out))
		
		for b in range(batch_size):
			for oc in range(self.out_channels):
				for h_out in range(height_out):
					for w_out in range(width_out):
						h_start = h_out * sh
						w_start = w_out * sw
						
						# Extract receptive field
						h_end = h_start + kh * dh
						w_end = w_start + kw * dw
						
						receptive_field = x_np[
							b,
							:self.in_channels // self.groups,
							h_start:h_end:dh,
							w_start:w_end:dw
						]
						
						# Convolve with kernel
						kernel = weight_np[oc]
						
						# Handle edge cases
						if receptive_field.shape != kernel.shape:
							min_h = min(receptive_field.shape[1], kernel.shape[1])
							min_w = min(receptive_field.shape[2], kernel.shape[2])
							receptive_field = receptive_field[:, :min_h, :min_w]
							kernel = kernel[:, :min_h, :min_w]
						
						output_np[b, oc, h_out, w_out] = np.sum(receptive_field * kernel)
		
		# Add bias
		if self.bias is not None:
			bias_np = self.bias.data.numpy()
			output_np += bias_np.reshape(1, -1, 1, 1)
		
		# Convert back to tensor
		from .. import Tensor
		output = Tensor.__new__(Tensor)
		output._backend = backend
		output._dtype = x._dtype
		output.device = x.device
		output.active_device = x.active_device
		output.data = backend.asarray(output_np)
		output._requires_grad = x._requires_grad
		output._grad = None
		
		return output
	
	def extra_repr(self):
		s = (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
			 f"stride={self.stride}")
		if self.padding != (0, 0):
			s += f", padding={self.padding}"
		if self.dilation != (1, 1):
			s += f", dilation={self.dilation}"
		if self.groups != 1:
			s += f", groups={self.groups}"
		if self.bias is None:
			s += ", bias=False"
		return s


class Conv3d(Module):
	
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
				 padding=0, dilation=1, groups=1, bias=True):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		
		# Convert to 3-tuple
		if isinstance(kernel_size, int):
			self.kernel_size = (kernel_size, kernel_size, kernel_size)
		else:
			self.kernel_size = tuple(kernel_size)
		
		if isinstance(stride, int):
			self.stride = (stride, stride, stride)
		else:
			self.stride = tuple(stride)
		
		if isinstance(padding, int):
			self.padding = (padding, padding, padding)
		else:
			self.padding = tuple(padding)
		
		if isinstance(dilation, int):
			self.dilation = (dilation, dilation, dilation)
		else:
			self.dilation = tuple(dilation)
		
		self.groups = groups
		
		# Initialize weights
		self.weight = Parameter(self._initialize_weight())
		
		if bias:
			self.bias = Parameter(self._initialize_bias())
		else:
			self.bias = None
	
	def _initialize_weight(self):
		from .. import Tensor
		import numpy as np
		
		kd, kh, kw = self.kernel_size
		fan_in = self.in_channels * kd * kh * kw
		limit = math.sqrt(1.0 / fan_in)
		
		shape = (self.out_channels, self.in_channels // self.groups, kd, kh, kw)
		data = np.random.uniform(-limit, limit, shape)
		return Tensor(data, requires_grad=True)
	
	def _initialize_bias(self):
		from .. import Tensor
		import numpy as np
		
		kd, kh, kw = self.kernel_size
		fan_in = self.in_channels * kd * kh * kw
		limit = math.sqrt(1.0 / fan_in)
		
		data = np.random.uniform(-limit, limit, (self.out_channels,))
		return Tensor(data, requires_grad=True)
	
	def forward(self, x):
		# x: (batch, in_channels, depth, height, width)
		# output: (batch, out_channels, depth_out, height_out, width_out)
		
		import numpy as np
		backend = x._backend
		
		batch_size, in_channels, depth, height, width = x.shape
		kd, kh, kw = self.kernel_size
		sd, sh, sw = self.stride
		pd, ph, pw = self.padding
		
		# Calculate output dimensions
		depth_out = (depth + 2 * pd - kd) // sd + 1
		height_out = (height + 2 * ph - kh) // sh + 1
		width_out = (width + 2 * pw - kw) // sw + 1
		
		# Simplified 3D convolution (would use optimized kernels in production)
		x_np = x.numpy()
		weight_np = self.weight.data.numpy()
		
		# Pad
		if pd > 0 or ph > 0 or pw > 0:
			x_np = np.pad(x_np, ((0, 0), (0, 0), (pd, pd), (ph, ph), (pw, pw)))
		
		output_np = np.zeros((batch_size, self.out_channels, depth_out, height_out, width_out))
		
		# Naive 3D convolution
		for b in range(batch_size):
			for oc in range(self.out_channels):
				for d_out in range(depth_out):
					for h_out in range(height_out):
						for w_out in range(width_out):
							d_start = d_out * sd
							h_start = h_out * sh
							w_start = w_out * sw
							
							receptive_field = x_np[
								b,
								:self.in_channels // self.groups,
								d_start:d_start + kd,
								h_start:h_start + kh,
								w_start:w_start + kw
							]
							
							kernel = weight_np[oc]
							
							# Handle shapes
							if receptive_field.shape == kernel.shape:
								output_np[b, oc, d_out, h_out, w_out] = np.sum(receptive_field * kernel)
		
		# Add bias
		if self.bias is not None:
			bias_np = self.bias.data.numpy()
			output_np += bias_np.reshape(1, -1, 1, 1, 1)
		
		# Convert back
		from .. import Tensor
		output = Tensor.__new__(Tensor)
		output._backend = backend
		output._dtype = x._dtype
		output.device = x.device
		output.active_device = x.active_device
		output.data = backend.asarray(output_np)
		output._requires_grad = x._requires_grad
		output._grad = None
		
		return output
	
	def extra_repr(self):
		s = (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
			 f"stride={self.stride}")
		if self.padding != (0, 0, 0):
			s += f", padding={self.padding}"
		if self.groups != 1:
			s += f", groups={self.groups}"
		if self.bias is None:
			s += ", bias=False"
		return s


class ConvTranspose2d(Module):
	
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
				 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		
		self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
		self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
		self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
		self.output_padding = (output_padding, output_padding) if isinstance(output_padding, int) else tuple(output_padding)
		self.dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
		self.groups = groups
		
		# Initialize weights (note: shape is transposed compared to Conv2d)
		self.weight = Parameter(self._initialize_weight())
		
		if bias:
			self.bias = Parameter(self._initialize_bias())
		else:
			self.bias = None
	
	def _initialize_weight(self):
		from .. import Tensor
		import numpy as np
		
		kh, kw = self.kernel_size
		fan_in = self.in_channels * kh * kw
		limit = math.sqrt(1.0 / fan_in)
		
		# Note: transposed conv has in_channels and out_channels swapped in weight shape
		shape = (self.in_channels, self.out_channels // self.groups, kh, kw)
		data = np.random.uniform(-limit, limit, shape)
		return Tensor(data, requires_grad=True)
	
	def _initialize_bias(self):
		from .. import Tensor
		import numpy as np
		
		kh, kw = self.kernel_size
		fan_in = self.in_channels * kh * kw
		limit = math.sqrt(1.0 / fan_in)
		
		data = np.random.uniform(-limit, limit, (self.out_channels,))
		return Tensor(data, requires_grad=True)
	
	def forward(self, x):
		# x: (batch, in_channels, height, width)
		# output: (batch, out_channels, height_out, width_out)
		
		# Transposed convolution (upsampling)
		# Simplified implementation
		
		import numpy as np
		backend = x._backend
		
		batch_size, in_channels, height, width = x.shape
		kh, kw = self.kernel_size
		sh, sw = self.stride
		ph, pw = self.padding
		oph, opw = self.output_padding
		
		# Calculate output size
		height_out = (height - 1) * sh - 2 * ph + kh + oph
		width_out = (width - 1) * sw - 2 * pw + kw + opw
		
		# Simplified transposed convolution
		x_np = x.numpy()
		weight_np = self.weight.data.numpy()
		
		output_np = np.zeros((batch_size, self.out_channels, height_out, width_out))
		
		# For each input position, spread to output
		for b in range(batch_size):
			for ic in range(in_channels):
				for h in range(height):
					for w in range(width):
						value = x_np[b, ic, h, w]
						
						# Calculate output position
						h_out_start = h * sh - ph
						w_out_start = w * sw - pw
						
						# Spread using kernel
						for oc in range(self.out_channels // self.groups):
							kernel = weight_np[ic, oc]
							
							for kh_i in range(kh):
								for kw_i in range(kw):
									h_out = h_out_start + kh_i
									w_out = w_out_start + kw_i
									
									if 0 <= h_out < height_out and 0 <= w_out < width_out:
										output_np[b, oc, h_out, w_out] += value * kernel[kh_i, kw_i]
		
		# Add bias
		if self.bias is not None:
			bias_np = self.bias.data.numpy()
			output_np += bias_np.reshape(1, -1, 1, 1)
		
		# Convert back
		from .. import Tensor
		output = Tensor.__new__(Tensor)
		output._backend = backend
		output._dtype = x._dtype
		output.device = x.device
		output.active_device = x.active_device
		output.data = backend.asarray(output_np)
		output._requires_grad = x._requires_grad
		output._grad = None
		
		return output
	
	def extra_repr(self):
		s = (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
			 f"stride={self.stride}")
		if self.padding != (0, 0):
			s += f", padding={self.padding}"
		if self.output_padding != (0, 0):
			s += f", output_padding={self.output_padding}"
		if self.groups != 1:
			s += f", groups={self.groups}"
		if self.bias is None:
			s += ", bias=False"
		return s


__all__ = [
	'Conv1d',
	'Conv2d',
	'Conv3d',
	'ConvTranspose2d',
]
