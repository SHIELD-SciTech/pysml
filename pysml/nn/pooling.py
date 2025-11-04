from .module import Module
import math


class MaxPool1d(Module):
	
	def __init__(self, kernel_size, stride=None, padding=0, dilation=1, 
				 return_indices=False, ceil_mode=False):
		super().__init__()
		self.kernel_size = kernel_size
		self.stride = stride if stride is not None else kernel_size
		self.padding = padding
		self.dilation = dilation
		self.return_indices = return_indices
		self.ceil_mode = ceil_mode
	
	def forward(self, x):
		# x: (batch, channels, length)
		import numpy as np
		backend = x._backend
		
		batch_size, channels, length = x.shape
		
		# Pad input
		if self.padding > 0:
			x_np = x.numpy()
			x_np = np.pad(x_np, ((0, 0), (0, 0), (self.padding, self.padding)), 
						 mode='constant', constant_values=-np.inf)
		else:
			x_np = x.numpy()
		
		# Calculate output length
		if self.ceil_mode:
			length_out = math.ceil((length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)
		else:
			length_out = math.floor((length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)
		
		output_np = np.zeros((batch_size, channels, length_out))
		indices_np = np.zeros((batch_size, channels, length_out), dtype=np.int64) if self.return_indices else None
		
		# Max pooling
		for b in range(batch_size):
			for c in range(channels):
				for i in range(length_out):
					start = i * self.stride
					end = start + self.kernel_size * self.dilation
					pool_region = x_np[b, c, start:end:self.dilation]
					
					if len(pool_region) > 0:
						max_val = np.max(pool_region)
						output_np[b, c, i] = max_val
						if self.return_indices:
							indices_np[b, c, i] = start + np.argmax(pool_region) * self.dilation
		
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
		
		if self.return_indices:
			indices = Tensor.__new__(Tensor)
			indices._backend = backend
			indices._dtype = x._dtype
			indices.device = x.device
			indices.active_device = x.active_device
			indices.data = backend.asarray(indices_np)
			indices._requires_grad = False
			indices._grad = None
			return output, indices
		
		return output
	
	def extra_repr(self):
		return (f"kernel_size={self.kernel_size}, stride={self.stride}, "
				f"padding={self.padding}, dilation={self.dilation}")


class MaxPool2d(Module):
	
	def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
				 return_indices=False, ceil_mode=False):
		super().__init__()
		
		self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
		self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride) if stride is not None else self.kernel_size
		self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
		self.dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
		self.return_indices = return_indices
		self.ceil_mode = ceil_mode
	
	def forward(self, x):
		# x: (batch, channels, height, width)
		import numpy as np
		backend = x._backend
		
		batch_size, channels, height, width = x.shape
		kh, kw = self.kernel_size
		sh, sw = self.stride
		ph, pw = self.padding
		dh, dw = self.dilation
		
		# Pad input
		if ph > 0 or pw > 0:
			x_np = x.numpy()
			x_np = np.pad(x_np, ((0, 0), (0, 0), (ph, ph), (pw, pw)),
						 mode='constant', constant_values=-np.inf)
		else:
			x_np = x.numpy()
		
		# Calculate output dimensions
		if self.ceil_mode:
			height_out = math.ceil((height + 2 * ph - dh * (kh - 1) - 1) / sh + 1)
			width_out = math.ceil((width + 2 * pw - dw * (kw - 1) - 1) / sw + 1)
		else:
			height_out = math.floor((height + 2 * ph - dh * (kh - 1) - 1) / sh + 1)
			width_out = math.floor((width + 2 * pw - dw * (kw - 1) - 1) / sw + 1)
		
		output_np = np.zeros((batch_size, channels, height_out, width_out))
		indices_np = np.zeros((batch_size, channels, height_out, width_out), dtype=np.int64) if self.return_indices else None
		
		# Max pooling
		for b in range(batch_size):
			for c in range(channels):
				for h_out in range(height_out):
					for w_out in range(width_out):
						h_start = h_out * sh
						w_start = w_out * sw
						h_end = h_start + kh * dh
						w_end = w_start + kw * dw
						
						pool_region = x_np[b, c, h_start:h_end:dh, w_start:w_end:dw]
						
						if pool_region.size > 0:
							max_val = np.max(pool_region)
							output_np[b, c, h_out, w_out] = max_val
							
							if self.return_indices:
								max_idx = np.argmax(pool_region)
								h_offset, w_offset = np.unravel_index(max_idx, pool_region.shape)
								indices_np[b, c, h_out, w_out] = (h_start + h_offset * dh) * width + (w_start + w_offset * dw)
		
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
		
		if self.return_indices:
			indices = Tensor.__new__(Tensor)
			indices._backend = backend
			indices._dtype = x._dtype
			indices.device = x.device
			indices.active_device = x.active_device
			indices.data = backend.asarray(indices_np)
			indices._requires_grad = False
			indices._grad = None
			return output, indices
		
		return output
	
	def extra_repr(self):
		return (f"kernel_size={self.kernel_size}, stride={self.stride}, "
				f"padding={self.padding}, dilation={self.dilation}")


class AvgPool1d(Module):
	
	def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, 
				 count_include_pad=True):
		super().__init__()
		self.kernel_size = kernel_size
		self.stride = stride if stride is not None else kernel_size
		self.padding = padding
		self.ceil_mode = ceil_mode
		self.count_include_pad = count_include_pad
	
	def forward(self, x):
		# x: (batch, channels, length)
		import numpy as np
		backend = x._backend
		
		batch_size, channels, length = x.shape
		
		# Pad input
		if self.padding > 0:
			x_np = x.numpy()
			x_np = np.pad(x_np, ((0, 0), (0, 0), (self.padding, self.padding)), mode='constant')
		else:
			x_np = x.numpy()
		
		# Calculate output length
		if self.ceil_mode:
			length_out = math.ceil((length + 2 * self.padding - self.kernel_size) / self.stride + 1)
		else:
			length_out = math.floor((length + 2 * self.padding - self.kernel_size) / self.stride + 1)
		
		output_np = np.zeros((batch_size, channels, length_out))
		
		# Average pooling
		for b in range(batch_size):
			for c in range(channels):
				for i in range(length_out):
					start = i * self.stride
					end = start + self.kernel_size
					pool_region = x_np[b, c, start:end]
					
					if len(pool_region) > 0:
						if self.count_include_pad:
							output_np[b, c, i] = np.sum(pool_region) / self.kernel_size
						else:
							# Only count non-padded elements
							valid_count = np.sum(pool_region != 0)
							output_np[b, c, i] = np.sum(pool_region) / max(valid_count, 1)
		
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
		return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class AvgPool2d(Module):
	
	def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
				 count_include_pad=True, divisor_override=None):
		super().__init__()
		
		self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
		self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride) if stride is not None else self.kernel_size
		self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
		self.ceil_mode = ceil_mode
		self.count_include_pad = count_include_pad
		self.divisor_override = divisor_override
	
	def forward(self, x):
		# x: (batch, channels, height, width)
		import numpy as np
		backend = x._backend
		
		batch_size, channels, height, width = x.shape
		kh, kw = self.kernel_size
		sh, sw = self.stride
		ph, pw = self.padding
		
		# Pad input
		if ph > 0 or pw > 0:
			x_np = x.numpy()
			x_np = np.pad(x_np, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode='constant')
		else:
			x_np = x.numpy()
		
		# Calculate output dimensions
		if self.ceil_mode:
			height_out = math.ceil((height + 2 * ph - kh) / sh + 1)
			width_out = math.ceil((width + 2 * pw - kw) / sw + 1)
		else:
			height_out = math.floor((height + 2 * ph - kh) / sh + 1)
			width_out = math.floor((width + 2 * pw - kw) / sw + 1)
		
		output_np = np.zeros((batch_size, channels, height_out, width_out))
		
		# Average pooling
		for b in range(batch_size):
			for c in range(channels):
				for h_out in range(height_out):
					for w_out in range(width_out):
						h_start = h_out * sh
						w_start = w_out * sw
						h_end = h_start + kh
						w_end = w_start + kw
						
						pool_region = x_np[b, c, h_start:h_end, w_start:w_end]
						
						if pool_region.size > 0:
							if self.divisor_override:
								divisor = self.divisor_override
							elif self.count_include_pad:
								divisor = kh * kw
							else:
								divisor = pool_region.size
							
							output_np[b, c, h_out, w_out] = np.sum(pool_region) / divisor
		
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
		return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class AdaptiveAvgPool1d(Module):
	
	def __init__(self, output_size):
		super().__init__()
		self.output_size = output_size
	
	def forward(self, x):
		# x: (batch, channels, length)
		import numpy as np
		backend = x._backend
		
		batch_size, channels, length = x.shape
		output_size = self.output_size
		
		x_np = x.numpy()
		output_np = np.zeros((batch_size, channels, output_size))
		
		# Adaptive pooling: divide input into output_size regions
		for b in range(batch_size):
			for c in range(channels):
				for i in range(output_size):
					start = int(math.floor(i * length / output_size))
					end = int(math.ceil((i + 1) * length / output_size))
					
					pool_region = x_np[b, c, start:end]
					output_np[b, c, i] = np.mean(pool_region)
		
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
		return f"output_size={self.output_size}"


class AdaptiveAvgPool2d(Module):
	
	def __init__(self, output_size):
		super().__init__()
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			self.output_size = tuple(output_size)
	
	def forward(self, x):
		# x: (batch, channels, height, width)
		import numpy as np
		backend = x._backend
		
		batch_size, channels, height, width = x.shape
		output_h, output_w = self.output_size
		
		x_np = x.numpy()
		output_np = np.zeros((batch_size, channels, output_h, output_w))
		
		# Adaptive pooling
		for b in range(batch_size):
			for c in range(channels):
				for h_out in range(output_h):
					for w_out in range(output_w):
						h_start = int(math.floor(h_out * height / output_h))
						h_end = int(math.ceil((h_out + 1) * height / output_h))
						w_start = int(math.floor(w_out * width / output_w))
						w_end = int(math.ceil((w_out + 1) * width / output_w))
						
						pool_region = x_np[b, c, h_start:h_end, w_start:w_end]
						output_np[b, c, h_out, w_out] = np.mean(pool_region)
		
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
		return f"output_size={self.output_size}"


class AdaptiveMaxPool1d(Module):
	
	def __init__(self, output_size, return_indices=False):
		super().__init__()
		self.output_size = output_size
		self.return_indices = return_indices
	
	def forward(self, x):
		# x: (batch, channels, length)
		import numpy as np
		backend = x._backend
		
		batch_size, channels, length = x.shape
		output_size = self.output_size
		
		x_np = x.numpy()
		output_np = np.zeros((batch_size, channels, output_size))
		indices_np = np.zeros((batch_size, channels, output_size), dtype=np.int64) if self.return_indices else None
		
		for b in range(batch_size):
			for c in range(channels):
				for i in range(output_size):
					start = int(math.floor(i * length / output_size))
					end = int(math.ceil((i + 1) * length / output_size))
					
					pool_region = x_np[b, c, start:end]
					output_np[b, c, i] = np.max(pool_region)
					if self.return_indices:
						indices_np[b, c, i] = start + np.argmax(pool_region)
		
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
		
		if self.return_indices:
			indices = Tensor.__new__(Tensor)
			indices._backend = backend
			indices._dtype = x._dtype
			indices.device = x.device
			indices.active_device = x.active_device
			indices.data = backend.asarray(indices_np)
			indices._requires_grad = False
			indices._grad = None
			return output, indices
		
		return output
	
	def extra_repr(self):
		return f"output_size={self.output_size}"


class AdaptiveMaxPool2d(Module):
	
	def __init__(self, output_size, return_indices=False):
		super().__init__()
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			self.output_size = tuple(output_size)
		self.return_indices = return_indices
	
	def forward(self, x):
		# x: (batch, channels, height, width)
		import numpy as np
		backend = x._backend
		
		batch_size, channels, height, width = x.shape
		output_h, output_w = self.output_size
		
		x_np = x.numpy()
		output_np = np.zeros((batch_size, channels, output_h, output_w))
		indices_np = np.zeros((batch_size, channels, output_h, output_w), dtype=np.int64) if self.return_indices else None
		
		for b in range(batch_size):
			for c in range(channels):
				for h_out in range(output_h):
					for w_out in range(output_w):
						h_start = int(math.floor(h_out * height / output_h))
						h_end = int(math.ceil((h_out + 1) * height / output_h))
						w_start = int(math.floor(w_out * width / output_w))
						w_end = int(math.ceil((w_out + 1) * width / output_w))
						
						pool_region = x_np[b, c, h_start:h_end, w_start:w_end]
						output_np[b, c, h_out, w_out] = np.max(pool_region)
						
						if self.return_indices:
							max_idx = np.argmax(pool_region)
							h_offset, w_offset = np.unravel_index(max_idx, pool_region.shape)
							indices_np[b, c, h_out, w_out] = (h_start + h_offset) * width + (w_start + w_offset)
		
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
		
		if self.return_indices:
			indices = Tensor.__new__(Tensor)
			indices._backend = backend
			indices._dtype = x._dtype
			indices.device = x.device
			indices.active_device = x.active_device
			indices.data = backend.asarray(indices_np)
			indices._requires_grad = False
			indices._grad = None
			return output, indices
		
		return output
	
	def extra_repr(self):
		return f"output_size={self.output_size}"


class GlobalAvgPool2d(Module):
	
	def __init__(self):
		super().__init__()
	
	def forward(self, x):
		# x: (batch, channels, height, width) -> (batch, channels, 1, 1)
		from .. import engine
		return engine.mean_with_grad(x, axis=(2, 3), keepdims=True)


class GlobalMaxPool2d(Module):
	
	def __init__(self):
		super().__init__()
	
	def forward(self, x):
		# x: (batch, channels, height, width) -> (batch, channels, 1, 1)
		backend = x._backend
		from .. import Tensor
		
		output = Tensor.__new__(Tensor)
		output._backend = backend
		output._dtype = x._dtype
		output.device = x.device
		output.active_device = x.active_device
		output.data = backend.max(x.data, axis=(2, 3), keepdims=True)
		output._requires_grad = x._requires_grad
		output._grad = None
		
		return output


__all__ = [
	'MaxPool1d', 'MaxPool2d',
	'AvgPool1d', 'AvgPool2d',
	'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d',
	'AdaptiveMaxPool1d', 'AdaptiveMaxPool2d',
	'GlobalAvgPool2d', 'GlobalMaxPool2d',
]