from .module import Module
import math


class Upsample(Module):
	
	def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
		super().__init__()
		
		if size is None and scale_factor is None:
			raise ValueError("Either size or scale_factor must be specified")
		if size is not None and scale_factor is not None:
			raise ValueError("Only one of size or scale_factor should be specified")
		
		self.size = size
		self.scale_factor = scale_factor
		self.mode = mode
		self.align_corners = align_corners
		
		if mode not in ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear']:
			raise ValueError(f"Mode {mode} is not supported")
	
	def forward(self, x):
		import numpy as np
		backend = x._backend
		
		# Determine output size
		if self.size is not None:
			output_size = self.size if isinstance(self.size, (tuple, list)) else (self.size,)
		else:
			scale = self.scale_factor if isinstance(self.scale_factor, (tuple, list)) else (self.scale_factor,)
			input_shape = x.shape[2:]  # Spatial dimensions
			output_size = tuple(int(s * sc) for s, sc in zip(input_shape, scale))
		
		if len(x.shape) == 3:
			# 1D input: (batch, channels, length)
			return self._upsample_1d(x, output_size[0])
		elif len(x.shape) == 4:
			# 2D input: (batch, channels, height, width)
			return self._upsample_2d(x, output_size)
		elif len(x.shape) == 5:
			# 3D input: (batch, channels, depth, height, width)
			return self._upsample_3d(x, output_size)
		else:
			raise ValueError(f"Expected 3D, 4D or 5D input, got {len(x.shape)}D")
	
	def _upsample_1d(self, x, output_length):
		import numpy as np
		backend = x._backend
		
		batch_size, channels, input_length = x.shape
		x_np = x.numpy()
		
		if self.mode == 'nearest':
			# Nearest neighbor upsampling
			indices = np.linspace(0, input_length - 1, output_length)
			indices = np.round(indices).astype(int)
			output_np = x_np[:, :, indices]
		else:  # linear
			# Linear interpolation
			output_np = np.zeros((batch_size, channels, output_length))
			for b in range(batch_size):
				for c in range(channels):
					output_np[b, c] = np.interp(
						np.linspace(0, input_length - 1, output_length),
						np.arange(input_length),
						x_np[b, c]
					)
		
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
	
	def _upsample_2d(self, x, output_size):
		import numpy as np
		from scipy.ndimage import zoom
		backend = x._backend
		
		batch_size, channels, input_h, input_w = x.shape
		output_h, output_w = output_size
		
		x_np = x.numpy()
		output_np = np.zeros((batch_size, channels, output_h, output_w))
		
		if self.mode == 'nearest':
			# Nearest neighbor
			for b in range(batch_size):
				for c in range(channels):
					zoom_factors = (output_h / input_h, output_w / input_w)
					output_np[b, c] = zoom(x_np[b, c], zoom_factors, order=0)
		elif self.mode == 'bilinear':
			# Bilinear interpolation
			for b in range(batch_size):
				for c in range(channels):
					zoom_factors = (output_h / input_h, output_w / input_w)
					output_np[b, c] = zoom(x_np[b, c], zoom_factors, order=1)
		elif self.mode == 'bicubic':
			# Bicubic interpolation
			for b in range(batch_size):
				for c in range(channels):
					zoom_factors = (output_h / input_h, output_w / input_w)
					output_np[b, c] = zoom(x_np[b, c], zoom_factors, order=3)
		
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
	
	def _upsample_3d(self, x, output_size):
		import numpy as np
		from scipy.ndimage import zoom
		backend = x._backend
		
		batch_size, channels, input_d, input_h, input_w = x.shape
		output_d, output_h, output_w = output_size
		
		x_np = x.numpy()
		output_np = np.zeros((batch_size, channels, output_d, output_h, output_w))
		
		order = 0 if self.mode == 'nearest' else 1 if self.mode == 'trilinear' else 3
		
		for b in range(batch_size):
			for c in range(channels):
				zoom_factors = (output_d / input_d, output_h / input_h, output_w / input_w)
				output_np[b, c] = zoom(x_np[b, c], zoom_factors, order=order)
		
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
		if self.size is not None:
			return f"size={self.size}, mode='{self.mode}'"
		else:
			return f"scale_factor={self.scale_factor}, mode='{self.mode}'"


class UpsamplingNearest2d(Module):
	
	def __init__(self, size=None, scale_factor=None):
		super().__init__()
		self.size = size
		self.scale_factor = scale_factor
		self.upsample = Upsample(size=size, scale_factor=scale_factor, mode='nearest')
	
	def forward(self, x):
		return self.upsample(x)
	
	def extra_repr(self):
		if self.size is not None:
			return f"size={self.size}"
		else:
			return f"scale_factor={self.scale_factor}"


class UpsamplingBilinear2d(Module):
	
	def __init__(self, size=None, scale_factor=None):
		super().__init__()
		self.size = size
		self.scale_factor = scale_factor
		self.upsample = Upsample(size=size, scale_factor=scale_factor, mode='bilinear')
	
	def forward(self, x):
		return self.upsample(x)
	
	def extra_repr(self):
		if self.size is not None:
			return f"size={self.size}"
		else:
			return f"scale_factor={self.scale_factor}"


class PixelShuffle(Module):
	
	def __init__(self, upscale_factor):
		super().__init__()
		self.upscale_factor = upscale_factor
	
	def forward(self, x):
		# x: (batch, channels * r^2, height, width)
		# output: (batch, channels, height * r, width * r)
		
		import numpy as np
		backend = x._backend
		
		batch_size, channels, height, width = x.shape
		r = self.upscale_factor
		
		if channels % (r * r) != 0:
			raise ValueError(
				f"Number of channels must be divisible by upscale_factor^2 ({r*r}), "
				f"got {channels}"
			)
		
		out_channels = channels // (r * r)
		
		x_np = x.numpy()
		
		# Reshape and permute
		# (B, C*r^2, H, W) -> (B, C, r, r, H, W) -> (B, C, H, r, W, r) -> (B, C, H*r, W*r)
		output_np = x_np.reshape(batch_size, out_channels, r, r, height, width)
		output_np = output_np.transpose(0, 1, 4, 2, 5, 3)
		output_np = output_np.reshape(batch_size, out_channels, height * r, width * r)
		
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
		return f"upscale_factor={self.upscale_factor}"


class PixelUnshuffle(Module):
	
	def __init__(self, downscale_factor):
		super().__init__()
		self.downscale_factor = downscale_factor
	
	def forward(self, x):
		# x: (batch, channels, height, width)
		# output: (batch, channels * r^2, height // r, width // r)
		
		import numpy as np
		backend = x._backend
		
		batch_size, channels, height, width = x.shape
		r = self.downscale_factor
		
		if height % r != 0 or width % r != 0:
			raise ValueError(
				f"Height and width must be divisible by downscale_factor ({r}), "
				f"got {height}x{width}"
			)
		
		out_channels = channels * (r * r)
		
		x_np = x.numpy()
		
		# Reshape and permute (inverse of PixelShuffle)
		output_np = x_np.reshape(batch_size, channels, height // r, r, width // r, r)
		output_np = output_np.transpose(0, 1, 3, 5, 2, 4)
		output_np = output_np.reshape(batch_size, out_channels, height // r, width // r)
		
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
		return f"downscale_factor={self.downscale_factor}"


class Interpolate(Module):
	
	def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
		super().__init__()
		self.upsample = Upsample(size, scale_factor, mode, align_corners)
	
	def forward(self, x):
		return self.upsample(x)


# Utility function for functional-style usage
def interpolate(x, size=None, scale_factor=None, mode='nearest', align_corners=None):
	"""Functional interface for interpolation"""
	upsample = Upsample(size, scale_factor, mode, align_corners)
	return upsample(x)


__all__ = [
	'Upsample',
	'UpsamplingNearest2d',
	'UpsamplingBilinear2d',
	'PixelShuffle',
	'PixelUnshuffle',
	'Interpolate',
	'interpolate',
]