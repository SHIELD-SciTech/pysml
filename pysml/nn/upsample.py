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
		backend = x._backend

		spatial_dims = len(x.shape) - 2
		if spatial_dims <= 0:
			raise ValueError(f"Expected at least 3D input, got {len(x.shape)}D")

		if self.size is not None:
			if isinstance(self.size, (tuple, list)):
				output_size = tuple(self.size)
			else:
				output_size = (self.size,) * spatial_dims
		else:
			if isinstance(self.scale_factor, (tuple, list)):
				scale = tuple(self.scale_factor)
				if len(scale) == 1 and spatial_dims > 1:
					scale = scale * spatial_dims
				elif len(scale) != spatial_dims:
					scale = tuple(scale[i if i < len(scale) else -1] for i in range(spatial_dims))
			else:
				scale = (self.scale_factor,) * spatial_dims
			input_shape = x.shape[2:]
			output_size = tuple(int(round(dim * sc)) for dim, sc in zip(input_shape, scale))

		if len(output_size) != spatial_dims:
			raise ValueError(f"Output size {output_size} does not match spatial dims {spatial_dims}")

		if spatial_dims == 1:
			return self._upsample_1d(x, output_size[0])
		elif spatial_dims == 2:
			return self._upsample_2d(x, output_size)
		elif spatial_dims == 3:
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
		backend = x._backend

		batch_size, channels, input_h, input_w = x.shape
		output_h, output_w = output_size

		x_np = x.numpy()
		dtype = x_np.dtype

		if self.mode == 'nearest':
			h_idx = np.clip(np.round(np.linspace(0, max(input_h - 1, 0), output_h)), 0, max(input_h - 1, 0)).astype(np.int64)
			w_idx = np.clip(np.round(np.linspace(0, max(input_w - 1, 0), output_w)), 0, max(input_w - 1, 0)).astype(np.int64)
			output_np = np.take(x_np, h_idx, axis=2)
			output_np = np.take(output_np, w_idx, axis=3)
		elif self.mode in ('bilinear', 'bicubic'):
			def compute_positions(size_in, size_out):
				if size_out <= 1:
					return np.zeros((1,), dtype=np.float32)
				if self.align_corners:
					scale = (size_in - 1) / (size_out - 1) if size_out > 1 else 0.0
					return scale * np.arange(size_out, dtype=np.float32)
				scale = size_in / size_out
				return (np.arange(size_out, dtype=np.float32) + 0.5) * scale - 0.5

			pos_h = np.clip(compute_positions(input_h, output_h), 0, max(input_h - 1, 0)).astype(np.float32)
			pos_w = np.clip(compute_positions(input_w, output_w), 0, max(input_w - 1, 0)).astype(np.float32)

			top = np.floor(pos_h).astype(np.int64)
			bottom = np.clip(top + 1, 0, max(input_h - 1, 0))
			lh = (pos_h - top).astype(np.float32)

			left = np.floor(pos_w).astype(np.int64)
			right = np.clip(left + 1, 0, max(input_w - 1, 0))
			lw = (pos_w - left).astype(np.float32)

			lh = lh[:, None].astype(dtype, copy=False)
			lw = lw[None, :].astype(dtype, copy=False)
			top_idx = top[:, None]
			bottom_idx = bottom[:, None]
			left_idx = left[None, :]
			right_idx = right[None, :]

			one = np.array(1.0, dtype=dtype)
			output_np = np.zeros((batch_size, channels, output_h, output_w), dtype=dtype)
			for b in range(batch_size):
				for c in range(channels):
					img = x_np[b, c]
					top_left = img[top_idx, left_idx]
					top_right = img[top_idx, right_idx]
					bottom_left = img[bottom_idx, left_idx]
					bottom_right = img[bottom_idx, right_idx]

					top_mix = (one - lw) * top_left + lw * top_right
					bottom_mix = (one - lw) * bottom_left + lw * bottom_right
					output_np[b, c] = (one - lh) * top_mix + lh * bottom_mix
		else:
			raise ValueError(f"Unsupported mode '{self.mode}' for 2D upsampling")

		output_np = output_np.astype(dtype, copy=False)

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
		backend = x._backend

		batch_size, channels, input_d, input_h, input_w = x.shape
		output_d, output_h, output_w = output_size

		x_np = x.numpy()
		dtype = x_np.dtype

		if self.mode == 'nearest':
			d_idx = np.clip(np.round(np.linspace(0, max(input_d - 1, 0), output_d)), 0, max(input_d - 1, 0)).astype(np.int64)
			h_idx = np.clip(np.round(np.linspace(0, max(input_h - 1, 0), output_h)), 0, max(input_h - 1, 0)).astype(np.int64)
			w_idx = np.clip(np.round(np.linspace(0, max(input_w - 1, 0), output_w)), 0, max(input_w - 1, 0)).astype(np.int64)
			output_np = np.take(x_np, d_idx, axis=2)
			output_np = np.take(output_np, h_idx, axis=3)
			output_np = np.take(output_np, w_idx, axis=4)
		elif self.mode in ('trilinear',):
			def compute_positions(size_in, size_out):
				if size_out <= 1:
					return np.zeros((1,), dtype=np.float32)
				if self.align_corners:
					scale = (size_in - 1) / (size_out - 1) if size_out > 1 else 0.0
					return scale * np.arange(size_out, dtype=np.float32)
				scale = size_in / size_out
				return (np.arange(size_out, dtype=np.float32) + 0.5) * scale - 0.5

			pos_d = np.clip(compute_positions(input_d, output_d), 0, max(input_d - 1, 0)).astype(np.float32)
			pos_h = np.clip(compute_positions(input_h, output_h), 0, max(input_h - 1, 0)).astype(np.float32)
			pos_w = np.clip(compute_positions(input_w, output_w), 0, max(input_w - 1, 0)).astype(np.float32)

			d0 = np.floor(pos_d).astype(np.int64)
			d1 = np.clip(d0 + 1, 0, max(input_d - 1, 0))
			ld = (pos_d - d0).astype(np.float32)

			h0 = np.floor(pos_h).astype(np.int64)
			h1 = np.clip(h0 + 1, 0, max(input_h - 1, 0))
			lh = (pos_h - h0).astype(np.float32)

			w0 = np.floor(pos_w).astype(np.int64)
			w1 = np.clip(w0 + 1, 0, max(input_w - 1, 0))
			lw = (pos_w - w0).astype(np.float32)

			ld = ld[:, None, None].astype(dtype, copy=False)
			lh = lh[None, :, None].astype(dtype, copy=False)
			lw = lw[None, None, :].astype(dtype, copy=False)

			d0_idx = d0[:, None, None]
			d1_idx = d1[:, None, None]
			h0_idx = h0[None, :, None]
			h1_idx = h1[None, :, None]
			w0_idx = w0[None, None, :]
			w1_idx = w1[None, None, :]

			one = np.array(1.0, dtype=dtype)
			output_np = np.zeros((batch_size, channels, output_d, output_h, output_w), dtype=dtype)
			for b in range(batch_size):
				for c in range(channels):
					vol = x_np[b, c]
					c000 = vol[d0_idx, h0_idx, w0_idx]
					c001 = vol[d0_idx, h0_idx, w1_idx]
					c010 = vol[d0_idx, h1_idx, w0_idx]
					c011 = vol[d0_idx, h1_idx, w1_idx]
					c100 = vol[d1_idx, h0_idx, w0_idx]
					c101 = vol[d1_idx, h0_idx, w1_idx]
					c110 = vol[d1_idx, h1_idx, w0_idx]
					c111 = vol[d1_idx, h1_idx, w1_idx]

					wd0 = one - ld
					wd1 = ld
					wh0 = one - lh
					wh1 = lh
					ww0 = one - lw
					ww1 = lw

					output_np[b, c] = (
						wd0 * wh0 * ww0 * c000 +
						wd0 * wh0 * ww1 * c001 +
						wd0 * wh1 * ww0 * c010 +
						wd0 * wh1 * ww1 * c011 +
						wd1 * wh0 * ww0 * c100 +
						wd1 * wh0 * ww1 * c101 +
						wd1 * wh1 * ww0 * c110 +
						wd1 * wh1 * ww1 * c111
					)
		else:
			raise ValueError(f"Unsupported mode '{self.mode}' for 3D upsampling")

		output_np = output_np.astype(dtype, copy=False)

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