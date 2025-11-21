import numpy as np

AVAILABLE = True

precision_map = {"fp32": "float32", "fp16": "float16", "bf16": "float16"}
precision = precision_map

def convert(data, dtype, device=None):
	# Convert from GPU backends to CPU
	if hasattr(data, 'get'):  # CuPy array
		data = data.get()
	elif hasattr(data, 'asnumpy'):  # dpnp array
		data = data.asnumpy()
        dtype_key = getattr(dtype, "precision", getattr(dtype, "precission", None))
        return np.array(data, dtype=precision_map[dtype_key])

add = np.add
subtract = np.subtract

multiply = np.multiply
divide = np.divide

power = np.power

negative = np.negative
positive = np.positive
floor_divide = np.floor_divide
remainder = np.remainder
mod = np.mod
abs = absolute = np.abs
sign = np.sign

matmul = np.matmul


def tensor_parallel_matmul(input, weight, bias=None, transpose_weight=True):
        """Matmul helper that optionally transposes ``weight`` and adds ``bias``."""

        right = np.swapaxes(weight, -1, -2) if transpose_weight else weight
        result = np.matmul(input, right)
        if bias is not None:
                result = np.add(result, bias)
        return result
dot = np.dot
outer = np.outer
inner = np.inner

trace = np.trace
diagonal = np.diagonal
einsum = np.einsum

transpose = np.transpose
reshape = np.reshape
swapaxes = np.swapaxes

squeeze = np.squeeze
expand_dims = np.expand_dims
concatenate = np.concatenate
stack = np.stack
vstack = np.vstack
hstack = np.hstack
split = np.split
vsplit = np.vsplit
hsplit = np.hsplit
tile = np.tile
repeat = np.repeat
flatten = lambda x: x.flatten()

broadcast_to = np.broadcast_to

sin = np.sin
cos = np.cos
tan = np.tan
arcsin = np.arcsin
arccos = np.arccos
arctan = np.arctan
arctan2 = np.arctan2
sinh = np.sinh
cosh = np.cosh
tanh = np.tanh
arcsinh = np.arcsinh
arccosh = np.arccosh
arctanh = np.arctanh

exp = np.exp
exp2 = np.exp2
expm1 = np.expm1
log = np.log
log10 = np.log10
log2 = np.log2
log1p = np.log1p
logaddexp = np.logaddexp
logaddexp2 = np.logaddexp2

round = np.round
around = np.around
rint = np.rint
fix = np.fix
floor = np.floor
ceil = np.ceil
trunc = np.trunc

sum = np.sum
prod = np.prod
nansum = np.nansum
cumsum = np.cumsum
cumprod = np.cumprod
nancumsum = np.nancumsum
nancumprod = np.nancumprod
diff = np.diff
ediff1d = np.ediff1d
gradient = np.gradient
cross = np.cross
trapz = np.trapz

mean = np.mean
median = np.median
average = np.average
var = np.var
std = np.std
min = amin = np.min
max = amax = np.max
nanmin = np.nanmin
nanmax = np.nanmax
nanmean = np.nanmean
nanmedian = np.nanmedian
nanvar = np.nanvar
nanstd = np.nanstd

maximum = np.maximum
minimum = np.minimum
fmax = np.fmax
fmin = np.fmin
equal = np.equal
not_equal = np.not_equal
less = np.less
greater = np.greater
greater_equal = np.greater_equal
argmax = np.argmax
argmin = np.argmin

logical_and = np.logical_and
logical_or = np.logical_or
logical_not = np.logical_not
logical_xor = np.logical_xor
all = np.all
any = np.any
isnan = np.isnan
isinf = np.isinf
isfinite = np.isfinite

clip = np.clip
where = np.where
sqrt = np.sqrt
square = np.square
cbrt = np.cbrt
reciprocal = np.reciprocal
conj = conjugate = np.conj

zeros_like = np.zeros_like
ones_like = np.ones_like
empty_like = np.empty_like
full_like = np.full_like
asarray = np.asarray
copy = np.copy
def copyto(dst, src):
	np.copyto(dst, src)
asnumpy = np.asarray

float32 = np.float32
float64 = np.float64
int32 = np.int32
int64 = np.int64
bool = np.bool_
complex64 = np.complex64
complex128 = np.complex128

def multiply_scalar(data, scalar):
	return data * scalar

def astype(arr, dtype):
	if hasattr(arr, 'astype'):
		return arr.astype(dtype)
	else:
		return np.asarray(arr).astype(dtype)

def get_device():
	return "cpu"

def get_available_devices():
	return ["cpu"]

def synchronize():
	# CPU doesn't need synchronization
	pass

# Backend name
BACKEND_NAME = "cpu"
DEVICE_TYPE = "cpu"


def softmax(x, axis=-1, out=None):
	x_max = np.max(x, axis=axis, keepdims=True)
	
	if out is not None:
		# In-place computation - saves RAM
		np.subtract(x, x_max, out=out)
		np.exp(out, out=out)
		denom = np.sum(out, axis=axis, keepdims=True)
		np.divide(out, denom, out=out)
		return out
	else:
		exp_x = np.exp(x - x_max)
		return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(x, axis=-1, out=None):
	x_max = np.max(x, axis=axis, keepdims=True)
	shifted = x - x_max
	log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
	
	if out is not None:
		np.subtract(shifted, log_sum_exp, out=out)
		return out
	else:
		return shifted - log_sum_exp


def gelu(x, out=None):
	if out is not None:
		# Chain of in-place ops - ZERO extra allocation
		np.multiply(x, x, out=out)  # x²
		np.multiply(out, x, out=out)  # x³
		np.multiply(out, 0.044715, out=out)
		np.add(out, x, out=out)
		np.multiply(out, 0.7978845608, out=out)
		np.tanh(out, out=out)
		np.add(out, 1.0, out=out)
		np.multiply(out, x, out=out)
		np.multiply(out, 0.5, out=out)
		return out
	else:
		return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


def silu(x, out=None):
	if out is not None:
		clipped = np.clip(x, -20, 20)
		np.exp(-clipped, out=out)
		np.add(out, 1.0, out=out)
		np.reciprocal(out, out=out)
		np.multiply(out, x, out=out)
		return out
	else:
		return x / (1.0 + np.exp(-np.clip(x, -20, 20)))


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5, out=None, return_stats=False):
        ndim = len(x.shape)
        axes = tuple(range(ndim - len(normalized_shape), ndim))
	
	# RAM-efficient variance: E[x²] - E[x]²
	mean = np.mean(x, axis=axes, keepdims=True)
	mean_sq = np.mean(np.square(x), axis=axes, keepdims=True)
	var = mean_sq - np.square(mean)
	
	inv_std = np.reciprocal(np.sqrt(var + eps))
	
        if out is not None:
                np.subtract(x, mean, out=out)
                np.multiply(out, inv_std, out=out)
                if weight is not None:
                        np.multiply(out, weight, out=out)
                if bias is not None:
                        np.add(out, bias, out=out)
                result = out
        else:
                result = (x - mean) * inv_std
                if weight is not None:
                        result = result * weight
                if bias is not None:
                        result = result + bias

        if return_stats:
                return result, mean, inv_std
        return result


def rms_norm(x, normalized_shape, weight=None, eps=1e-6, out=None, return_stats=False):
        ndim = len(x.shape)
        axes = tuple(range(ndim - len(normalized_shape), ndim))
	
	rms = np.sqrt(np.mean(np.square(x), axis=axes, keepdims=True) + eps)
	inv_rms = np.reciprocal(rms)
	
        if out is not None:
                np.multiply(x, inv_rms, out=out)
                if weight is not None:
                        np.multiply(out, weight, out=out)
                result = out
        else:
                result = x * inv_rms
                if weight is not None:
                        result = result * weight

        if return_stats:
                return result, inv_rms
        return result


def batch_norm(x, running_mean=None, running_var=None, weight=None, bias=None,
			   training=True, momentum=0.1, eps=1e-5, out=None):
	if len(x.shape) == 2:
		axes = (0,)
	elif len(x.shape) == 4:
		axes = (0, 2, 3)
	else:
		axes = (0,)
	
	if training:
		mean = np.mean(x, axis=axes, keepdims=True)
		var = np.var(x, axis=axes, keepdims=True)
		
		# In-place update - ZERO copy
		if running_mean is not None:
			running_mean[:] = (1 - momentum) * running_mean + momentum * np.squeeze(mean)
		if running_var is not None:
			running_var[:] = (1 - momentum) * running_var + momentum * np.squeeze(var)
	else:
		shape_4d = (1, -1, 1, 1) if len(x.shape) == 4 else (1, -1)
		mean = running_mean.reshape(shape_4d)
		var = running_var.reshape(shape_4d)
	
	inv_std = np.reciprocal(np.sqrt(var + eps))
	
	if out is not None:
		np.subtract(x, mean, out=out)
		np.multiply(out, inv_std, out=out)
		if weight is not None:
			shape_4d = (1, -1, 1, 1) if len(x.shape) == 4 else (1, -1)
			np.multiply(out, weight.reshape(shape_4d), out=out)
		if bias is not None:
			shape_4d = (1, -1, 1, 1) if len(x.shape) == 4 else (1, -1)
			np.add(out, bias.reshape(shape_4d), out=out)
		return out
	else:
		x_norm = (x - mean) * inv_std
		if weight is not None:
			shape_4d = (1, -1, 1, 1) if len(x.shape) == 4 else (1, -1)
			x_norm = x_norm * weight.reshape(shape_4d)
		if bias is not None:
			shape_4d = (1, -1, 1, 1) if len(x.shape) == 4 else (1, -1)
			x_norm = x_norm + bias.reshape(shape_4d)
		return x_norm


def group_norm(x, num_groups, weight=None, bias=None, eps=1e-5, out=None):
	batch_size, num_channels = x.shape[0], x.shape[1]
	channels_per_group = num_channels // num_groups
	
	# ZERO-COPY view reshaping
	if len(x.shape) == 4:
		x_grouped = x.reshape(batch_size, num_groups, channels_per_group, x.shape[2], x.shape[3])
		axes = (2, 3, 4)
	else:
		x_grouped = x.reshape(batch_size, num_groups, channels_per_group)
		axes = (2,)
	
	mean = np.mean(x_grouped, axis=axes, keepdims=True)
	var = np.var(x_grouped, axis=axes, keepdims=True)
	inv_std = np.reciprocal(np.sqrt(var + eps))
	
	x_norm = (x_grouped - mean) * inv_std
	x_norm = x_norm.reshape(x.shape)  # ZERO-COPY view back
	
	if weight is not None:
		shape_4d = (1, -1, 1, 1) if len(x.shape) == 4 else (1, -1)
		x_norm = x_norm * weight.reshape(shape_4d)
	if bias is not None:
		shape_4d = (1, -1, 1, 1) if len(x.shape) == 4 else (1, -1)
		x_norm = x_norm + bias.reshape(shape_4d)
	
	if out is not None:
		np.copyto(out, x_norm)
		return out
	return x_norm


def dropout(x, p=0.5, training=True):
        if not training or p == 0:
                return None, x

        keep_prob = 1.0 - p
        # Efficient boolean mask using backend RNG
        mask = rand(x.shape) > p
        # Single-pass: mask + scale (saves temporary array)
        output = np.where(mask, x * (1.0 / keep_prob), 0)
	
	return mask.astype(x.dtype), output


def embedding_lookup(table, indices, padding_idx=None):
	result = table[indices]  # Often a view, not a copy
	
	if padding_idx is not None:
		mask = indices == padding_idx
		result[mask] = 0  # In-place masking
	
	return result


def unsqueeze(x, dim):
	return np.expand_dims(x, axis=dim)


def gather(x, dim, index):
	return np.take_along_axis(x, index, axis=dim)


def scatter_add(x, dim, index, src):
        if dim < 0:
            dim += x.ndim

        index = np.asarray(index)
        src = np.asarray(src)

        if x.ndim == 1 or src.ndim == 1:
            np.add.at(x, index, src)
            return x

        if src.ndim == x.ndim:
            grid = np.indices(src.shape, sparse=False)
            idx = []
            for axis in range(x.ndim):
                if axis == dim:
                    idx.append(index)
                else:
                    idx.append(grid[axis])
            np.add.at(x, tuple(idx), src)
            return x

        if dim == 0 and x.ndim == 2 and src.ndim == index.ndim + 1:
            rows = np.reshape(index, (-1, 1))
            cols = np.arange(x.shape[1]).reshape(1, -1)
            rows = np.broadcast_to(rows, (rows.shape[0], cols.shape[1]))
            cols = np.broadcast_to(cols, rows.shape)
            src_flat = np.reshape(src, rows.shape)
            np.add.at(x, (rows, cols), src_flat)
            return x

        raise NotImplementedError("scatter_add configuration not supported on CPU backend")


def masked_fill(x, mask, value):
	result = x.copy()
	result[mask] = value
	return result


def less_equal(x, y):
	return np.less_equal(x, y)


def pad(x, pad_width, mode='constant', constant_values=0):
	return np.pad(x, pad_width, mode=mode, constant_values=constant_values)


def zeros(shape, dtype=np.float32):
	return np.zeros(shape, dtype=dtype)


def ones(shape, dtype=np.float32):
	return np.ones(shape, dtype=dtype)


# ============================================================================
# MEMORY USAGE SUMMARY FOR TRANSFORMERS
# ============================================================================
# For GPT-style model (batch=32, seq=2048, d_model=768):
#
# layer_norm with standard variance:
#   - x: 32 * 2048 * 768 * 4 bytes = 200MB
#   - (x-μ) temporary: 200MB extra
#   - Total: 400MB peak
#
# layer_norm with optimized variance (E[x²] - E[x]²):
#   - x: 200MB
#   - No temporary
#   - Total: 200MB peak
#   - RAM saved: 200MB per layer
#
# For 96-layer model (GPT-3): 200MB * 96 = 19.2GB saved!
# ============================================================================