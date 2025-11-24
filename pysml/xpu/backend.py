import warnings

try:
        import dpnp as ng
        import dpctl
        AVAILABLE = True
except ImportError:
        warnings.warn(
                "Intel XPU backend (dpnp/dpctl) not found. Falling back to CPU (NumPy). Performance will be significantly degraded.",
                RuntimeWarning
        )
        AVAILABLE = False
        import numpy as ng
        dpctl = None



precision_map = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
precision = precision_map
def convert(data, dtype, device=None, backend="level_zero", shape=None):
        dtype_key = getattr(dtype, "precision", getattr(dtype, "precission", None))
        target_dtype = precision_map[dtype_key]
        if AVAILABLE and device is not None:
                if "xpu" in str(device):
                        device = str(device).split(":")[1]
                return ng.array(data, dtype=target_dtype, device=f"{backend}:gpu:{str(device)}")

        # CPU fallback or when device handling is unsupported
        return ng.array(data, dtype=target_dtype)

add = ng.add
subtract = ng.subtract

multiply = ng.multiply
divide = ng.divide

power = ng.power

negative = ng.negative
positive = ng.positive
floor_divide = ng.floor_divide
remainder = ng.remainder
mod = ng.mod
abs = absolute = ng.abs
sign = ng.sign

matmul = ng.matmul


def tensor_parallel_matmul(input, weight, bias=None, transpose_weight=True):
        """Tensor parallel friendly matmul that fuses bias when possible."""

        right = ng.swapaxes(weight, -1, -2) if transpose_weight else weight
        out = ng.matmul(input, right)
        if bias is not None:
                out = ng.add(out, bias)
        return out
dot = ng.dot
outer = ng.outer
inner = ng.inner

trace = ng.trace
diagonal = ng.diagonal
einsum = ng.einsum

transpose = ng.transpose
reshape = ng.reshape
swapaxes = ng.swapaxes

squeeze = ng.squeeze
expand_dims = ng.expand_dims
concatenate = ng.concatenate
stack = ng.stack
vstack = ng.vstack
hstack = ng.hstack
split = ng.split
vsplit = ng.vsplit if hasattr(ng, 'vsplit') else lambda x, n: ng.split(x, n, axis=0)
hsplit = ng.hsplit if hasattr(ng, 'hsplit') else lambda x, n: ng.split(x, n, axis=1)
tile = ng.tile
repeat = ng.repeat
flatten = lambda x: x.flatten()

broadcast_to = ng.broadcast_to if hasattr(ng, 'broadcast_to') else None

sin = ng.sin
cos = ng.cos
tan = ng.tan
arcsin = ng.arcsin
arccos = ng.arccos
arctan = ng.arctan
arctan2 = ng.arctan2
sinh = ng.sinh
cosh = ng.cosh
tanh = ng.tanh
arcsinh = ng.arcsinh
arccosh = ng.arccosh
arctanh = ng.arctanh

exp = ng.exp
exp2 = ng.exp2
expm1 = ng.expm1
log = ng.log
log10 = ng.log10
log2 = ng.log2
log1p = ng.log1p
logaddexp = ng.logaddexp
logaddexp2 = ng.logaddexp2

round = ng.round
around = ng.around
rint = ng.rint
fix = ng.fix
floor = ng.floor
ceil = ng.ceil
trunc = ng.trunc

sum = ng.sum
prod = ng.prod
nansum = ng.nansum
cumsum = ng.cumsum
cumprod = ng.cumprod
nancumsum = ng.nancumsum
nancumprod = ng.nancumprod
diff = ng.diff
ediff1d = ng.ediff1d
gradient = ng.gradient
cross = ng.cross

def xpu_trapz(y, x=None, dx=1.0):
    y_array = ng.asarray(y)
    if y_array.shape[0] < 2:
        raise ValueError("Need at least 2 points for integration")

    if x is None:
        segment_widths = dx
    else:
        x_array = ng.asarray(x)
        if x_array.shape[0] != y_array.shape[0]:
            raise ValueError("x and y must have same length")
        segment_widths = x_array[1:] - x_array[:-1]

    mid_heights = (y_array[:-1] + y_array[1:]) / 2
    return ng.sum(segment_widths * mid_heights)

if AVAILABLE:
    trapz = xpu_trapz
else:
    trapz = ng.trapz

mean = ng.mean
median = ng.median
average = ng.average
var = ng.var
std = ng.std
min = amin = ng.min
max = amax = ng.max
nanmin = ng.nanmin
nanmax = ng.nanmax
nanmean = ng.nanmean
nanmedian = ng.nanmedian
nanvar = ng.nanvar
nanstd = ng.nanstd

maximum = ng.maximum
minimum = ng.minimum
fmax = ng.fmax
fmin = ng.fmin
equal = ng.equal
not_equal = ng.not_equal
less = ng.less
greater = ng.greater
greater_equal = ng.greater_equal
argmax = ng.argmax
argmin = ng.argmin

logical_and = ng.logical_and
logical_or = ng.logical_or
logical_not = ng.logical_not
logical_xor = ng.logical_xor
all = ng.all
any = ng.any
isnan = ng.isnan
isinf = ng.isinf
isfinite = ng.isfinite

clip = ng.clip
where = ng.where
sqrt = ng.sqrt
square = ng.square
cbrt = ng.cbrt
reciprocal = ng.reciprocal
conj = conjugate = ng.conj

zeros_like = ng.zeros_like
ones_like = ng.ones_like
empty_like = ng.empty_like
full_like = ng.full_like
asarray = ng.asarray
copy = ng.copy
def copyto(dst, src):
    ng.copyto(dst, src)
asnumpy = ng.asnumpy if AVAILABLE else ng.asarray

float32 = ng.float32
float64 = ng.float64
int32 = ng.int32
int64 = ng.int64
bool = ng.bool_
complex64 = ng.complex64
complex128 = ng.complex128


def astype(arr, dtype):
    if hasattr(arr, 'astype'):
        return arr.astype(dtype)
    else:
        return np.asarray(arr).astype(dtype)

def get_device():
    if AVAILABLE and dpctl:
        try:
            device = dpctl.select_default_device()
            return f"xpu:{device}"
        except:
            return "cpu"
    return "cpu"

def get_available_devices(backend="level_zero"):
    if not AVAILABLE or not dpctl:
        return []
    devices = []
    try:
        gpu_devices = dpctl.get_devices(device_type="gpu", backend=backend)
        devices.extend([f"xpu:{i}" for i in range(len(gpu_devices))])
    except Exception as ex:
        print(ex)
    return devices

def synchronize():
    if AVAILABLE and dpctl:
        try:
            dpctl.SyclQueue().wait()
        except:
            pass

# Backend name
BACKEND_NAME = "xpu"
DEVICE_TYPE = "xpu"



def softmax(x, axis=-1, out=None):
    x_max = ng.max(x, axis=axis, keepdims=True)
    
    if out is not None and hasattr(ng, 'subtract'):
        # In-place XPU computation - saves VRAM
        ng.subtract(x, x_max, out=out)
        ng.exp(out, out=out)
        denom = ng.sum(out, axis=axis, keepdims=True)
        ng.divide(out, denom, out=out)
        return out
    else:
        # Standard path with XPU memory pool
        exp_x = ng.exp(x - x_max)
        return exp_x / ng.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(x, axis=-1, out=None):
    x_max = ng.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    log_sum_exp = ng.log(ng.sum(ng.exp(shifted), axis=axis, keepdims=True))
    
    if out is not None and hasattr(ng, 'subtract'):
        ng.subtract(shifted, log_sum_exp, out=out)
        return out
    else:
        return shifted - log_sum_exp


def gelu(x, out=None):
    sqrt_2_over_pi = 0.7978845608028654
    
    if out is not None and hasattr(ng, 'multiply'):
        # Chain of in-place XPU ops - ZERO extra VRAM
        ng.multiply(x, x, out=out)  # x²
        ng.multiply(out, x, out=out)  # x³
        ng.multiply(out, 0.044715, out=out)
        ng.add(out, x, out=out)
        ng.multiply(out, sqrt_2_over_pi, out=out)
        ng.tanh(out, out=out)  # oneMKL accelerated
        ng.add(out, 1.0, out=out)
        ng.multiply(out, x, out=out)
        ng.multiply(out, 0.5, out=out)
        return out
    else:
        return 0.5 * x * (1.0 + ng.tanh(sqrt_2_over_pi * (x + 0.044715 * x * x * x)))


def silu(x, out=None):
    if out is not None and hasattr(ng, 'exp'):
        clipped = ng.clip(x, -20, 20)
        ng.exp(-clipped, out=out)
        ng.add(out, 1.0, out=out)
        ng.reciprocal(out, out=out)  # XPU reciprocal
        ng.multiply(out, x, out=out)
        return out
    else:
        return x / (1.0 + ng.exp(-ng.clip(x, -20, 20)))


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5, out=None):
    ndim = len(x.shape)
    axes = tuple(range(ndim - len(normalized_shape), ndim))

    mean = ng.mean(x, axis=axes, keepdims=True)
    centered = ng.subtract(x, mean)
    var = ng.mean(ng.multiply(centered, centered), axis=axes, keepdims=True)
    
    inv_std = ng.reciprocal(ng.sqrt(var + eps))
    
    if out is not None and hasattr(ng, 'subtract'):
        ng.subtract(x, mean, out=out)
        ng.multiply(out, inv_std, out=out)
        if weight is not None:
            ng.multiply(out, weight, out=out)
        if bias is not None:
            ng.add(out, bias, out=out)
        return out
    else:
        x_norm = (x - mean) * inv_std
        if weight is not None:
            x_norm = x_norm * weight
        if bias is not None:
            x_norm = x_norm + bias
        return x_norm


def rms_norm(x, normalized_shape, weight=None, eps=1e-6, out=None):
    ndim = len(x.shape)
    axes = tuple(range(ndim - len(normalized_shape), ndim))
    
    rms = ng.sqrt(ng.mean(ng.square(x), axis=axes, keepdims=True) + eps)
    inv_rms = ng.reciprocal(rms)
    
    if out is not None and hasattr(ng, 'multiply'):
        ng.multiply(x, inv_rms, out=out)
        if weight is not None:
            ng.multiply(out, weight, out=out)
        return out
    else:
        x_norm = x * inv_rms
        if weight is not None:
            x_norm = x_norm * weight
        return x_norm


def batch_norm(x, running_mean=None, running_var=None, weight=None, bias=None,
               training=True, momentum=0.1, eps=1e-5, out=None):
    if len(x.shape) == 2:
        axes = (0,)
    elif len(x.shape) == 4:
        axes = (0, 2, 3)
    else:
        axes = (0,)
    
    if training:
        mean = ng.mean(x, axis=axes, keepdims=True)
        var = ng.var(x, axis=axes, keepdims=True)
        
        # In-place XPU update - ZERO VRAM copy
        if running_mean is not None:
            running_mean[:] = (1 - momentum) * running_mean + momentum * ng.squeeze(mean)
        if running_var is not None:
            running_var[:] = (1 - momentum) * running_var + momentum * ng.squeeze(var)
    else:
        shape_4d = (1, -1, 1, 1) if len(x.shape) == 4 else (1, -1)
        mean = running_mean.reshape(shape_4d)
        var = running_var.reshape(shape_4d)
    
    inv_std = ng.reciprocal(ng.sqrt(var + eps))
    
    if out is not None and hasattr(ng, 'subtract'):
        ng.subtract(x, mean, out=out)
        ng.multiply(out, inv_std, out=out)
        if weight is not None:
            shape_4d = (1, -1, 1, 1) if len(x.shape) == 4 else (1, -1)
            ng.multiply(out, weight.reshape(shape_4d), out=out)
        if bias is not None:
            shape_4d = (1, -1, 1, 1) if len(x.shape) == 4 else (1, -1)
            ng.add(out, bias.reshape(shape_4d), out=out)
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
    
    # ZERO-COPY XPU view reshaping
    if len(x.shape) == 4:
        x_grouped = x.reshape(batch_size, num_groups, channels_per_group, x.shape[2], x.shape[3])
        axes = (2, 3, 4)
    else:
        x_grouped = x.reshape(batch_size, num_groups, channels_per_group)
        axes = (2,)
    
    mean = ng.mean(x_grouped, axis=axes, keepdims=True)
    var = ng.var(x_grouped, axis=axes, keepdims=True)
    inv_std = ng.reciprocal(ng.sqrt(var + eps))
    
    x_norm = (x_grouped - mean) * inv_std
    x_norm = x_norm.reshape(x.shape)  # ZERO-COPY XPU view
    
    if weight is not None:
        shape_4d = (1, -1, 1, 1) if len(x.shape) == 4 else (1, -1)
        x_norm = x_norm * weight.reshape(shape_4d)
    if bias is not None:
        shape_4d = (1, -1, 1, 1) if len(x.shape) == 4 else (1, -1)
        x_norm = x_norm + bias.reshape(shape_4d)
    
    if out is not None:
        ng.copyto(out, x_norm)
        return out
    return x_norm


def dropout(x, p=0.5, training=True):
        if not training or p == 0:
                return None, x

        keep_prob = 1.0 - p

        # XPU random generation with fallback
        if AVAILABLE and hasattr(ng.random, 'rand'):
                mask = rand(x.shape, device=getattr(x, "device", None)) > p
                output = ng.where(mask, x * (1.0 / keep_prob), 0)
                return mask.astype(x.dtype), output
        else:
                # Fallback to NumPy (on CPU then transfer)
                import numpy as np
                mask_np = rand(x.shape) > p
                mask = ng.asarray(mask_np)
                output = ng.where(mask, x * (1.0 / keep_prob), 0)
                return mask.astype(x.dtype), output


def embedding_lookup(table, indices, padding_idx=None):
    result = table[indices]  # XPU indexing
    
    if padding_idx is not None:
        mask = indices == padding_idx
        result[mask] = 0  # In-place XPU masking
    
    return result


def unsqueeze(x, dim):
    return ng.expand_dims(x, axis=dim)


def gather(x, dim, index):
    if hasattr(ng, 'take_along_axis'):
        return ng.take_along_axis(x, index, axis=dim)
    else:
        # Fallback to NumPy
        import numpy as np
        x_np = ng.asnumpy(x) if AVAILABLE else x
        idx_np = ng.asnumpy(index) if AVAILABLE else index
        result = np.take_along_axis(x_np, idx_np, axis=dim)
        return ng.asarray(result) if AVAILABLE else result


def scatter_add(x, dim, index, src):
        if dim < 0:
                dim += x.ndim

        if AVAILABLE:
                lib = ng
        else:
                import numpy as _np
                lib = _np

        index_arr = lib.asarray(index)
        src_arr = lib.asarray(src)

        def _broadcast(value, shape):
                if hasattr(lib, 'broadcast_to'):
                        return lib.broadcast_to(value, shape)
                import numpy as np
                if AVAILABLE:
                        value_np = ng.asnumpy(value)
                        broadcasted = np.broadcast_to(value_np, shape)
                        return ng.asarray(broadcasted)
                return np.broadcast_to(value, shape)

        def _add_at(target, idx, values):
                if hasattr(lib.add, 'at'):
                        lib.add.at(target, idx, values)
                else:
                        import numpy as np
                        if AVAILABLE:
                                target_np = ng.asnumpy(target)
                                idx_np = ng.asnumpy(idx)
                                val_np = ng.asnumpy(values)
                        else:
                                target_np = target
                                idx_np = idx
                                val_np = values
                        np.add.at(target_np, idx_np, val_np)
                        if AVAILABLE:
                                target[...] = ng.asarray(target_np)
                        else:
                                target[...] = target_np

        if x.ndim == 1 or src_arr.ndim == 1:
                _add_at(x, index_arr, src_arr)
                return x

        if src_arr.ndim == x.ndim:
                if hasattr(lib, 'indices'):
                        grid = lib.indices(src_arr.shape, sparse=False)
                else:
                        import numpy as np
                        grid = np.indices(src_arr.shape, sparse=False)
                        grid = lib.asarray(grid)
                idx = []
                for axis in range(x.ndim):
                        if axis == dim:
                                idx.append(index_arr)
                        else:
                                idx.append(grid[axis])
                _add_at(x, tuple(idx), src_arr)
                return x

        if dim == 0 and x.ndim == 2 and src_arr.ndim == index_arr.ndim + 1:
                rows = lib.reshape(index_arr, (-1, 1))
                if hasattr(lib, 'arange'):
                        cols = lib.arange(x.shape[1]).reshape(1, -1)
                else:
                        import numpy as np
                        cols = np.arange(x.shape[1]).reshape(1, -1)
                        cols = lib.asarray(cols)
                rows = _broadcast(rows, (rows.shape[0], cols.shape[1]))
                cols = _broadcast(cols, rows.shape)
                src_flat = lib.reshape(src_arr, rows.shape)
                _add_at(x, (rows, cols), src_flat)
                return x

        raise NotImplementedError("scatter_add configuration not supported on XPU backend")


def masked_fill(x, mask, value):
    result = x.copy()
    result[mask] = value
    return result


def less_equal(x, y):
    if hasattr(ng, 'less_equal'):
        return ng.less_equal(x, y)
    else:
        return x <= y


def pad(x, pad_width, mode='constant', constant_values=0):
    if hasattr(ng, 'pad'):
        return ng.pad(x, pad_width, mode=mode, constant_values=constant_values)
    else:
        # Fallback to NumPy
        import numpy as np
        x_np = ng.asnumpy(x) if AVAILABLE else x
        result = np.pad(x_np, pad_width, mode=mode, constant_values=constant_values)
        return ng.asarray(result) if AVAILABLE else result


def zeros(shape, dtype=ng.float32):
    return ng.zeros(shape, dtype=dtype)


def ones(shape, dtype=ng.float32):
    return ng.ones(shape, dtype=dtype)


