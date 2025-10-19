"""
CUDA Backend for PySML using CuPy
"""

try:
    import cupy as cp
    AVAILABLE = True
except ImportError:
    AVAILABLE = False
    import numpy as cp

# Export all cupy functions
array = cp.array
zeros = cp.zeros
ones = cp.ones
full = cp.full
eye = cp.eye
arange = cp.arange
linspace = cp.linspace
empty = cp.empty

# Random
class random:
    randn = staticmethod(cp.random.randn)
    rand = staticmethod(cp.random.rand)
    randint = staticmethod(cp.random.randint)
    uniform = staticmethod(cp.random.uniform)
    normal = staticmethod(cp.random.normal)
    binomial = staticmethod(cp.random.binomial)

# Arithmetic
add = cp.add
subtract = cp.subtract
multiply = cp.multiply
divide = cp.divide
power = cp.power
negative = cp.negative
positive = cp.positive
floor_divide = cp.floor_divide
remainder = cp.remainder
mod = cp.mod
abs = absolute = cp.abs
sign = cp.sign

# Linear algebra
matmul = cp.matmul
dot = cp.dot
outer = cp.outer
inner = cp.inner
trace = cp.trace
diagonal = cp.diagonal
einsum = cp.einsum

# Trigonometric
sin = cp.sin
cos = cp.cos
tan = cp.tan
arcsin = cp.arcsin
arccos = cp.arccos
arctan = cp.arctan
arctan2 = cp.arctan2
sinh = cp.sinh
cosh = cp.cosh
tanh = cp.tanh
arcsinh = cp.arcsinh
arccosh = cp.arccosh
arctanh = cp.arctanh

# Exponential/Logarithmic
exp = cp.exp
exp2 = cp.exp2
expm1 = cp.expm1
log = cp.log
log10 = cp.log10
log2 = cp.log2
log1p = cp.log1p
logaddexp = cp.logaddexp
logaddexp2 = cp.logaddexp2

# Rounding
round = cp.round
around = cp.around
rint = cp.rint
fix = cp.fix
floor = cp.floor
ceil = cp.ceil
trunc = cp.trunc

# Sums/Products/Differences
sum = cp.sum
prod = cp.prod
nansum = cp.nansum
nanprod = cp.nanprod
cumsum = cp.cumsum
cumprod = cp.cumprod
nancumsum = cp.nancumsum
nancumprod = cp.nancumprod
diff = cp.diff
ediff1d = cp.ediff1d
gradient = cp.gradient
cross = cp.cross
trapz = cp.trapz

# Statistics
mean = cp.mean
median = cp.median
average = cp.average
var = cp.var
std = cp.std
min = amin = cp.min
max = amax = cp.max
nanmin = cp.nanmin
nanmax = cp.nanmax
nanmean = cp.nanmean
nanmedian = cp.nanmedian
nanvar = cp.nanvar
nanstd = cp.nanstd

# Comparison
maximum = cp.maximum
minimum = cp.minimum
fmax = cp.fmax
fmin = cp.fmin
equal = cp.equal
not_equal = cp.not_equal
less = cp.less
less_equal = cp.less_equal
greater = cp.greater
greater_equal = cp.greater_equal

# Logic
logical_and = cp.logical_and
logical_or = cp.logical_or
logical_not = cp.logical_not
logical_xor = cp.logical_xor
all = cp.all
any = cp.any
isnan = cp.isnan
isinf = cp.isinf
isfinite = cp.isfinite

# Shape manipulation
reshape = cp.reshape
transpose = cp.transpose
swapaxes = cp.swapaxes
squeeze = cp.squeeze
expand_dims = cp.expand_dims
concatenate = cp.concatenate
stack = cp.stack
vstack = cp.vstack
hstack = cp.hstack
split = cp.split
vsplit = cp.vsplit
hsplit = cp.hsplit
tile = cp.tile
repeat = cp.repeat
flatten = lambda x: x.flatten()

# Other operations
clip = cp.clip
where = cp.where
sqrt = cp.sqrt
square = cp.square
cbrt = cp.cbrt
reciprocal = cp.reciprocal
conj = conjugate = cp.conj

# Utilities
zeros_like = cp.zeros_like
ones_like = cp.ones_like
empty_like = cp.empty_like
full_like = cp.full_like
asarray = cp.asarray
copy = cp.copy
asnumpy = cp.asnumpy if AVAILABLE else lambda x: x

# Data types
float32 = cp.float32
float64 = cp.float64
int32 = cp.int32
int64 = cp.int64
bool = cp.bool_
complex64 = cp.complex64
complex128 = cp.complex128

# Device management
def get_device():
    """Get current CUDA device"""
    if AVAILABLE:
        try:
            return f"cuda:{cp.cuda.Device().id}"
        except:
            return "cpu"
    return "cpu"

def get_available_devices():
    """Get list of available CUDA devices"""
    if not AVAILABLE:
        return []
    
    try:
        num_devices = cp.cuda.runtime.getDeviceCount()
        return [f"cuda:{i}" for i in range(num_devices)]
    except:
        return []

def set_device(device_id: int):
    """Set active CUDA device"""
    if AVAILABLE:
        cp.cuda.Device(device_id).use()

def synchronize():
    """Synchronize CUDA operations"""
    if AVAILABLE:
        cp.cuda.Stream.null.synchronize()

# Backend name
BACKEND_NAME = "cuda"
DEVICE_TYPE = "cuda"