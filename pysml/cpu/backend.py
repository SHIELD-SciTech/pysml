"""
CPU Backend for PySML using NumPy
"""

import numpy as np

# Export all numpy functions we need
array = np.array
zeros = np.zeros
ones = np.ones
full = np.full
eye = np.eye
arange = np.arange
linspace = np.linspace
empty = np.empty

# Random
class random:
    randn = staticmethod(np.random.randn)
    rand = staticmethod(np.random.rand)
    randint = staticmethod(np.random.randint)
    uniform = staticmethod(np.random.uniform)
    normal = staticmethod(np.random.normal)
    binomial = staticmethod(np.random.binomial)

# Arithmetic
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

# Linear algebra
matmul = np.matmul
dot = np.dot
outer = np.outer
inner = np.inner
trace = np.trace
diagonal = np.diagonal
einsum = np.einsum

# Trigonometric
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

# Exponential/Logarithmic
exp = np.exp
exp2 = np.exp2
expm1 = np.expm1
log = np.log
log10 = np.log10
log2 = np.log2
log1p = np.log1p
logaddexp = np.logaddexp
logaddexp2 = np.logaddexp2

# Rounding
round = np.round
around = np.around
rint = np.rint
fix = np.fix
floor = np.floor
ceil = np.ceil
trunc = np.trunc

# Sums/Products/Differences
sum = np.sum
prod = np.prod
nansum = np.nansum
nanprod = np.nanprod
cumsum = np.cumsum
cumprod = np.cumprod
nancumsum = np.nancumsum
nancumprod = np.nancumprod
diff = np.diff
ediff1d = np.ediff1d
gradient = np.gradient
cross = np.cross
trapz = np.trapz

# Statistics
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

# Comparison
maximum = np.maximum
minimum = np.minimum
fmax = np.fmax
fmin = np.fmin
equal = np.equal
not_equal = np.not_equal
less = np.less
less_equal = np.less_equal
greater = np.greater
greater_equal = np.greater_equal

# Logic
logical_and = np.logical_and
logical_or = np.logical_or
logical_not = np.logical_not
logical_xor = np.logical_xor
all = np.all
any = np.any
isnan = np.isnan
isinf = np.isinf
isfinite = np.isfinite

# Shape manipulation
reshape = np.reshape
transpose = np.transpose
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

# Other operations
clip = np.clip
where = np.where
sqrt = np.sqrt
square = np.square
cbrt = np.cbrt
reciprocal = np.reciprocal
conj = conjugate = np.conj

# Utilities
zeros_like = np.zeros_like
ones_like = np.ones_like
empty_like = np.empty_like
full_like = np.full_like
asarray = np.asarray
copy = np.copy
asnumpy = np.asarray  # For compatibility

# Data types
float32 = np.float32
float64 = np.float64
int32 = np.int32
int64 = np.int64
bool = np.bool_
complex64 = np.complex64
complex128 = np.complex128

# Device management (dummy for CPU)
def get_device():
    """Get CPU device"""
    return "cpu"

def synchronize():
    """Synchronize operations (no-op for CPU)"""
    pass

# Backend name
BACKEND_NAME = "cpu"
DEVICE_TYPE = "cpu"