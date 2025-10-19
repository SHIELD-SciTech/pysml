"""
Intel XPU Backend for PySML using dpnp
FIXED: Robust type detection and conversion
"""

try:
    import dpnp as np
    import dpctl
    AVAILABLE = True
except ImportError:
    AVAILABLE = False
    import numpy as np
    dpctl = None

# Export all dpnp functions
array = np.array
zeros = np.zeros
ones = np.ones
full = np.full
eye = np.eye
arange = np.arange
linspace = np.linspace
empty = np.empty

# Random - with XPU compatibility fixes
class random:
    randn = staticmethod(lambda *shape, **kwargs: np.random.randn(*shape))
    rand = staticmethod(lambda *shape, **kwargs: np.random.rand(*shape))
    randint = staticmethod(np.random.randint)
    uniform = staticmethod(np.random.uniform)
    normal = staticmethod(np.random.normal)
    
    @staticmethod
    def binomial(n, p, size):
        """
        Binomial distribution with XPU fp64 compatibility fix
        
        XPU devices don't support fp64, so we need to use fp32
        """
        if AVAILABLE:
            try:
                # Try with default (may use fp64)
                return np.random.binomial(n, p, size)
            except RuntimeError as e:
                if "fp64" in str(e) or "aspect" in str(e):
                    # XPU doesn't support fp64, use alternative approach
                    # Generate uniform random numbers and threshold them
                    # This is equivalent to binomial for n=1
                    if n == 1:
                        uniform_vals = np.random.uniform(0.0, 1.0, size)
                        result = (uniform_vals < p).astype(np.float32)
                        return result
                    else:
                        # For n > 1, sum n Bernoulli trials
                        result = np.zeros(size, dtype=np.float32)
                        for _ in range(n):
                            uniform_vals = np.random.uniform(0.0, 1.0, size)
                            result = result + (uniform_vals < p).astype(np.float32)
                        return result
                else:
                    raise
        else:
            return np.random.binomial(n, p, size)


# ============================================================================
# ROBUST TYPE CONVERSION HELPERS
# ============================================================================

def _is_numpy_array(arr):
    """Check if array is pure numpy (not dpnp)"""
    if not AVAILABLE:
        return True
    
    # Check module name - most reliable way
    arr_type = type(arr)
    module_name = arr_type.__module__
    
    # Pure numpy arrays have module 'numpy'
    # dpnp arrays have module 'dpnp' or 'dpnp.dpnp_array'
    return module_name == 'numpy' or module_name.startswith('numpy.')


def _ensure_dpnp_array(arr):
    """
    Convert to dpnp array if needed
    
    This is more robust than isinstance checks because:
    - Checks the actual module of the type
    - Handles edge cases where dpnp inherits from numpy
    """
    if not AVAILABLE:
        return arr
    
    # If it's already a dpnp array, return as-is
    if not _is_numpy_array(arr):
        return arr
    
    # It's a numpy array, convert to dpnp
    try:
        return np.array(arr)
    except Exception as e:
        # If conversion fails, log and return original
        # This shouldn't happen but better safe than sorry
        print(f"Warning: Could not convert to dpnp array: {e}")
        return arr


# ============================================================================
# TYPE-SAFE ARITHMETIC OPERATIONS
# ============================================================================

def add(x1, x2):
    """Addition with automatic type conversion"""
    if not AVAILABLE:
        import numpy
        return numpy.add(x1, x2)
    
    x1 = _ensure_dpnp_array(x1)
    x2 = _ensure_dpnp_array(x2)
    return np.add(x1, x2)


def subtract(x1, x2):
    """Subtraction with automatic type conversion"""
    if not AVAILABLE:
        import numpy
        return numpy.subtract(x1, x2)
    
    x1 = _ensure_dpnp_array(x1)
    x2 = _ensure_dpnp_array(x2)
    return np.subtract(x1, x2)


def multiply(x1, x2):
    """Multiplication with automatic type conversion"""
    if not AVAILABLE:
        import numpy
        return numpy.multiply(x1, x2)
    
    x1 = _ensure_dpnp_array(x1)
    x2 = _ensure_dpnp_array(x2)
    return np.multiply(x1, x2)


def divide(x1, x2):
    """Division with automatic type conversion"""
    if not AVAILABLE:
        import numpy
        return numpy.divide(x1, x2)
    
    x1 = _ensure_dpnp_array(x1)
    x2 = _ensure_dpnp_array(x2)
    return np.divide(x1, x2)


def power(x1, x2):
    """Power with automatic type conversion"""
    if not AVAILABLE:
        import numpy
        return numpy.power(x1, x2)
    
    x1 = _ensure_dpnp_array(x1)
    x2 = _ensure_dpnp_array(x2)
    return np.power(x1, x2)


# Other arithmetic that may need conversion
negative = np.negative
positive = np.positive
floor_divide = np.floor_divide
remainder = np.remainder
mod = np.mod
abs = absolute = np.abs
sign = np.sign

# ============================================================================
# TYPE-SAFE LINEAR ALGEBRA OPERATIONS
# ============================================================================

def matmul(x1, x2):
    """Matrix multiplication with automatic type conversion"""
    if not AVAILABLE:
        import numpy
        return numpy.matmul(x1, x2)
    
    x1 = _ensure_dpnp_array(x1)
    x2 = _ensure_dpnp_array(x2)
    return np.matmul(x1, x2)


def dot(x1, x2):
    """Dot product with automatic type conversion"""
    if not AVAILABLE:
        import numpy
        return numpy.dot(x1, x2)
    
    x1 = _ensure_dpnp_array(x1)
    x2 = _ensure_dpnp_array(x2)
    return np.dot(x1, x2)


def outer(x1, x2):
    """Outer product with automatic type conversion"""
    if not AVAILABLE:
        import numpy
        return numpy.outer(x1, x2)
    
    x1 = _ensure_dpnp_array(x1)
    x2 = _ensure_dpnp_array(x2)
    return np.outer(x1, x2)


def inner(x1, x2):
    """Inner product with automatic type conversion"""
    if not AVAILABLE:
        import numpy
        return numpy.inner(x1, x2)
    
    x1 = _ensure_dpnp_array(x1)
    x2 = _ensure_dpnp_array(x2)
    return np.inner(x1, x2)


trace = np.trace
diagonal = np.diagonal
einsum = np.einsum

# ============================================================================
# TYPE-SAFE SHAPE OPERATIONS
# ============================================================================

def transpose(arr, axes=None):
    """Transpose with automatic type conversion"""
    if not AVAILABLE:
        import numpy
        return numpy.transpose(arr, axes=axes)
    
    arr = _ensure_dpnp_array(arr)
    return np.transpose(arr, axes=axes)


def reshape(arr, shape):
    """Reshape with automatic type conversion"""
    if not AVAILABLE:
        import numpy
        return numpy.reshape(arr, shape)
    
    arr = _ensure_dpnp_array(arr)
    return np.reshape(arr, shape)


def swapaxes(arr, axis1, axis2):
    """Swap axes with automatic type conversion"""
    if not AVAILABLE:
        import numpy
        return numpy.swapaxes(arr, axis1, axis2)
    
    arr = _ensure_dpnp_array(arr)
    return np.swapaxes(arr, axis1, axis2)


# Other shape operations
squeeze = np.squeeze
expand_dims = np.expand_dims
concatenate = np.concatenate
stack = np.stack
vstack = np.vstack
hstack = np.hstack
split = np.split
vsplit = np.vsplit if hasattr(np, 'vsplit') else lambda x, n: np.split(x, n, axis=0)
hsplit = np.hsplit if hasattr(np, 'hsplit') else lambda x, n: np.split(x, n, axis=1)
tile = np.tile
repeat = np.repeat
flatten = lambda x: x.flatten()

# Broadcast - critical for memory-efficient gradient operations
broadcast_to = np.broadcast_to if hasattr(np, 'broadcast_to') else None

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

def xpu_trapz(y, x=None, dx=1.0):
    y = list(y) if not isinstance(y, list) else y
    n = len(y)
    
    if n < 2:
        raise ValueError("Need at least 2 points for integration")
    
    if x is None:
        # Uniform spacing
        return dx * (sum(y) - (y[0] + y[-1]) / 2)
    else:
        x = list(x) if not isinstance(x, list) else x
        if len(x) != n:
            raise ValueError("x and y must have same length")
        
        # Non-uniform spacing
        integral = 0.0
        for i in range(n - 1):
            integral += (x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2
        
        return integral

if AVAILABLE:
    trapz = xpu_trapz
else:
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
asnumpy = np.asnumpy if AVAILABLE else np.asarray

# Data types
float32 = np.float32
float64 = np.float64
int32 = np.int32
int64 = np.int64
bool = np.bool_
complex64 = np.complex64
complex128 = np.complex128

# Add astype for type conversions
def astype(arr, dtype):
    """Convert array to specified dtype"""
    if hasattr(arr, 'astype'):
        return arr.astype(dtype)
    else:
        return np.asarray(arr).astype(dtype)

# Device management
def get_device():
    """Get default XPU device"""
    if AVAILABLE and dpctl:
        try:
            device = dpctl.select_default_device()
            return f"xpu:{device}"
        except:
            return "cpu"
    return "cpu"

def get_available_devices(backend="level_zero"):
    """Get list of available XPU devices"""
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
    """Synchronize all XPU operations"""
    if AVAILABLE and dpctl:
        try:
            dpctl.SyclQueue().wait()
        except:
            pass

# Backend name
BACKEND_NAME = "xpu"
DEVICE_TYPE = "xpu"