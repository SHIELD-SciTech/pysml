import numpy as np
import re
# from pysml.nn.autograd import Function


SYCL_QUEUE = None  # Global variable to hold the current SYCL queue for XPU operations
STANDARD_DTYPE = np.float32

class Tensor:
    def __init__(self, data, dtype=None, requires_grad=False, _ctx=None):
        np = self._get_backend_module()
        if isinstance(data, Tensor):
            data = data.data
        if dtype is None:
            dtype = STANDARD_DTYPE
        self.data = np.array(data, dtype=dtype) if dtype else np.array(data)
        self.grad = None
        self.requires_grad = requires_grad
        self._ctx = _ctx
        self.shape = self.data.shape
        if dtype == None:
            self.dtype = self.data.dtype
        else:
            self.dtype = dtype
            self.data = self.data.astype(dtype)

    def _get_backend_module(self):
        if TensorType.backend == 'cpu':
            import numpy as np
            return np
        elif TensorType.backend == 'cuda':
            import cupy as np
            return np
        elif TensorType.backend == 'xpu':
            import dpnp as np
            return np
        else:
            raise ValueError(f"Unknown backend '{TensorType.backend}'")
        
    def requires_grad_(self, state =True):
        self.requires_grad = state
        return self

    def to(self, device: str):
        global SYCL_QUEUE
        TensorType.set_device(device)

        backend = TensorType.backend
        if backend == 'cpu':
            import numpy as np_target
            if hasattr(self.data, 'asnumpy'):
                data_target = np_target.array(self.data.asnumpy())
            elif hasattr(self.data, 'get'):
                data_target = np_target.array(self.data.get())
            else:
                data_target = np_target.array(self.data)
            data_target = data_target.astype(self.dtype)
        elif backend == 'cuda':
            import cupy as np_target
            data_target = np_target.array(self.data)
        elif backend == 'xpu':
            import dpnp as np_target
            import dpctl
            dev = dpctl.get_devices(backend="level_zero", device_type="gpu")[int(device.split(':')[1])]
            ctx = dpctl.SyclContext(dev)
            queue = None
            if SYCL_QUEUE is None:
                SYCL_QUEUE = {}
            if not int(device.split(':')[1]) in SYCL_QUEUE:
                queue = dpctl.SyclQueue(dev)
                SYCL_QUEUE[int(device.split(':')[1])] = queue
            else:
                queue = SYCL_QUEUE[int(device.split(':')[1])]
            data_target = np_target.array(self.data.asnumpy(), sycl_queue=queue)
        else:
            raise ValueError(f"Unknown backend '{backend}'")

        return Tensor(data_target)
    
    def item(self):
        if self.data.size != 1:
            raise ValueError("Tensor must be a scalar (shape must be ()) to call item()")
        return self.data.item() if hasattr(self.data, 'item') else self.data[0]
    
    def zeros_like(self):
        # Returns a new Tensor with zeros, matching the shape, dtype, and backend of the current tensor.
        np_backend = self._get_backend_module()
        zero_data = np_backend.zeros_like(self.data)
        return Tensor(zero_data, dtype=self.dtype)
    
    def backward(self):
        if self._ctx is None:
            return

        if self.grad is None:
            # The initial gradient for the start of the chain is all ones.
            self.grad = Tensor(np.ones(self.shape, dtype=self.dtype))

        # Build a topological order of the graph
        topo_sorted = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                if v._ctx:
                    for parent in v._ctx.parents:
                        build_topo(parent)
                    topo_sorted.append(v)
        
        build_topo(self)

        # Go backwards through the topological sort and apply gradients
        for v in reversed(topo_sorted):
            if v._ctx:
                grads = v._ctx.backward(v.grad)
                if not isinstance(grads, tuple):
                    grads = (grads,)
                
                for parent, grad in zip(v._ctx.parents, grads):
                    if grad is not None and parent.requires_grad:
                        if parent.grad is None:
                            # Use backend-aware zeros_like through parent's method
                            parent.grad = parent.zeros_like()
                        
                        grad_data = grad.data
                        if grad_data.shape != parent.data.shape:
                            # Get backend-aware numpy module
                            np_backend = parent._get_backend_module()
                            
                            # Reduce dimensions that were broadcasted
                            ndims_diff = len(grad_data.shape) - len(parent.data.shape)
                            for _ in range(ndims_diff):
                                grad_data = np_backend.sum(grad_data, axis=0)
                            
                            # Sum over dimensions that were size 1 in parent
                            for i in range(len(parent.data.shape)):
                                if parent.data.shape[i] == 1 and grad_data.shape[i] > 1:
                                    grad_data = np_backend.sum(grad_data, axis=i, keepdims=True)
                        
                        parent.grad.data += grad_data
    
    def _binary_op(self, other, op_class):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return op_class.apply(op_class, self, other)
    
    def _unary_op(self, op_class):
        return op_class.apply(op_class, self)
        
    def mean(self, axis=None, keepdims=False):
        from pysml.nn.autograd import Mean
        # Correctly pass arguments to the autograd function
        return Mean.apply(Mean, self, axis, keepdims)

    def transpose(self, dim0: int, dim1: int):
        from pysml.nn.autograd import Transpose
        return Transpose.apply(Transpose, self, dim0, dim1)
    
    def view(self, *shape):
        from pysml.nn.autograd import View
        return View.apply(View, self, *shape)
    
    @property
    def T(self):
        return self.transpose(0, 1)

    def __getitem__(self, key):
        from pysml.nn.autograd import Slice
        return Slice.apply(Slice, self, key)

    def __matmul__(self, other): 
        from pysml.nn.autograd import MatMul
        return self._binary_op(other, MatMul)
    
    def __truediv__(self, other):
        from pysml.nn.autograd import Div
        return self._binary_op(other, Div)
        
    def __pow__(self, other):
        from pysml.nn.autograd import Pow
        return self._binary_op(other, Pow)

    def softmax(self, axis: int = -1):
        # Softmax is a special case, often combined with loss for stability
        np_backend = self._get_backend_module()
        max_val = np_backend.max(self.data, axis=axis, keepdims=True)
        exp_data = np_backend.exp(self.data - max_val)
        sum_exp = np_backend.sum(exp_data, axis=axis, keepdims=True)
        return Tensor(exp_data / sum_exp, dtype=self.dtype)

    def masked_fill(self, mask_tensor, value):
        from pysml.nn.autograd import MaskedFill
        return MaskedFill.apply(MaskedFill, self, mask_tensor, value)

    def __mul__(self, other):
        from pysml.nn.autograd import Mul
        return self._binary_op(other, Mul)
        
    def __add__(self, other):
        from pysml.nn.autograd import Add
        return self._binary_op(other, Add)
        
    def __sub__(self, other):
        from pysml.nn.autograd import Sub
        return self._binary_op(other, Sub)
        
    def __neg__(self):
        from pysml.nn.autograd import Neg
        return self._unary_op(Neg)

    def __radd__(self, other): return self.__add__(other)
    def __rsub__(self, other): return self.__sub__(other) * -1
    def __rmul__(self, other): return self.__mul__(other)
    def __rtruediv__(self, other):
        from pysml.nn.autograd import Div
        return Tensor(other) / self

    def __repr__(self):
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, backend={TensorType.backend}, grad_fn={self._ctx.__class__.__name__ if self._ctx else 'None'})"


class TensorType:
    backend = 'cpu'
    device = 'cpu'

    @classmethod
    def set_backend(cls, backend):
        if backend not in ['cpu', 'cuda', 'xpu']:
            raise ValueError("Unsupported backend. Choose from 'cpu', 'cuda', or 'xpu'.")
        cls.backend = backend

    @classmethod
    def set_device(cls, device):
        if not isinstance(device, str):
            raise ValueError("Device must be a string.")
        if not re.match(r"^(cpu|cuda:\d+|xpu:\d+)$", device):
            raise ValueError("Device must be 'cpu', 'cuda:<id>', or 'xpu:<id>'.")
        cls.device = device
        if 'cuda' in device:
            cls.backend = 'cuda'
        elif 'xpu' in device:
            cls.backend = 'xpu'
        else:
            cls.backend = 'cpu'
    
    @classmethod
    def get_backend(cls):
        return cls.backend

    @classmethod
    def get_device(cls):
        return cls.device

class dtype:
    float16 = np.float16
    float32 = np.float32
    float64 = np.float64
    int32 = np.int32
    int64 = np.int64
    uint8 = np.uint8
    bool = np.bool_