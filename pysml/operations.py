from .tensor import Tensor, TensorType
import numpy as backend_cpu
np = backend_cpu

def update_backend(backend = ''):
    global np
    if backend != '':
        if backend not in ['cpu', 'cuda', 'xpu']:
            raise ValueError("Unsupported backend. Choose from 'cpu', 'cuda', or 'xpu'.")
        TensorType.set_backend(backend)
    
    if TensorType.backend == 'cpu':
        np = backend_cpu
    if TensorType.backend == 'cuda':
        from pysml.cuda.backend import backend as backend_cuda
        np = backend_cuda
    if TensorType.backend == 'xpu':
        from pysml.xpu.backend import backend as backend_xpu
        np = backend_xpu
    
    return TensorType

def multiply(t1, t2):
    if isinstance(t1, Tensor) and isinstance(t2, Tensor):
        return Tensor(np.multiply(t1.data, t2.data), dtype=t1.dtype if t1.dtype == t2.dtype else np.result_type(t1.dtype, t2.dtype))
    else:
        raise ValueError("Both inputs must be Tensor instances.")
    
def mm(t1, t2):
    return matmul(t1, t2)

def add(t1, t2):
    if isinstance(t1, Tensor) and isinstance(t2, Tensor):
        return Tensor(np.add(t1.data, t2.data), dtype=t1.dtype if t1.dtype == t2.dtype else np.result_type(t1.dtype, t2.dtype))
    else:
        raise ValueError("Both inputs must be Tensor instances.")

def matmul(t1, t2):
    if isinstance(t1, Tensor) == False:
        t1 = Tensor(t1)
    if isinstance(t2, Tensor) == False:
        t2 = Tensor(t2)
    if isinstance(t1, Tensor) and isinstance(t2, Tensor):
        return Tensor(np.matmul(t1.data, t2.data), dtype=t1.dtype if t1.dtype == t2.dtype else np.result_type(t1.dtype, t2.dtype))
    else:
        raise ValueError("Both inputs must be Tensor instances.")

def relu(t):
    if isinstance(t, Tensor):
        return Tensor(np.maximum(0, t.data), dtype=t.dtype)
    else:
        raise ValueError("Input must be a Tensor instance.")

def sigmoid(t):
    if isinstance(t, Tensor):
        return Tensor(1 / (1 + np.exp(-t.data)), dtype=t.dtype)
    else:
        raise ValueError("Input must be a Tensor instance.")

def tanh(t):
    if isinstance(t, Tensor):
        return Tensor(np.tanh(t.data), dtype=t.dtype)
    else:
        raise ValueError("Input must be a Tensor instance.")

def softmax(t, axis=-1):
    if isinstance(t, Tensor):
        e_x = np.exp(t.data - np.max(t.data, axis=axis, keepdims=True))
        return Tensor(e_x / e_x.sum(axis=axis, keepdims=True), dtype=t)
    else:
        raise ValueError("Input must be a Tensor instance.")

def mean(t, axis=None):
    if isinstance(t, Tensor):
        return Tensor(np.mean(t.data, axis=axis), dtype=t.dtype)
    else:
        raise ValueError("Input must be a Tensor instance.")
    
def randn(*shape, dtype=np.float32):
    return Tensor(np.random.randn(*shape).astype(dtype), dtype=dtype)

def zeros(*shape, dtype=np.float32):
    return Tensor(np.zeros(shape, dtype=dtype), dtype=dtype)