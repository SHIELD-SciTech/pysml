try:
    import dpnp as np
except:
    import numpy as np
from pysml.tensor import Tensor

backend = np


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
        return Tensor(e_x / e_x.sum(axis=axis, keepdims=True), dtype=t.dtype)
    else:
        raise ValueError("Input must be a Tensor instance.")


def mean(t, axis=None):
    if isinstance(t, Tensor):
        return Tensor(np.mean(t.data, axis=axis), dtype=t.dtype)
    else:
        raise ValueError("Input must be a Tensor instance.")
