from pysml.tensor import Tensor, dtype
import pysml.operations as ops


class Linear:
    def __init__(self, in_features, out_features, bias=True, dtype=dtype.float32):
        super().__init__()
        self.weight = Tensor(ops.randn(in_features, out_features).data * 0.02, dtype=dtype, requires_grad=True)
        self.bias = Tensor(ops.zeros(out_features, dtype=dtype).data, dtype=dtype, requires_grad=True) if bias else None

    def __call__(self, x: Tensor):
        # Use autograd-aware tensor operations instead of ops functions
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y
    
