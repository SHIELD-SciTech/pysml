from .module import Module, Parameter


class ReLU(Module):
    
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace
    
    def forward(self, x):
        from .. import engine
        return engine.relu(x)
    
    def extra_repr(self):
        return f"inplace={self.inplace}" if self.inplace else ""


class LeakyReLU(Module):
    
    def __init__(self, negative_slope=0.01, inplace=False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace
    
    def forward(self, x):
        from .. import engine
        # LeakyReLU: max(x, 0) + negative_slope * min(x, 0)
        pos = engine.maximum(x, 0)
        neg = engine.minimum(x, 0)
        return engine.add(pos, engine.multiply(neg, self.negative_slope))
    
    def extra_repr(self):
        return f"negative_slope={self.negative_slope}"


class PReLU(Module):
    
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.num_parameters = num_parameters
        
        # Initialize learnable slope parameter
        from .. import Tensor
        import numpy as np
        data = np.full((num_parameters,), init)
        self.weight = Parameter(Tensor(data, requires_grad=True))
    
    def forward(self, x):
        from .. import engine
        # PReLU: max(x, 0) + weight * min(x, 0)
        pos = engine.maximum(x, 0)
        neg = engine.minimum(x, 0)
        return engine.add(pos, engine.multiply(neg, self.weight.data))
    
    def extra_repr(self):
        return f"num_parameters={self.num_parameters}"


class ELU(Module):
    
    def __init__(self, alpha=1.0, inplace=False):
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace
    
    def forward(self, x):
        from .. import engine
        # ELU: x if x > 0 else alpha * (exp(x) - 1)
        backend = x._backend
        
        # Create mask for positive values
        mask = backend.greater(x.data, 0)
        
        # Compute exp(x) - 1 for negative values
        exp_part = engine.multiply(
            self.alpha,
            engine.subtract(engine.exp(x), 1.0)
        )
        
        # Combine: x where positive, alpha*(exp(x)-1) where negative
        return engine.where(mask, x, exp_part)
    
    def extra_repr(self):
        return f"alpha={self.alpha}"


class GELU(Module):
    
    def __init__(self, approximate='none'):
        super().__init__()
        self.approximate = approximate
        # 'none' uses exact computation
        # 'tanh' uses tanh approximation (faster, already implemented in backend)
    
    def forward(self, x):
        from .. import engine
        return engine.gelu(x)
    
    def extra_repr(self):
        return f"approximate='{self.approximate}'"


class SiLU(Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        from .. import engine
        return engine.silu(x)


class Swish(SiLU):
    # Swish is just an alias for SiLU
    pass


class Tanh(Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        from .. import engine
        return engine.tanh(x)


class Sigmoid(Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        from .. import engine
        return engine.sigmoid(x)


class Softmax(Module):
    
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        from .. import engine
        return engine.softmax(x, axis=self.dim)
    
    def extra_repr(self):
        return f"dim={self.dim}"


class LogSoftmax(Module):
    
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        from .. import engine
        return engine.log_softmax(x, axis=self.dim)
    
    def extra_repr(self):
        return f"dim={self.dim}"


class Softplus(Module):
    
    def __init__(self, beta=1.0, threshold=20.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
    
    def forward(self, x):
        from .. import engine
        # Softplus: (1/beta) * log(1 + exp(beta * x))
        # For numerical stability: if beta*x > threshold, return x
        
        backend = x._backend
        beta_x = engine.multiply(x, self.beta)
        
        # Mask for large values
        mask = backend.greater(beta_x.data, self.threshold)
        
        # Softplus computation
        softplus_val = engine.multiply(
            1.0 / self.beta,
            engine.log(engine.add(1.0, engine.exp(beta_x)))
        )
        
        # Return x for large values, softplus otherwise
        return engine.where(mask, x, softplus_val)
    
    def extra_repr(self):
        return f"beta={self.beta}, threshold={self.threshold}"


class Mish(Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        from .. import engine
        # Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        softplus = engine.log(engine.add(1.0, engine.exp(x)))
        return engine.multiply(x, engine.tanh(softplus))


class Hardswish(Module):
    
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace
    
    def forward(self, x):
        from .. import engine
        # Hardswish: x * ReLU6(x + 3) / 6
        # ReLU6(x) = min(max(x, 0), 6)
        
        x_plus_3 = engine.add(x, 3.0)
        relu6 = engine.clip(x_plus_3, 0.0, 6.0)
        return engine.multiply(x, engine.divide(relu6, 6.0))


class Hardsigmoid(Module):
    
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace
    
    def forward(self, x):
        from .. import engine
        # Hardsigmoid: ReLU6(x + 3) / 6
        x_plus_3 = engine.add(x, 3.0)
        return engine.divide(engine.clip(x_plus_3, 0.0, 6.0), 6.0)


class GLU(Module):
    
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        from .. import engine
        # GLU: split input in half, apply sigmoid to one half, multiply
        # x: (..., 2*hidden_size)
        # output: (..., hidden_size)
        
        backend = x._backend
        chunks = backend.split(x.data, 2, axis=self.dim)
        
        from .. import Tensor
        a = Tensor.__new__(Tensor)
        a._backend = backend
        a._dtype = x._dtype
        a.device = x.device
        a.active_device = x.active_device
        a.data = chunks[0]
        a._requires_grad = x._requires_grad
        a._grad = None
        
        b = Tensor.__new__(Tensor)
        b._backend = backend
        b._dtype = x._dtype
        b.device = x.device
        b.active_device = x.active_device
        b.data = chunks[1]
        b._requires_grad = x._requires_grad
        b._grad = None
        
        # a * sigmoid(b)
        return engine.multiply(a, engine.sigmoid(b))
    
    def extra_repr(self):
        return f"dim={self.dim}"


class SwiGLU(Module):
    
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        from .. import engine
        # SwiGLU: split input in half, apply SiLU to one half, multiply
        # Used in modern LLMs (PaLM, LLaMA)
        
        backend = x._backend
        chunks = backend.split(x.data, 2, axis=self.dim)
        
        from .. import Tensor
        a = Tensor.__new__(Tensor)
        a._backend = backend
        a._dtype = x._dtype
        a.device = x.device
        a.active_device = x.active_device
        a.data = chunks[0]
        a._requires_grad = x._requires_grad
        a._grad = None
        
        b = Tensor.__new__(Tensor)
        b._backend = backend
        b._dtype = x._dtype
        b.device = x.device
        b.active_device = x.active_device
        b.data = chunks[1]
        b._requires_grad = x._requires_grad
        b._grad = None
        
        # a * silu(b)
        return engine.multiply(a, engine.silu(b))
    
    def extra_repr(self):
        return f"dim={self.dim}"


class SELU(Module):
    
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace
        # SELU constants (self-normalizing property)
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
    
    def forward(self, x):
        from .. import engine
        # SELU: scale * (x if x > 0 else alpha * (exp(x) - 1))
        backend = x._backend
        
        mask = backend.greater(x.data, 0)
        
        # ELU part
        elu_part = engine.multiply(
            self.alpha,
            engine.subtract(engine.exp(x), 1.0)
        )
        
        # SELU = scale * (x or elu_part)
        result = engine.where(mask, x, elu_part)
        return engine.multiply(result, self.scale)


class Softmin(Module):
    
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        from .. import engine
        # Softmin: softmax(-x)
        neg_x = engine.negative(x)
        return engine.softmax(neg_x, axis=self.dim)
    
    def extra_repr(self):
        return f"dim={self.dim}"


class LogSigmoid(Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        from .. import engine
        # LogSigmoid: log(sigmoid(x)) = -log(1 + exp(-x))
        # For numerical stability
        return engine.negative(
            engine.log(
                engine.add(1.0, engine.exp(engine.negative(x)))
            )
        )


class Tanhshrink(Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        from .. import engine
        # Tanhshrink: x - tanh(x)
        return engine.subtract(x, engine.tanh(x))


class Softshrink(Module):
    
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd
    
    def forward(self, x):
        from .. import engine
        # Softshrink: x - lambd if x > lambd
        #             x + lambd if x < -lambd
        #             0 otherwise
        backend = x._backend
        
        mask_pos = backend.greater(x.data, self.lambd)
        mask_neg = backend.less(x.data, -self.lambd)
        
        pos_part = engine.subtract(x, self.lambd)
        neg_part = engine.add(x, self.lambd)
        
        # Start with zeros
        from .. import Tensor
        result = Tensor.__new__(Tensor)
        result._backend = backend
        result._dtype = x._dtype
        result.device = x.device
        result.active_device = x.active_device
        result.data = backend.zeros_like(x.data)
        result._requires_grad = x._requires_grad
        result._grad = None
        
        # Apply masks
        result = engine.where(mask_pos, pos_part, result)
        result = engine.where(mask_neg, neg_part, result)
        
        return result
    
    def extra_repr(self):
        return f"lambd={self.lambd}"


class Hardshrink(Module):
    
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd
    
    def forward(self, x):
        from .. import engine
        # Hardshrink: x if |x| > lambd else 0
        backend = x._backend
        
        mask = backend.greater(backend.abs(x.data), self.lambd)
        
        from .. import Tensor
        zeros = Tensor.__new__(Tensor)
        zeros._backend = backend
        zeros._dtype = x._dtype
        zeros.device = x.device
        zeros.active_device = x.active_device
        zeros.data = backend.zeros_like(x.data)
        zeros._requires_grad = False
        zeros._grad = None
        
        return engine.where(mask, x, zeros)
    
    def extra_repr(self):
        return f"lambd={self.lambd}"


class MultiheadAttention(Module):
    # Placeholder - will be implemented in attention.py
    pass


__all__ = [
    'ReLU', 'LeakyReLU', 'PReLU', 'ELU', 'SELU',
    'GELU', 'SiLU', 'Swish', 'Mish',
    'Tanh', 'Sigmoid', 'Hardsigmoid', 'Hardswish',
    'Softmax', 'LogSoftmax', 'Softmin', 'LogSigmoid',
    'Softplus', 'Softshrink', 'Hardshrink', 'Tanhshrink',
    'GLU', 'SwiGLU',
]