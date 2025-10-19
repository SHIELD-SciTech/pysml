"""
PySML Neural Network Activation Layers
"""

from .module import Module
from ..tensor import Tensor
from .. import engine


class ReLU(Module):
    """
    Applies the rectified linear unit function element-wise.
    
    ReLU(x) = max(0, x)
    
    Example:
        >>> layer = ReLU()
        >>> input = randn(2, 3)
        >>> output = layer(input)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply ReLU activation"""
        return engine.relu(x)
    
    def __repr__(self):
        return "ReLU()"


class LeakyReLU(Module):
    """
    Applies the leaky rectified linear unit function element-wise.
    
    LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)
    
    Args:
        negative_slope: Controls the angle of the negative slope. Default: 0.01
    
    Example:
        >>> layer = LeakyReLU(0.1)
        >>> input = randn(2, 3)
        >>> output = layer(input)
    """
    
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply Leaky ReLU activation"""
        return engine.maximum(x, self.negative_slope * x)
    
    def __repr__(self):
        return f"LeakyReLU(negative_slope={self.negative_slope})"


class GELU(Module):
    """
    Applies the Gaussian Error Linear Units function.
    
    GELU(x) = x * Φ(x)
    where Φ(x) is the Cumulative Distribution Function for Gaussian Distribution
    
    Example:
        >>> layer = GELU()
        >>> input = randn(2, 3)
        >>> output = layer(input)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply GELU activation"""
        return engine.gelu(x)
    
    def __repr__(self):
        return "GELU()"


class Sigmoid(Module):
    """
    Applies the sigmoid function element-wise.
    
    Sigmoid(x) = 1 / (1 + exp(-x))
    
    Example:
        >>> layer = Sigmoid()
        >>> input = randn(2, 3)
        >>> output = layer(input)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply sigmoid activation"""
        return engine.sigmoid(x)
    
    def __repr__(self):
        return "Sigmoid()"


class Tanh(Module):
    """
    Applies the hyperbolic tangent function element-wise.
    
    Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    Example:
        >>> layer = Tanh()
        >>> input = randn(2, 3)
        >>> output = layer(input)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply tanh activation"""
        return engine.tanh(x)
    
    def __repr__(self):
        return "Tanh()"


class Softmax(Module):
    """
    Applies the softmax function.
    
    Softmax(x_i) = exp(x_i) / sum_j exp(x_j)
    
    Args:
        dim: A dimension along which softmax will be computed. Default: -1
    
    Example:
        >>> layer = Softmax(dim=1)
        >>> input = randn(2, 3)
        >>> output = layer(input)
    """
    
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply softmax activation"""
        return engine.softmax(x, axis=self.dim)
    
    def __repr__(self):
        return f"Softmax(dim={self.dim})"


class LogSoftmax(Module):
    """
    Applies the log(softmax(x)) function.
    
    Numerically more stable than applying log after softmax.
    
    Args:
        dim: A dimension along which log_softmax will be computed. Default: -1
    
    Example:
        >>> layer = LogSoftmax(dim=1)
        >>> input = randn(2, 3)
        >>> output = layer(input)
    """
    
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply log softmax"""
        return engine.log(engine.softmax(x, axis=self.dim))
    
    def __repr__(self):
        return f"LogSoftmax(dim={self.dim})"


class ELU(Module):
    """
    Applies the exponential linear unit function element-wise.
    
    ELU(x) = max(0, x) + min(0, alpha * (exp(x) - 1))
    
    Args:
        alpha: The alpha value for ELU. Default: 1.0
    
    Example:
        >>> layer = ELU(alpha=1.0)
        >>> input = randn(2, 3)
        >>> output = layer(input)
    """
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply ELU activation"""
        return engine.where(
            x > 0,
            x,
            self.alpha * (engine.exp(x) - 1)
        )
    
    def __repr__(self):
        return f"ELU(alpha={self.alpha})"


class Softplus(Module):
    """
    Applies the softplus function element-wise.
    
    Softplus(x) = log(1 + exp(x))
    
    Example:
        >>> layer = Softplus()
        >>> input = randn(2, 3)
        >>> output = layer(input)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply softplus activation"""
        return engine.log(1 + engine.exp(x))
    
    def __repr__(self):
        return "Softplus()"


class Mish(Module):
    """
    Applies the Mish function element-wise.
    
    Mish(x) = x * tanh(softplus(x))
    
    Example:
        >>> layer = Mish()
        >>> input = randn(2, 3)
        >>> output = layer(input)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply Mish activation"""
        return x * engine.tanh(engine.log(1 + engine.exp(x)))
    
    def __repr__(self):
        return "Mish()"


class Swish(Module):
    """
    Applies the Swish function element-wise.
    
    Swish(x) = x * sigmoid(x)
    
    Also known as SiLU (Sigmoid Linear Unit).
    
    Example:
        >>> layer = Swish()
        >>> input = randn(2, 3)
        >>> output = layer(input)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply Swish activation"""
        return x * engine.sigmoid(x)
    
    def __repr__(self):
        return "Swish()"


# Alias for Swish
SiLU = Swish


class Hardswish(Module):
    """
    Applies the Hardswish function element-wise.
    
    Hardswish(x) = x * ReLU6(x + 3) / 6
    
    Computationally cheaper approximation of Swish.
    
    Example:
        >>> layer = Hardswish()
        >>> input = randn(2, 3)
        >>> output = layer(input)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply Hardswish activation"""
        return x * engine.clip(x + 3, 0, 6) / 6
    
    def __repr__(self):
        return "Hardswish()"


class PReLU(Module):
    """
    Applies the parametric rectified linear unit function element-wise.
    
    PReLU(x) = max(0, x) + a * min(0, x)
    
    where 'a' is a learnable parameter.
    
    Args:
        num_parameters: Number of 'a' to learn. Default: 1
        init: Initial value of 'a'. Default: 0.25
    
    Example:
        >>> layer = PReLU()
        >>> input = randn(2, 3)
        >>> output = layer(input)
    """
    
    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super().__init__()
        self.num_parameters = num_parameters
        self.weight = engine.full((num_parameters,), init, requires_grad=True)
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply PReLU activation"""
        return engine.maximum(0, x) + self.weight * engine.minimum(0, x)
    
    def __repr__(self):
        return f"PReLU(num_parameters={self.num_parameters})"