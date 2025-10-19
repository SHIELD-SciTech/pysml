"""
PySML Neural Network Linear Layers
"""

import math
from .module import Module
from ..tensor import Tensor
from .. import engine


class Linear(Module):
    """
    Applies a linear transformation: y = xW^T + b
    
    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        bias: If True, adds a learnable bias. Default: True
    
    Shape:
        - Input: (*, in_features) where * means any number of dimensions
        - Output: (*, out_features)
    
    Attributes:
        weight: Learnable weights of shape (out_features, in_features)
        bias: Learnable bias of shape (out_features)
    
    Example:
        >>> layer = Linear(20, 30)
        >>> input = randn(128, 20)
        >>> output = layer(input)
        >>> print(output.shape)
        (128, 30)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weight using Kaiming/He initialization
        # stddev = sqrt(2 / in_features) for ReLU networks
        # stddev = sqrt(1 / in_features) for other activations
        stddev = math.sqrt(1.0 / in_features)
        
        self.weight = engine.randn(out_features, in_features, requires_grad=True) * stddev
        
        if bias:
            self.bias = engine.zeros(out_features, requires_grad=True)
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of linear layer.
        
        Args:
            x: Input tensor of shape (*, in_features)
        
        Returns:
            Output tensor of shape (*, out_features)
        """
        # x @ W^T + b
        output = x @ self.weight.T
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"


class Bilinear(Module):
    """
    Applies a bilinear transformation: y = x1^T W x2 + b
    
    Args:
        in1_features: Size of first input
        in2_features: Size of second input
        out_features: Size of output
        bias: If True, adds a learnable bias. Default: True
    
    Shape:
        - Input1: (*, in1_features)
        - Input2: (*, in2_features)
        - Output: (*, out_features)
    """
    
    def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool = True):
        super().__init__()
        
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        
        stddev = math.sqrt(1.0 / (in1_features * in2_features))
        
        self.weight = engine.randn(out_features, in1_features, in2_features, requires_grad=True) * stddev
        
        if bias:
            self.bias = engine.zeros(out_features, requires_grad=True)
        else:
            self.bias = None
    
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Forward pass of bilinear layer.
        
        Args:
            x1: First input tensor
            x2: Second input tensor
        
        Returns:
            Output tensor
        """
        # This is a simplified implementation
        # Full implementation would be: einsum('bi,oij,bj->bo', x1, weight, x2)
        batch_size = x1.shape[0]
        output = engine.zeros(batch_size, self.out_features)
        
        # For each output dimension
        for o in range(self.out_features):
            # x1 @ W[o] @ x2.T for each sample
            result = (x1 @ self.weight[o]) * x2
            output[:, o] = engine.sum(result, axis=1)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def __repr__(self):
        return f"Bilinear(in1_features={self.in1_features}, in2_features={self.in2_features}, out_features={self.out_features})"


class Identity(Module):
    """
    A placeholder identity operator that returns input unchanged.
    
    Example:
        >>> layer = Identity()
        >>> input = randn(10, 20)
        >>> output = layer(input)
        >>> assert output is input
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        """Return input unchanged"""
        return x
    
    def __repr__(self):
        return "Identity()"


class Flatten(Module):
    """
    Flattens a contiguous range of dims into a tensor.
    
    Args:
        start_dim: First dim to flatten. Default: 1
        end_dim: Last dim to flatten. Default: -1
    
    Example:
        >>> layer = Flatten()
        >>> input = randn(32, 3, 28, 28)
        >>> output = layer(input)
        >>> print(output.shape)
        (32, 2352)
    """
    
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
    
    def forward(self, x: Tensor) -> Tensor:
        """Flatten the tensor"""
        return engine.flatten(x, start_dim=self.start_dim, end_dim=self.end_dim)
    
    def __repr__(self):
        return f"Flatten(start_dim={self.start_dim}, end_dim={self.end_dim})"


class Unflatten(Module):
    """
    Unflattens a tensor dim into a specified shape.
    
    Args:
        dim: Dimension to unflatten
        unflattened_size: New shape of the unflattened dimension
    
    Example:
        >>> layer = Unflatten(1, (3, 28, 28))
        >>> input = randn(32, 2352)
        >>> output = layer(input)
        >>> print(output.shape)
        (32, 3, 28, 28)
    """
    
    def __init__(self, dim: int, unflattened_size: tuple):
        super().__init__()
        self.dim = dim
        self.unflattened_size = unflattened_size
    
    def forward(self, x: Tensor) -> Tensor:
        """Unflatten the tensor"""
        shape = list(x.shape)
        shape[self.dim:self.dim+1] = self.unflattened_size
        return engine.reshape(x, tuple(shape))
    
    def __repr__(self):
        return f"Unflatten(dim={self.dim}, unflattened_size={self.unflattened_size})"


class Embedding(Module):
    """
    A simple lookup table that stores embeddings of a fixed dictionary and size.
    
    Args:
        num_embeddings: Size of the dictionary of embeddings
        embedding_dim: The size of each embedding vector
        padding_idx: If specified, entries at this index do not contribute to gradient
    
    Shape:
        - Input: (*) LongTensor of arbitrary shape containing indices
        - Output: (*, embedding_dim) where * is the input shape
    
    Example:
        >>> embedding = Embedding(10, 3)
        >>> input = Tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        >>> output = embedding(input)
        >>> print(output.shape)
        (2, 4, 3)
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = None):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        # Initialize embeddings with normal distribution
        self.weight = engine.randn(num_embeddings, embedding_dim, requires_grad=True) * 0.02
        
        # Zero out padding index if specified
        if padding_idx is not None:
            self.weight.data[padding_idx] = 0
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Look up embeddings for input indices.
        
        Args:
            x: Tensor of indices
        
        Returns:
            Tensor of embeddings
        """
        # In a full implementation, this would use proper indexing
        # For now, we'll assume x contains one-hot encoded indices
        # or we multiply by one-hot representation
        
        # Simplified: if x is (batch, seq_len), output is (batch, seq_len, embedding_dim)
        # This is a placeholder - proper implementation needs index-based lookup
        
        # For demonstration purposes with the current framework:
        # Convert indices to embeddings by matrix multiplication with one-hot
        return x @ self.weight
    
    def __repr__(self):
        s = f"Embedding({self.num_embeddings}, {self.embedding_dim}"
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        s += ")"
        return s


class LazyLinear(Module):
    """
    A linear layer that infers input size on first forward pass.
    
    Args:
        out_features: Size of output
        bias: If True, adds a learnable bias
    
    Example:
        >>> layer = LazyLinear(30)
        >>> input = randn(128, 20)  # Input size determined automatically
        >>> output = layer(input)
        >>> print(output.shape)
        (128, 30)
    """
    
    def __init__(self, out_features: int, bias: bool = True):
        super().__init__()
        
        self.out_features = out_features
        self.use_bias = bias
        
        self.weight = None
        self.bias = None
        self.in_features = None
    
    def _initialize(self, in_features: int):
        """Initialize weights when input size is known"""
        self.in_features = in_features
        
        stddev = math.sqrt(1.0 / in_features)
        self.weight = engine.randn(self.out_features, in_features, requires_grad=True) * stddev
        
        if self.use_bias:
            self.bias = engine.zeros(self.out_features, requires_grad=True)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass, initializing weights if needed"""
        if self.weight is None:
            # Infer input size from first forward pass
            in_features = x.shape[-1]
            self._initialize(in_features)
        
        output = x @ self.weight.T
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def __repr__(self):
        if self.in_features is None:
            return f"LazyLinear(in_features=?, out_features={self.out_features}, bias={self.use_bias})"
        return f"LazyLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias})"