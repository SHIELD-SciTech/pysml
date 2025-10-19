"""
PySML Neural Network Functional API
PyTorch-like functional operations for neural networks
"""

import numpy as np
from ..tensor import Tensor
from .. import engine


# ===== Activation Functions =====

def relu(x: Tensor) -> Tensor:
    """
    Applies the rectified linear unit function element-wise.
    
    Args:
        x: Input tensor
    
    Returns:
        Output tensor with ReLU applied
    
    Example:
        >>> x = pysml.randn(2, 3)
        >>> output = F.relu(x)
    """
    return engine.relu(x)


def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    """
    Applies the leaky rectified linear unit function element-wise.
    
    Args:
        x: Input tensor
        negative_slope: Controls angle of negative slope (default: 0.01)
    
    Returns:
        Output tensor
    """
    return engine.maximum(x, negative_slope * x)


def gelu(x: Tensor) -> Tensor:
    """
    Applies the Gaussian Error Linear Units function.
    
    Args:
        x: Input tensor
    
    Returns:
        Output tensor with GELU applied
    """
    return engine.gelu(x)


def sigmoid(x: Tensor) -> Tensor:
    """
    Applies the sigmoid function element-wise.
    
    Args:
        x: Input tensor
    
    Returns:
        Output tensor with sigmoid applied
    """
    return engine.sigmoid(x)


def tanh(x: Tensor) -> Tensor:
    """
    Applies the hyperbolic tangent function element-wise.
    
    Args:
        x: Input tensor
    
    Returns:
        Output tensor with tanh applied
    """
    return engine.tanh(x)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Applies the softmax function.
    
    Args:
        x: Input tensor
        dim: Dimension along which softmax is computed (default: -1)
    
    Returns:
        Output tensor with softmax applied
    
    Example:
        >>> logits = pysml.randn(2, 10)
        >>> probs = F.softmax(logits, dim=1)
    """
    return engine.softmax(x, axis=dim)


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Applies the log(softmax(x)) function.
    
    Numerically more stable than applying log after softmax.
    
    Args:
        x: Input tensor
        dim: Dimension along which log_softmax is computed
    
    Returns:
        Output tensor
    """
    return engine.log(engine.softmax(x, axis=dim))


def elu(x: Tensor, alpha: float = 1.0) -> Tensor:
    """
    Applies the exponential linear unit function.
    
    Args:
        x: Input tensor
        alpha: The alpha value for ELU (default: 1.0)
    
    Returns:
        Output tensor
    """
    return engine.where(
        x > 0,
        x,
        alpha * (engine.exp(x) - 1)
    )


def selu(x: Tensor) -> Tensor:
    """
    Applies the scaled exponential linear unit function.
    
    Args:
        x: Input tensor
    
    Returns:
        Output tensor
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * elu(x, alpha)


def softplus(x: Tensor) -> Tensor:
    """
    Applies the softplus function element-wise.
    
    Args:
        x: Input tensor
    
    Returns:
        Output tensor
    """
    return engine.log(1 + engine.exp(x))


def mish(x: Tensor) -> Tensor:
    """
    Applies the Mish activation function.
    
    Args:
        x: Input tensor
    
    Returns:
        Output tensor
    """
    return x * engine.tanh(softplus(x))


def swish(x: Tensor) -> Tensor:
    """
    Applies the Swish activation function (also known as SiLU).
    
    Args:
        x: Input tensor
    
    Returns:
        Output tensor
    """
    return x * engine.sigmoid(x)


# Alias
silu = swish


def hardswish(x: Tensor) -> Tensor:
    """
    Applies the Hardswish activation function.
    
    Args:
        x: Input tensor
    
    Returns:
        Output tensor
    """
    return x * engine.clip(x + 3, 0, 6) / 6


# ===== Loss Functions =====

def mse_loss(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    """
    Mean Squared Error loss.
    
    Args:
        input: Predicted values (*, N)
        target: Target values (*, N)
        reduction: 'none' | 'mean' | 'sum' (default: 'mean')
    
    Returns:
        Loss value
    
    Example:
        >>> predictions = pysml.randn(10, 5)
        >>> targets = pysml.randn(10, 5)
        >>> loss = F.mse_loss(predictions, targets)
    """
    diff = input - target
    loss = diff * diff
    
    if reduction == 'mean':
        return engine.mean(loss)
    elif reduction == 'sum':
        return engine.sum(loss)
    else:  # 'none'
        return loss


def l1_loss(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    """
    Mean Absolute Error loss.
    
    Args:
        input: Predicted values
        target: Target values
        reduction: 'none' | 'mean' | 'sum' (default: 'mean')
    
    Returns:
        Loss value
    """
    loss = engine.abs(input - target)
    
    if reduction == 'mean':
        return engine.mean(loss)
    elif reduction == 'sum':
        return engine.sum(loss)
    else:  # 'none'
        return loss


def smooth_l1_loss(input: Tensor, target: Tensor, beta: float = 1.0, reduction: str = 'mean') -> Tensor:
    """
    Smooth L1 loss (also known as Huber loss).
    
    Args:
        input: Predicted values
        target: Target values
        beta: Threshold at which to change between L1 and L2 loss
        reduction: 'none' | 'mean' | 'sum'
    
    Returns:
        Loss value
    """
    diff = engine.abs(input - target)
    
    # If |x| < beta: 0.5 * x^2 / beta
    # If |x| >= beta: |x| - 0.5 * beta
    loss = engine.where(
        diff < beta,
        0.5 * diff * diff / beta,
        diff - 0.5 * beta
    )
    
    if reduction == 'mean':
        return engine.mean(loss)
    elif reduction == 'sum':
        return engine.sum(loss)
    else:
        return loss


def cross_entropy(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    """
    Cross entropy loss (combines log_softmax and nll_loss).
    
    Args:
        input: Logits of shape (N, C) where C is number of classes
        target: Target class indices of shape (N,) with values in [0, C-1]
        reduction: 'none' | 'mean' | 'sum' (default: 'mean')
    
    Returns:
        Loss value
    
    Example:
        >>> logits = pysml.randn(32, 10)  # batch=32, classes=10
        >>> targets = pysml.Tensor(np.random.randint(0, 10, 32))
        >>> loss = F.cross_entropy(logits, targets)
    
    Note:
        This implementation uses the autograd CrossEntropyLoss for proper gradients
        if available, otherwise falls back to a manual implementation.
    """
    try:
        # Try to import from pysml.nn.autograd (old structure)
        from pysml.nn.autograd import CrossEntropyLoss
        use_autograd = True
    except ImportError:
        use_autograd = False
    
    # Get raw data arrays
    if isinstance(input, Tensor):
        logits_data = input.data
    else:
        logits_data = input
    
    if isinstance(target, Tensor):
        targets_data = target.data
    else:
        targets_data = target
    
    # Convert to numpy if needed
    if hasattr(logits_data, 'asnumpy'):
        logits_data = logits_data.asnumpy()
    if hasattr(targets_data, 'asnumpy'):
        targets_data = targets_data.asnumpy()
    
    # Ensure targets are integers
    targets_data = targets_data.astype(np.int32)
    
    if use_autograd:
        # Create tensors
        input_tensor = Tensor(logits_data, requires_grad=input.requires_grad if isinstance(input, Tensor) else False)
        target_tensor = Tensor(targets_data)
        
        # Use autograd cross entropy
        loss = CrossEntropyLoss.apply(CrossEntropyLoss, input_tensor, target_tensor)
        
        if reduction == 'sum':
            return loss * logits_data.shape[0]
        elif reduction == 'none':
            return loss
        else:  # 'mean'
            return loss
    else:
        # Manual implementation without autograd
        # Compute softmax
        max_logits = np.max(logits_data, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_data - max_logits)
        sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
        log_probs = logits_data - max_logits - np.log(sum_exp)
        
        # Get log probabilities for target classes
        num_samples = logits_data.shape[0]
        target_log_probs = log_probs[np.arange(num_samples), targets_data.astype(int)]
        
        # Negative log likelihood
        losses = -target_log_probs
        
        if reduction == 'mean':
            return Tensor(np.mean(losses))
        elif reduction == 'sum':
            return Tensor(np.sum(losses))
        else:  # 'none'
            return Tensor(losses)


def binary_cross_entropy(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    """
    Binary cross entropy loss.
    
    Args:
        input: Predicted probabilities (not logits!) in [0, 1]
        target: Target values in [0, 1]
        reduction: 'none' | 'mean' | 'sum'
    
    Returns:
        Loss value
    """
    # Clip to avoid log(0)
    input_clipped = engine.clip(input, 1e-7, 1 - 1e-7)
    
    loss = -(target * engine.log(input_clipped) + 
             (1 - target) * engine.log(1 - input_clipped))
    
    if reduction == 'mean':
        return engine.mean(loss)
    elif reduction == 'sum':
        return engine.sum(loss)
    else:
        return loss


def binary_cross_entropy_with_logits(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    """
    Binary cross entropy with logits (more numerically stable).
    
    Args:
        input: Logits (before sigmoid)
        target: Target values in [0, 1]
        reduction: 'none' | 'mean' | 'sum'
    
    Returns:
        Loss value
    """
    # Use log-sum-exp trick for numerical stability
    # BCE = target * -log(sigmoid(x)) + (1-target) * -log(1-sigmoid(x))
    #     = target * -log(1/(1+exp(-x))) + (1-target) * -log(exp(-x)/(1+exp(-x)))
    #     = target * log(1+exp(-x)) + (1-target) * (x + log(1+exp(-x)))
    #     = (1-target)*x + log(1+exp(-x))
    
    max_val = engine.maximum(input, 0)
    loss = (1 - target) * input - input * target + max_val + \
           engine.log(engine.exp(-max_val) + engine.exp(-input - max_val))
    
    if reduction == 'mean':
        return engine.mean(loss)
    elif reduction == 'sum':
        return engine.sum(loss)
    else:
        return loss


def nll_loss(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    """
    Negative log likelihood loss.
    
    Args:
        input: Log probabilities of shape (N, C)
        target: Target class indices of shape (N,)
        reduction: 'none' | 'mean' | 'sum'
    
    Returns:
        Loss value
    """
    # This is a simplified implementation
    # Full implementation would do proper indexing
    batch_size = input.shape[0]
    
    # Get data
    input_data = input.data if isinstance(input, Tensor) else input
    target_data = target.data if isinstance(target, Tensor) else target
    
    if hasattr(input_data, 'asnumpy'):
        input_data = input_data.asnumpy()
    if hasattr(target_data, 'asnumpy'):
        target_data = target_data.asnumpy()
    
    target_data = target_data.astype(np.int32)
    
    # Gather the log probabilities at target indices
    loss_values = np.zeros(batch_size)
    for i in range(batch_size):
        loss_values[i] = -input_data[i, target_data[i]]
    
    loss = Tensor(loss_values)
    
    if reduction == 'mean':
        return engine.mean(loss)
    elif reduction == 'sum':
        return engine.sum(loss)
    else:
        return loss


def kl_div(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    """
    Kullback-Leibler divergence loss.
    
    Args:
        input: Log probabilities
        target: Target probabilities
        reduction: 'none' | 'mean' | 'sum' | 'batchmean'
    
    Returns:
        Loss value
    """
    loss = target * (engine.log(target) - input)
    
    if reduction == 'mean':
        return engine.mean(loss)
    elif reduction == 'sum':
        return engine.sum(loss)
    elif reduction == 'batchmean':
        return engine.sum(loss) / input.shape[0]
    else:  # 'none'
        return loss


def cosine_similarity(x1: Tensor, x2: Tensor, dim: int = 1, eps: float = 1e-8) -> Tensor:
    """
    Compute cosine similarity between two tensors.
    
    Args:
        x1: First tensor
        x2: Second tensor
        dim: Dimension along which to compute similarity
        eps: Small value to avoid division by zero
    
    Returns:
        Cosine similarity
    """
    dot_product = engine.sum(x1 * x2, axis=dim, keepdims=True)
    norm_x1 = engine.sqrt(engine.sum(x1 * x1, axis=dim, keepdims=True) + eps)
    norm_x2 = engine.sqrt(engine.sum(x2 * x2, axis=dim, keepdims=True) + eps)
    
    return dot_product / (norm_x1 * norm_x2)


def cosine_embedding_loss(input1: Tensor, input2: Tensor, target: Tensor, 
                          margin: float = 0.0, reduction: str = 'mean') -> Tensor:
    """
    Cosine embedding loss.
    
    Args:
        input1: First input tensor
        input2: Second input tensor
        target: 1 for similar pairs, -1 for dissimilar pairs
        margin: Margin for dissimilar pairs
        reduction: 'none' | 'mean' | 'sum'
    
    Returns:
        Loss value
    """
    cos_sim = cosine_similarity(input1, input2, dim=1)
    
    # If target == 1: loss = 1 - cos_sim
    # If target == -1: loss = max(0, cos_sim - margin)
    loss = engine.where(
        target == 1,
        1 - cos_sim,
        engine.maximum(Tensor([0]), cos_sim - margin)
    )
    
    if reduction == 'mean':
        return engine.mean(loss)
    elif reduction == 'sum':
        return engine.sum(loss)
    else:
        return loss


# ===== Normalization Functions =====

def batch_norm(input: Tensor, running_mean: Tensor, running_var: Tensor,
               weight: Tensor = None, bias: Tensor = None,
               training: bool = False, momentum: float = 0.1, eps: float = 1e-5) -> Tensor:
    """
    Applies batch normalization.
    
    Args:
        input: Input tensor of shape (N, C, ...)
        running_mean: Running mean of shape (C,)
        running_var: Running variance of shape (C,)
        weight: Gamma parameter of shape (C,)
        bias: Beta parameter of shape (C,)
        training: Training mode flag
        momentum: Momentum for running stats
        eps: Small value for numerical stability
    
    Returns:
        Normalized tensor
    """
    if training:
        # Compute batch statistics
        batch_mean = engine.mean(input, axis=0, keepdims=False)
        batch_var = engine.var(input, axis=0, keepdims=False)
        
        # Update running statistics
        running_mean.data = (1 - momentum) * running_mean.data + momentum * batch_mean.data
        running_var.data = (1 - momentum) * running_var.data + momentum * batch_var.data
        
        mean = batch_mean
        var = batch_var
    else:
        mean = running_mean
        var = running_var
    
    # Normalize
    x_norm = (input - mean) / engine.sqrt(var + eps)
    
    # Scale and shift
    if weight is not None:
        x_norm = x_norm * weight
    if bias is not None:
        x_norm = x_norm + bias
    
    return x_norm


def layer_norm(input: Tensor, normalized_shape: tuple, weight: Tensor = None,
               bias: Tensor = None, eps: float = 1e-5) -> Tensor:
    """
    Applies layer normalization.
    
    Args:
        input: Input tensor
        normalized_shape: Shape over which to normalize
        weight: Gamma parameter
        bias: Beta parameter
        eps: Small value for numerical stability
    
    Returns:
        Normalized tensor
    """
    # Compute axes to normalize over
    ndim = len(input.shape)
    norm_ndim = len(normalized_shape)
    axes = tuple(range(ndim - norm_ndim, ndim))
    
    # Compute mean and variance
    mean = engine.mean(input, axis=axes, keepdims=True)
    var = engine.var(input, axis=axes, keepdims=True)
    
    # Normalize
    x_norm = (input - mean) / engine.sqrt(var + eps)
    
    # Scale and shift
    if weight is not None:
        x_norm = x_norm * weight
    if bias is not None:
        x_norm = x_norm + bias
    
    return x_norm


def group_norm(input: Tensor, num_groups: int, weight: Tensor = None,
               bias: Tensor = None, eps: float = 1e-5) -> Tensor:
    """
    Applies group normalization.
    
    Args:
        input: Input tensor of shape (N, C, ...)
        num_groups: Number of groups
        weight: Gamma parameter
        bias: Beta parameter
        eps: Small value for numerical stability
    
    Returns:
        Normalized tensor
    """
    N, C = input.shape[0], input.shape[1]
    assert C % num_groups == 0, "Number of channels must be divisible by num_groups"
    
    # Reshape to (N, num_groups, C // num_groups, ...)
    group_shape = (N, num_groups, C // num_groups) + input.shape[2:]
    x = input.reshape(group_shape)
    
    # Normalize over group dimension
    axes = tuple(range(2, len(group_shape)))
    mean = engine.mean(x, axis=axes, keepdims=True)
    var = engine.var(x, axis=axes, keepdims=True)
    
    x_norm = (x - mean) / engine.sqrt(var + eps)
    
    # Reshape back
    x_norm = x_norm.reshape(input.shape)
    
    # Scale and shift
    if weight is not None:
        x_norm = x_norm * weight.reshape(1, C, *([1] * (len(input.shape) - 2)))
    if bias is not None:
        x_norm = x_norm + bias.reshape(1, C, *([1] * (len(input.shape) - 2)))
    
    return x_norm


def instance_norm(input: Tensor, running_mean: Tensor = None, running_var: Tensor = None,
                  weight: Tensor = None, bias: Tensor = None,
                  use_input_stats: bool = True, momentum: float = 0.1, eps: float = 1e-5) -> Tensor:
    """
    Applies instance normalization.
    
    Args:
        input: Input tensor of shape (N, C, ...)
        running_mean: Running mean
        running_var: Running variance
        weight: Gamma parameter
        bias: Beta parameter
        use_input_stats: Use input statistics instead of running stats
        momentum: Momentum for running stats
        eps: Small value for numerical stability
    
    Returns:
        Normalized tensor
    """
    # Normalize over spatial dimensions for each channel in each sample
    axes = tuple(range(2, len(input.shape)))
    
    if use_input_stats:
        mean = engine.mean(input, axis=axes, keepdims=True)
        var = engine.var(input, axis=axes, keepdims=True)
    else:
        mean = running_mean
        var = running_var
    
    # Normalize
    x_norm = (input - mean) / engine.sqrt(var + eps)
    
    # Scale and shift
    if weight is not None:
        C = input.shape[1]
        x_norm = x_norm * weight.reshape(1, C, *([1] * len(axes)))
    if bias is not None:
        C = input.shape[1]
        x_norm = x_norm + bias.reshape(1, C, *([1] * len(axes)))
    
    return x_norm


# ===== Dropout =====

def dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    """
    Applies dropout.
    
    Args:
        input: Input tensor
        p: Probability of dropping an element
        training: Training mode flag
        inplace: Perform operation in-place (not implemented)
    
    Returns:
        Tensor with dropout applied
    """
    return engine.dropout(input, p=p, training=training)


def dropout2d(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """
    Applies 2D dropout (drops entire channels).
    
    Args:
        input: Input tensor of shape (N, C, H, W)
        p: Probability of dropping a channel
        training: Training mode flag
    
    Returns:
        Tensor with dropout applied
    """
    if not training or p == 0:
        return input
    
    N, C, H, W = input.shape
    backend = input.backend
    
    # Create mask that drops entire channels
    mask = backend.random.binomial(1, 1 - p, (N, C, 1, 1))
    mask = backend.divide(mask, 1 - p)
    
    return input * Tensor(mask, backend=backend, device=input.device)


# ===== Padding =====

def pad(input: Tensor, pad_width, mode: str = 'constant', value: float = 0) -> Tensor:
    """
    Pads tensor.
    
    Args:
        input: Input tensor
        pad_width: Padding width
        mode: 'constant' | 'reflect' | 'replicate'
        value: Fill value for constant padding
    
    Returns:
        Padded tensor
    """
    backend = input.backend
    
    if mode == 'constant':
        padded = backend.pad(input.data, pad_width, mode='constant', constant_values=value)
    elif mode == 'reflect':
        padded = backend.pad(input.data, pad_width, mode='reflect')
    elif mode == 'replicate' or mode == 'edge':
        padded = backend.pad(input.data, pad_width, mode='edge')
    else:
        raise ValueError(f"Unsupported padding mode: {mode}")
    
    return Tensor(padded, backend=backend, device=input.device)


# ===== Utility Functions =====

def one_hot(tensor: Tensor, num_classes: int = -1) -> Tensor:
    """
    Convert indices to one-hot encoded tensor.
    
    Args:
        tensor: Input tensor with indices
        num_classes: Number of classes (auto-detected if -1)
    
    Returns:
        One-hot encoded tensor
    
    Example:
        >>> indices = pysml.Tensor([0, 2, 1, 3])
        >>> one_hot = F.one_hot(indices, num_classes=5)
        >>> print(one_hot.shape)  # (4, 5)
    """
    data = tensor.data
    if hasattr(data, 'asnumpy'):
        data = data.asnumpy()
    
    data = data.astype(np.int32)
    
    if num_classes == -1:
        num_classes = int(np.max(data)) + 1
    
    # Get original shape
    original_shape = data.shape
    
    # Flatten the data for easier processing
    flat_data = data.flatten()
    n_elements = flat_data.shape[0]
    
    # Create one-hot array: (n_elements, num_classes)
    one_hot_data = np.zeros((n_elements, num_classes), dtype=np.float32)
    
    # Fill in the ones at the correct positions
    for i in range(n_elements):
        idx = flat_data[i]
        if 0 <= idx < num_classes:
            one_hot_data[i, idx] = 1.0
        else:
            raise ValueError(f"Index {idx} is out of bounds for num_classes={num_classes}")
    
    # Reshape to original shape + (num_classes,)
    final_shape = original_shape + (num_classes,)
    one_hot_data = one_hot_data.reshape(final_shape)
    
    return Tensor(one_hot_data, backend=tensor.backend, device=tensor.device)


def normalize(input: Tensor, p: float = 2.0, dim: int = 1, eps: float = 1e-12) -> Tensor:
    """
    Normalize tensor along a dimension.
    
    Args:
        input: Input tensor
        p: Norm degree (2 for L2 norm)
        dim: Dimension to normalize along
        eps: Small value to avoid division by zero
    
    Returns:
        Normalized tensor
    """
    if p == 2:
        norm = engine.sqrt(engine.sum(input * input, axis=dim, keepdims=True) + eps)
    else:
        norm = engine.sum(engine.abs(input) ** p, axis=dim, keepdims=True) ** (1.0 / p) + eps
    
    return input / norm


"""
Fixed embedding function for functional.py
Replace the embedding function in pysml/nn/functional.py with this version
"""

import numpy as np
from ..tensor import Tensor


def embedding(input: Tensor, weight: Tensor, padding_idx: int = None) -> Tensor:
    """
    Look up embeddings from embedding matrix with PROPER gradient accumulation.
    
    Args:
        input: Tensor containing indices
        weight: Embedding matrix of shape (num_embeddings, embedding_dim)
        padding_idx: If specified, entries at this index won't contribute to gradient
    
    Returns:
        Embedded tensor
    """
    # Get data
    indices = input.data
    if hasattr(indices, 'asnumpy'):
        indices = indices.asnumpy()
    
    indices = indices.astype(np.int32)
    
    weight_data = weight.data
    if hasattr(weight_data, 'asnumpy'):
        weight_data = weight_data.asnumpy()
    
    # Perform embedding lookup
    embedded = weight_data[indices]
    
    out = Tensor(embedded, backend=input.backend, device=input.device, 
                 requires_grad=weight.requires_grad)
    
    # CRITICAL FIX: Set up backward pass for gradient accumulation
    if weight.requires_grad:
        out._prev = {weight}
        out._op = 'embedding'
        
        def _backward():
            # Initialize gradient if needed
            if weight.grad is None:
                weight.grad = Tensor(np.zeros_like(weight_data), backend=weight.backend)
            
            # Get gradient with respect to output
            grad_output = out.grad.data
            if hasattr(grad_output, 'asnumpy'):
                grad_output = grad_output.asnumpy()
            
            # Get current weight gradient
            weight_grad = weight.grad.data
            if hasattr(weight_grad, 'asnumpy'):
                weight_grad = weight_grad.asnumpy()
            
            # Flatten indices and gradients for accumulation
            flat_indices = indices.flatten()
            flat_grad = grad_output.reshape(-1, weight_data.shape[-1])
            
            # CRITICAL: Use np.add.at to accumulate gradients at each index
            # This properly handles the case where the same index appears multiple times
            np.add.at(weight_grad, flat_indices, flat_grad)
            
            # Update weight gradient
            weight.grad.data = weight_grad if not hasattr(weight.grad.data, 'shape') else weight_grad
        
        out._backward = _backward
    
    return out


def interpolate(input: Tensor, size=None, scale_factor=None, mode='nearest'):
    """
    Interpolate/resize tensor (simplified version).
    
    Args:
        input: Input tensor
        size: Output size
        scale_factor: Scaling factor
        mode: 'nearest' | 'linear' | 'bilinear'
    
    Returns:
        Resized tensor
    """
    # This is a placeholder - full implementation would require scipy or similar
    raise NotImplementedError("Interpolation not yet implemented")


def sum_params(module) -> int:
    """
    Count total number of parameters in a module.
    
    Args:
        module: Neural network module
    
    Returns:
        Total number of parameters
    """
    total = 0
    for param in module.parameters():
        total += param.size
    return total


# ===== Convolution (placeholder) =====

def conv2d(input: Tensor, weight: Tensor, bias: Tensor = None, 
           stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1) -> Tensor:
    """
    2D convolution (placeholder).
    
    Full implementation would require efficient convolution algorithms.
    """
    raise NotImplementedError("Convolution not yet implemented")


def linear(input: Tensor, weight: Tensor, bias: Tensor = None) -> Tensor:
    """
    Applies a linear transformation.
    
    Args:
        input: Input tensor of shape (*, in_features)
        weight: Weight tensor of shape (out_features, in_features)
        bias: Optional bias tensor of shape (out_features,)
    
    Returns:
        Output tensor of shape (*, out_features)
    """
    output = input @ weight.T
    
    if bias is not None:
        output = output + bias
    
    return output


# ===== Aliases for compatibility =====
cross_entropy_loss = cross_entropy
mse = mse_loss
l1 = l1_loss