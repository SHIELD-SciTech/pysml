import numpy as np
from pysml.tensor import Tensor, dtype
from pysml.nn.module import Module
import pysml.operations as ops


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, 
                 stride=1, padding=0, bias=True, dtype=dtype.float32):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Handle kernel_size as int or tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        
        # Handle stride as int or tuple
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = tuple(stride)
        
        # Handle padding as int or tuple
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = tuple(padding)
        
        # Initialize weights using Kaiming/He initialization
        k_h, k_w = self.kernel_size
        fan_in = in_channels * k_h * k_w
        std = np.sqrt(2.0 / fan_in)
        
        weight_data = np.random.randn(out_channels, in_channels, k_h, k_w) * std
        self.weight = Tensor(weight_data.astype(np.float32), dtype=dtype, requires_grad=True)
        
        if bias:
            self.bias = Tensor(np.zeros(out_channels, dtype=np.float32), dtype=dtype, requires_grad=True)
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        from pysml.nn.autograd import Conv2dFunction
        
        # Use autograd-aware convolution
        output = Conv2dFunction.apply(
            Conv2dFunction, 
            x, 
            self.weight, 
            self.bias,
            self.stride, 
            self.padding
        )
        
        return output


class Conv1d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride=1, padding=0, bias=True, dtype=dtype.float32):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights
        fan_in = in_channels * kernel_size
        std = np.sqrt(2.0 / fan_in)
        
        weight_data = np.random.randn(out_channels, in_channels, kernel_size) * std
        self.weight = Tensor(weight_data.astype(np.float32), dtype=dtype, requires_grad=True)
        
        if bias:
            self.bias = Tensor(np.zeros(out_channels, dtype=np.float32), dtype=dtype, requires_grad=True)
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        from pysml.nn.autograd import Conv1dFunction
        
        output = Conv1dFunction.apply(
            Conv1dFunction,
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding
        )
        
        return output


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        
        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = tuple(stride)
        
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = tuple(padding)
    
    def forward(self, x: Tensor) -> Tensor:
        from pysml.nn.autograd import MaxPool2dFunction
        
        return MaxPool2dFunction.apply(
            MaxPool2dFunction,
            x,
            self.kernel_size,
            self.stride,
            self.padding
        )


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        
        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = tuple(stride)
        
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = tuple(padding)
    
    def forward(self, x: Tensor) -> Tensor:
        from pysml.nn.autograd import AvgPool2dFunction
        
        return AvgPool2dFunction.apply(
            AvgPool2dFunction,
            x,
            self.kernel_size,
            self.stride,
            self.padding
        )


class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
    
    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        # Flatten everything after batch dimension
        return x.view(batch_size, -1)


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability must be in [0, 1], got {p}")
        self.p = p
    
    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        
        # Get backend-aware numpy
        np_backend = x._get_backend_module()
        
        # Create dropout mask
        mask = np_backend.random.random(x.shape) > self.p
        
        # Scale by 1/(1-p) to maintain expected value
        scale = 1.0 / (1.0 - self.p)
        
        return Tensor(x.data * mask * scale, requires_grad=x.requires_grad, _ctx=x._ctx)


class BatchNorm2d(Module):
    def __init__(self, num_features: int, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)
        
        # Running statistics (not trainable)
        self.running_mean = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=False)
        self.running_var = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=False)
    
    def forward(self, x: Tensor) -> Tensor:
        np_backend = x._get_backend_module()
        
        if self.training:
            # Calculate batch statistics
            # x shape: (N, C, H, W)
            mean = np_backend.mean(x.data, axis=(0, 2, 3), keepdims=True)
            var = np_backend.var(x.data, axis=(0, 2, 3), keepdims=True)
            
            # Update running statistics
            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * mean.squeeze()
            self.running_var.data = (1 - self.momentum) * self.running_var.data + self.momentum * var.squeeze()
        else:
            # Use running statistics
            mean = self.running_mean.data.reshape(1, -1, 1, 1)
            var = self.running_var.data.reshape(1, -1, 1, 1)
        
        # Normalize
        x_norm = (x.data - mean) / np_backend.sqrt(var + self.eps)
        
        # Scale and shift
        gamma = self.gamma.data.reshape(1, -1, 1, 1)
        beta = self.beta.data.reshape(1, -1, 1, 1)
        
        out = gamma * x_norm + beta
        
        return Tensor(out, requires_grad=x.requires_grad)


# Helper function for calculating output size
def _calculate_output_size(input_size, kernel_size, stride, padding):
    return (input_size + 2 * padding - kernel_size) // stride + 1