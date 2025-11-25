from .module import Module


class Dropout(Module):

    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability must be in [0, 1], got {p}")
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        # Only apply dropout during training
        if not self.training or self.p == 0:
            return x

        from .. import engine
        backend = x._backend

        rand = backend.rand(x.shape, device=x.device)
        mask = backend.greater(rand, self.p)
        mask = backend.astype(mask, x.data.dtype)
        mask = backend.divide(mask, 1.0 - self.p)

        from .. import Tensor

        mask_tensor = Tensor.__new__(Tensor)
        mask_tensor._backend = backend
        mask_tensor._dtype = x._dtype
        mask_tensor.device = x.device
        mask_tensor.active_device = x.active_device
        mask_tensor.data = mask
        mask_tensor._requires_grad = False
        mask_tensor._grad = None

        return engine.multiply(x, mask_tensor)

    def extra_repr(self):
        return f"p={self.p}" + (", inplace=True" if self.inplace else "")


class Dropout1d(Module):
    
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability must be in [0, 1], got {p}")
        self.p = p
        self.inplace = inplace
    
    def forward(self, x):
        # x: (batch, channels, length)
        # Drop entire channels

        if not self.training or self.p == 0:
            return x

        from .. import engine
        backend = x._backend

        # Create channel-wise mask
        # Shape: (batch, channels, 1)
        batch_size, num_channels = x.shape[0], x.shape[1]
        rand = backend.rand((batch_size, num_channels, 1), device=x.device)
        mask = backend.greater(rand, self.p)
        mask = backend.astype(mask, x.data.dtype)

        # Scale by keep probability
        mask = backend.divide(mask, 1.0 - self.p)

        # Broadcast mask to full shape
        mask_full = backend.broadcast_to(mask, x.shape)

        from .. import Tensor
        mask_tensor = Tensor.__new__(Tensor)
        mask_tensor._backend = backend
        mask_tensor._dtype = x._dtype
        mask_tensor.device = x.device
        mask_tensor.active_device = x.active_device
        mask_tensor.data = mask_full
        mask_tensor._requires_grad = False
        mask_tensor._grad = None

        return engine.multiply(x, mask_tensor)
    
    def extra_repr(self):
        return f"p={self.p}" + (", inplace=True" if self.inplace else "")


class Dropout2d(Module):
    
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability must be in [0, 1], got {p}")
        self.p = p
        self.inplace = inplace
    
    def forward(self, x):
        # x: (batch, channels, height, width)
        # Drop entire channels (feature maps)

        if not self.training or self.p == 0:
            return x

        from .. import engine
        backend = x._backend

        # Create channel-wise mask
        # Shape: (batch, channels, 1, 1)
        batch_size, num_channels = x.shape[0], x.shape[1]

        rand = backend.rand((batch_size, num_channels, 1, 1), device=x.device)
        mask = backend.greater(rand, self.p)
        mask = backend.astype(mask, x.data.dtype)

        # Scale by keep probability
        mask = backend.divide(mask, 1.0 - self.p)

        # Broadcast mask to full shape
        mask_full = backend.broadcast_to(mask, x.shape)

        from .. import Tensor
        mask_tensor = Tensor.__new__(Tensor)
        mask_tensor._backend = backend
        mask_tensor._dtype = x._dtype
        mask_tensor.device = x.device
        mask_tensor.active_device = x.active_device
        mask_tensor.data = mask_full
        mask_tensor._requires_grad = False
        mask_tensor._grad = None

        return engine.multiply(x, mask_tensor)
    
    def extra_repr(self):
        return f"p={self.p}" + (", inplace=True" if self.inplace else "")


class Dropout3d(Module):
    
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability must be in [0, 1], got {p}")
        self.p = p
        self.inplace = inplace
    
    def forward(self, x):
        # x: (batch, channels, depth, height, width)
        # Drop entire channels
        
        if not self.training or self.p == 0:
            return x

        from .. import engine
        backend = x._backend

        # Create channel-wise mask
        # Shape: (batch, channels, 1, 1, 1)
        batch_size, num_channels = x.shape[0], x.shape[1]

        rand = backend.rand((batch_size, num_channels, 1, 1, 1), device=x.device)
        mask = backend.greater(rand, self.p)
        mask = backend.astype(mask, x.data.dtype)

        # Scale by keep probability
        mask = backend.divide(mask, 1.0 - self.p)

        # Broadcast mask to full shape
        mask_full = backend.broadcast_to(mask, x.shape)

        from .. import Tensor
        mask_tensor = Tensor.__new__(Tensor)
        mask_tensor._backend = backend
        mask_tensor._dtype = x._dtype
        mask_tensor.device = x.device
        mask_tensor.active_device = x.active_device
        mask_tensor.data = mask_full
        mask_tensor._requires_grad = False
        mask_tensor._grad = None

        return engine.multiply(x, mask_tensor)
    
    def extra_repr(self):
        return f"p={self.p}" + (", inplace=True" if self.inplace else "")


class AlphaDropout(Module):
    
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"AlphaDropout probability must be in [0, 1], got {p}")
        self.p = p
        self.inplace = inplace
        
        # SELU constants for self-normalization
        # After alpha dropout, mean and variance are preserved
        self.alpha = -1.7580993408473766
        self.fixedPointMean = 0.0
        self.fixedPointVar = 1.0
    
    def forward(self, x):
        # Alpha dropout maintains the self-normalizing property of SELU

        if not self.training or self.p == 0:
            return x

        from .. import engine
        backend = x._backend

        # Create binary mask using backend RNG for determinism
        mask = backend.greater(backend.rand(x.shape, device=x.device), self.p)

        # Alpha dropout transformation
        # Sets dropped values to alpha (not zero)
        keep_prob = 1.0 - self.p

        # Compute affine transformation parameters
        a = ((1 - keep_prob) * (1 + keep_prob * self.alpha ** 2)) ** -0.5
        b = -a * self.alpha * keep_prob

        # Apply mask with alpha dropout
        masked = backend.where(mask, x.data, self.alpha)
        output = backend.add(backend.multiply(masked, a), b)

        from .. import Tensor
        result = Tensor.__new__(Tensor)
        result._backend = backend
        result._dtype = x._dtype
        result.device = x.device
        result.active_device = x.active_device
        result.data = output
        result._requires_grad = x._requires_grad
        result._grad = None

        return result
    
    def extra_repr(self):
        return f"p={self.p}" + (", inplace=True" if self.inplace else "")


class FeatureAlphaDropout(Module):
    
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"FeatureAlphaDropout probability must be in [0, 1], got {p}")
        self.p = p
        self.inplace = inplace
        
        # SELU constants
        self.alpha = -1.7580993408473766
    
    def forward(self, x):
        # Feature-wise alpha dropout (drops entire features/channels)

        if not self.training or self.p == 0:
            return x

        from .. import engine
        backend = x._backend

        # x: (batch, features, ...)
        batch_size, num_features = x.shape[0], x.shape[1]

        # Create feature-wise mask
        mask = backend.greater(
            backend.rand((batch_size, num_features), device=x.device), self.p
        )

        # Reshape mask to broadcast
        mask_shape = [batch_size, num_features] + [1] * (len(x.shape) - 2)
        mask = backend.reshape(mask, mask_shape)

        # Alpha dropout parameters
        keep_prob = 1.0 - self.p
        a = ((1 - keep_prob) * (1 + keep_prob * self.alpha ** 2)) ** -0.5
        b = -a * self.alpha * keep_prob

        # Apply transformation
        masked = backend.where(mask, x.data, self.alpha)
        output = backend.add(backend.multiply(masked, a), b)

        from .. import Tensor
        result = Tensor.__new__(Tensor)
        result._backend = backend
        result._dtype = x._dtype
        result.device = x.device
        result.active_device = x.active_device
        result.data = output
        result._requires_grad = x._requires_grad
        result._grad = None

        return result
    
    def extra_repr(self):
        return f"p={self.p}" + (", inplace=True" if self.inplace else "")


__all__ = [
    'Dropout',
    'Dropout1d',
        'Dropout2d',
        'Dropout3d',
        'AlphaDropout',
        'FeatureAlphaDropout',
]
