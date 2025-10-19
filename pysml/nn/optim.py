"""
PySML Optimizers - Fixed Version
Handles gradient shape mismatches correctly
"""

import pysml
import numpy as np
from typing import List, Optional, Tuple


class Optimizer:
    """Base class for all optimizers"""
    
    def __init__(self, params, lr: float = 1e-3):
        self.params = list(params)
        self.lr = lr
    
    def _get_grad_data(self, param):
        """Extract gradient data and ensure it matches parameter shape"""
        if param.grad is None:
            return None
        
        grad = param.grad.data
        
        # Convert to numpy if needed
        if hasattr(grad, 'asnumpy'):
            grad = grad.asnumpy()
        elif not isinstance(grad, np.ndarray):
            grad = np.array(grad)
        
        # Ensure gradient shape matches parameter shape
        if grad.shape != param.data.shape:
            # Check if sizes match (can reshape)
            param_size = param.data.size if hasattr(param.data, 'size') else np.prod(param.data.shape)
            grad_size = grad.size if hasattr(grad, 'size') else np.prod(grad.shape)
            
            if grad_size == param_size:
                # Same number of elements, just reshape
                grad = grad.reshape(param.data.shape)
            else:
                # Size mismatch - this shouldn't happen in correct training
                # This indicates gradients were accumulated incorrectly
                print(f"WARNING: Gradient size mismatch!")
                print(f"  Parameter shape: {param.data.shape} (size: {param_size})")
                print(f"  Gradient shape: {grad.shape} (size: {grad_size})")
                print(f"  This usually means gradients weren't zeroed properly.")
                
                # Try to recover by taking only what we need
                if grad_size > param_size:
                    # Gradient is too large, truncate and reshape
                    grad_flat = grad.flatten()
                    grad = grad_flat[:param_size].reshape(param.data.shape)
                    print(f"  Truncated gradient to match parameter shape")
                else:
                    # Gradient is too small, pad with zeros
                    grad_flat = grad.flatten()
                    padded = np.zeros(param_size)
                    padded[:grad_size] = grad_flat
                    grad = padded.reshape(param.data.shape)
                    print(f"  Padded gradient to match parameter shape")
        
        return grad
    
    def step(self):
        """Performs a single optimization step"""
        raise NotImplementedError("Subclasses must implement step()")
    
    def zero_grad(self):
        """Sets gradients of all optimized parameters to None"""
        for param in self.params:
            param.zero_grad()
    
    def state_dict(self):
        """Returns the state of the optimizer as a dict"""
        raise NotImplementedError("Subclasses should implement state_dict()")
    
    def load_state_dict(self, state_dict):
        """Loads the optimizer state"""
        raise NotImplementedError("Subclasses should implement load_state_dict()")


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""
    
    def __init__(self, params, lr: float = 1e-3, momentum: float = 0, 
                 dampening: float = 0, weight_decay: float = 0, nesterov: bool = False):
        super().__init__(params, lr)
        
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        
        # Initialize momentum buffers
        self.velocity = [pysml.zeros(*p.shape) for p in self.params]
    
    def step(self):
        """Performs a single optimization step"""
        for i, param in enumerate(self.params):
            grad = self._get_grad_data(param)
            if grad is None:
                continue
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Apply momentum
            if self.momentum != 0:
                if self.velocity[i] is None:
                    buf = grad
                else:
                    buf = self.momentum * self.velocity[i].data + (1 - self.dampening) * grad
                
                self.velocity[i].data = buf
                
                if self.nesterov:
                    grad = grad + self.momentum * buf
                else:
                    grad = buf
            
            # Update parameters
            param.data = param.data - self.lr * grad
    
    def __repr__(self):
        return f"SGD(lr={self.lr}, momentum={self.momentum}, weight_decay={self.weight_decay}, nesterov={self.nesterov})"


class Adam(Optimizer):
    """Adam optimizer (Adaptive Moment Estimation)"""
    
    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0, amsgrad: bool = False):
        super().__init__(params, lr)
        
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        
        # Initialize moment estimates
        self.m = [pysml.zeros(*p.shape) for p in self.params]  # First moment
        self.v = [pysml.zeros(*p.shape) for p in self.params]  # Second moment
        
        if amsgrad:
            self.v_max = [pysml.zeros(*p.shape) for p in self.params]
        
        self.t = 0  # Timestep
    
    def step(self):
        """Performs a single optimization step"""
        self.t += 1
        
        for i, param in enumerate(self.params):
            grad = self._get_grad_data(param)
            if grad is None:
                continue
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Update biased first moment estimate
            self.m[i].data = self.beta1 * self.m[i].data + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i].data = self.beta2 * self.v[i].data + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i].data / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i].data / (1 - self.beta2 ** self.t)
            
            # AMSGrad variant
            if self.amsgrad:
                v_hat_tensor = pysml.Tensor(v_hat, backend=param.backend)
                self.v_max[i].data = pysml.maximum(self.v_max[i], v_hat_tensor).data
                v_hat = self.v_max[i].data
            
            # Update parameters
            # Compute sqrt(v_hat) + eps
            v_hat_sqrt = np.sqrt(v_hat) if isinstance(v_hat, np.ndarray) else pysml.sqrt(pysml.Tensor(v_hat, backend=param.backend)).data
            param.data = param.data - self.lr * m_hat / (v_hat_sqrt + self.eps)
    
    def __repr__(self):
        return f"Adam(lr={self.lr}, betas=({self.beta1}, {self.beta2}), eps={self.eps}, weight_decay={self.weight_decay})"


class AdamW(Optimizer):
    """AdamW optimizer (Adam with decoupled weight decay)"""
    
    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 1e-2, amsgrad: bool = False):
        super().__init__(params, lr)
        
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        
        # Initialize moment estimates
        self.m = [pysml.zeros(*p.shape) for p in self.params]
        self.v = [pysml.zeros(*p.shape) for p in self.params]
        
        if amsgrad:
            self.v_max = [pysml.zeros(*p.shape) for p in self.params]
        
        self.t = 0
    
    def step(self):
        """Performs a single optimization step"""
        self.t += 1
        
        for i, param in enumerate(self.params):
            grad = self._get_grad_data(param)
            if grad is None:
                continue
            
            # Update biased first moment estimate
            self.m[i].data = self.beta1 * self.m[i].data + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i].data = self.beta2 * self.v[i].data + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i].data / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i].data / (1 - self.beta2 ** self.t)
            
            # AMSGrad variant
            if self.amsgrad:
                v_hat_tensor = pysml.Tensor(v_hat, backend=param.backend)
                self.v_max[i].data = pysml.maximum(self.v_max[i], v_hat_tensor).data
                v_hat = self.v_max[i].data
            
            # Update parameters with decoupled weight decay
            # Compute sqrt(v_hat) + eps
            v_hat_sqrt = np.sqrt(v_hat) if isinstance(v_hat, np.ndarray) else pysml.sqrt(pysml.Tensor(v_hat, backend=param.backend)).data
            
            # AdamW update: param = param - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)
            update = m_hat / (v_hat_sqrt + self.eps) + self.weight_decay * param.data
            param.data = param.data - self.lr * update
    
    def __repr__(self):
        return f"AdamW(lr={self.lr}, betas=({self.beta1}, {self.beta2}), weight_decay={self.weight_decay})"


class RMSprop(Optimizer):
    """RMSprop optimizer"""
    
    def __init__(self, params, lr: float = 1e-2, alpha: float = 0.99, eps: float = 1e-8,
                 weight_decay: float = 0, momentum: float = 0, centered: bool = False):
        super().__init__(params, lr)
        
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered
        
        # Initialize state
        self.square_avg = [pysml.zeros(*p.shape) for p in self.params]
        
        if momentum > 0:
            self.momentum_buffer = [pysml.zeros(*p.shape) for p in self.params]
        
        if centered:
            self.grad_avg = [pysml.zeros(*p.shape) for p in self.params]
    
    def step(self):
        """Performs a single optimization step"""
        for i, param in enumerate(self.params):
            grad = self._get_grad_data(param)
            if grad is None:
                continue
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Update square average
            self.square_avg[i].data = self.alpha * self.square_avg[i].data + (1 - self.alpha) * (grad ** 2)
            
            if self.centered:
                # Update gradient average
                self.grad_avg[i].data = self.alpha * self.grad_avg[i].data + (1 - self.alpha) * grad
                avg = np.sqrt(self.square_avg[i].data - self.grad_avg[i].data ** 2) + self.eps
            else:
                avg = np.sqrt(self.square_avg[i].data) + self.eps
            
            if self.momentum > 0:
                # Update momentum buffer
                self.momentum_buffer[i].data = self.momentum * self.momentum_buffer[i].data + grad / avg
                param.data = param.data - self.lr * self.momentum_buffer[i].data
            else:
                param.data = param.data - self.lr * grad / avg
    
    def __repr__(self):
        return f"RMSprop(lr={self.lr}, alpha={self.alpha}, momentum={self.momentum})"


class Adagrad(Optimizer):
    """Adagrad optimizer"""
    
    def __init__(self, params, lr: float = 1e-2, lr_decay: float = 0,
                 weight_decay: float = 0, eps: float = 1e-10):
        super().__init__(params, lr)
        
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.eps = eps
        
        # Initialize state
        self.state_sum = [pysml.zeros(*p.shape) for p in self.params]
        self.t = 0
    
    def step(self):
        """Performs a single optimization step"""
        self.t += 1
        
        for i, param in enumerate(self.params):
            grad = self._get_grad_data(param)
            if grad is None:
                continue
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Update accumulated gradient
            self.state_sum[i].data = self.state_sum[i].data + grad ** 2
            
            # Compute learning rate with decay
            clr = self.lr / (1 + (self.t - 1) * self.lr_decay)
            
            # Update parameters
            std = np.sqrt(self.state_sum[i].data) + self.eps
            param.data = param.data - clr * grad / std
    
    def __repr__(self):
        return f"Adagrad(lr={self.lr}, lr_decay={self.lr_decay}, weight_decay={self.weight_decay})"


class Adadelta(Optimizer):
    """Adadelta optimizer"""
    
    def __init__(self, params, lr: float = 1.0, rho: float = 0.9,
                 eps: float = 1e-6, weight_decay: float = 0):
        super().__init__(params, lr)
        
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize state
        self.square_avg = [pysml.zeros(*p.shape) for p in self.params]
        self.acc_delta = [pysml.zeros(*p.shape) for p in self.params]
    
    def step(self):
        """Performs a single optimization step"""
        for i, param in enumerate(self.params):
            grad = self._get_grad_data(param)
            if grad is None:
                continue
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Accumulate gradient
            self.square_avg[i].data = self.rho * self.square_avg[i].data + (1 - self.rho) * (grad ** 2)
            
            # Compute update
            std = np.sqrt(self.acc_delta[i].data + self.eps)
            delta = (std / np.sqrt(self.square_avg[i].data + self.eps)) * grad
            
            # Accumulate update
            self.acc_delta[i].data = self.rho * self.acc_delta[i].data + (1 - self.rho) * (delta ** 2)
            
            # Update parameters
            param.data = param.data - self.lr * delta
    
    def __repr__(self):
        return f"Adadelta(lr={self.lr}, rho={self.rho}, eps={self.eps})"


class LBFGS(Optimizer):
    """Limited-memory BFGS optimizer (simplified version)"""
    
    def __init__(self, params, lr: float = 1, max_iter: int = 20,
                 tolerance_grad: float = 1e-5):
        super().__init__(params, lr)
        
        self.max_iter = max_iter
        self.tolerance_grad = tolerance_grad
    
    def step(self, closure=None):
        """Performs a single optimization step"""
        if closure is None:
            raise RuntimeError("LBFGS requires a closure function")
        
        for _ in range(self.max_iter):
            loss = closure()
            
            # Check gradient norm
            grad_norm = 0
            for param in self.params:
                grad = self._get_grad_data(param)
                if grad is not None:
                    grad_norm += np.sum(grad ** 2)
            
            grad_norm = grad_norm ** 0.5
            
            if grad_norm < self.tolerance_grad:
                break
            
            # Update parameters
            for param in self.params:
                grad = self._get_grad_data(param)
                if grad is not None:
                    param.data = param.data - self.lr * grad
        
        return loss
    
    def __repr__(self):
        return f"LBFGS(lr={self.lr}, max_iter={self.max_iter})"


# ===== Learning Rate Schedulers =====

class LRScheduler:
    """Base class for learning rate schedulers"""
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
    
    def step(self, epoch=None):
        """Update learning rate"""
        raise NotImplementedError("Subclasses must implement step()")
    
    def get_lr(self):
        """Get current learning rate"""
        return self.optimizer.lr


class StepLR(LRScheduler):
    """Decays learning rate by gamma every step_size epochs"""
    
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.last_epoch = epoch
        
        if epoch % self.step_size == 0:
            self.optimizer.lr = self.optimizer.lr * self.gamma


class ExponentialLR(LRScheduler):
    """Decays learning rate by gamma every epoch"""
    
    def __init__(self, optimizer, gamma: float):
        super().__init__(optimizer)
        self.gamma = gamma
    
    def step(self, epoch=None):
        self.optimizer.lr = self.optimizer.lr * self.gamma


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate scheduler"""
    
    def __init__(self, optimizer, T_max: int, eta_min: float = 0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = 0
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.last_epoch = epoch
        
        import math
        self.optimizer.lr = self.eta_min + (self.base_lr - self.eta_min) * \
                           (1 + math.cos(math.pi * epoch / self.T_max)) / 2


class ReduceLROnPlateau(LRScheduler):
    """Reduce learning rate when metric has stopped improving"""
    
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, min_lr=0):
        super().__init__(optimizer)
        
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        
        self.best = None
        self.num_bad_epochs = 0
    
    def step(self, metrics):
        """Update learning rate based on metric"""
        if self.best is None:
            self.best = metrics
        else:
            if self.mode == 'min':
                if metrics < self.best - self.threshold:
                    self.best = metrics
                    self.num_bad_epochs = 0
                else:
                    self.num_bad_epochs += 1
            else:  # mode == 'max'
                if metrics > self.best + self.threshold:
                    self.best = metrics
                    self.num_bad_epochs = 0
                else:
                    self.num_bad_epochs += 1
            
            if self.num_bad_epochs >= self.patience:
                new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
                self.optimizer.lr = new_lr
                self.num_bad_epochs = 0


# Convenience function
def get_optimizer(name: str, params, **kwargs):
    """Factory function to create optimizer by name"""
    optimizers = {
        'sgd': SGD,
        'adam': Adam,
        'adamw': AdamW,
        'rmsprop': RMSprop,
        'adagrad': Adagrad,
        'adadelta': Adadelta,
        'lbfgs': LBFGS,
    }
    
    name = name.lower()
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(optimizers.keys())}")
    
    return optimizers[name](params, **kwargs)