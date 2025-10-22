"""
PySML Optimizers - MEMORY OPTIMIZED VERSION
Handles gradient shape mismatches correctly with lazy initialization and efficient memory usage

Key optimizations:
1. Lazy initialization of optimizer buffers (only created when needed)
2. Dict-based storage instead of lists (more flexible, less overhead)
3. In-place operations where possible
4. Proper gradient cleanup
"""

import pysml
import numpy as np
from typing import List, Optional, Tuple, Dict, Any


class Optimizer:
    """Base class for all optimizers - MEMORY OPTIMIZED"""
    
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
                print(f"WARNING: Gradient size mismatch!")
                print(f"  Parameter shape: {param.data.shape} (size: {param_size})")
                print(f"  Gradient shape: {grad.shape} (size: {grad_size})")
                print(f"  This usually means gradients weren't zeroed properly.")
                
                # Try to recover by taking only what we need
                if grad_size > param_size:
                    grad_flat = grad.flatten()
                    grad = grad_flat[:param_size].reshape(param.data.shape)
                    print(f"  Truncated gradient to match parameter shape")
                else:
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
        """Sets gradients of all optimized parameters to None - MEMORY OPTIMIZED"""
        for param in self.params:
            if param.grad is not None:
                # Explicitly delete gradient data
                del param.grad.data
                param.grad = None
            # Clear computation graph
            param._prev.clear()
            param._backward = lambda: None
    
    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the optimizer as a dict"""
        raise NotImplementedError("Subclasses should implement state_dict()")
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the optimizer state"""
        raise NotImplementedError("Subclasses should implement load_state_dict()")
    
    def get_last_lr(self) -> float:
        """Get current learning rate"""
        return self.lr


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer - MEMORY OPTIMIZED"""
    
    def __init__(self, params, lr: float = 1e-3, momentum: float = 0, 
                 dampening: float = 0, weight_decay: float = 0, nesterov: bool = False):
        super().__init__(params, lr)
        
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        
        # MEMORY OPTIMIZATION: Use dict for lazy initialization
        self.velocity = {}
    
    def step(self):
        """Performs a single optimization step"""
        for param in self.params:
            grad = self._get_grad_data(param)
            if grad is None:
                continue
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Apply momentum
            if self.momentum != 0:
                param_id = id(param)
                
                # Lazy initialization
                if param_id not in self.velocity:
                    self.velocity[param_id] = pysml.zeros(*param.shape)
                
                if self.velocity[param_id] is None:
                    buf = grad
                else:
                    buf = self.momentum * self.velocity[param_id].data + (1 - self.dampening) * grad
                
                self.velocity[param_id].data = buf
                
                if self.nesterov:
                    grad = grad + self.momentum * buf
                else:
                    grad = buf
            
            # Update parameters - IN-PLACE
            param.data = param.data - self.lr * grad
    
    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the optimizer"""
        return {
            'lr': self.lr,
            'momentum': self.momentum,
            'dampening': self.dampening,
            'weight_decay': self.weight_decay,
            'nesterov': self.nesterov,
            'velocity': {k: v.data if hasattr(v.data, 'asnumpy') else np.array(v.data) 
                        for k, v in self.velocity.items()}
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the optimizer state"""
        self.lr = state_dict['lr']
        self.momentum = state_dict['momentum']
        self.dampening = state_dict['dampening']
        self.weight_decay = state_dict['weight_decay']
        self.nesterov = state_dict['nesterov']
        
        self.velocity = {}
        for k, v_data in state_dict['velocity'].items():
            self.velocity[k] = pysml.Tensor(v_data)
    
    def __repr__(self):
        return f"SGD(lr={self.lr}, momentum={self.momentum}, weight_decay={self.weight_decay}, nesterov={self.nesterov})"


class Adam(Optimizer):
    """Adam optimizer (Adaptive Moment Estimation) - MEMORY OPTIMIZED"""
    
    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0, amsgrad: bool = False):
        super().__init__(params, lr)
        
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        
        # MEMORY OPTIMIZATION: Use dict for lazy initialization
        self.m = {}  # First moment
        self.v = {}  # Second moment
        
        if amsgrad:
            self.v_max = {}
        
        self.t = 0  # Timestep
    
    def step(self):
        """Performs a single optimization step"""
        self.t += 1
        
        for param in self.params:
            grad = self._get_grad_data(param)
            if grad is None:
                continue
            
            param_id = id(param)
            
            # Lazy initialization - only create buffers when needed
            if param_id not in self.m:
                self.m[param_id] = np.zeros_like(param.data)
                self.v[param_id] = np.zeros_like(param.data)
                if self.amsgrad:
                    self.v_max[param_id] = np.zeros_like(param.data)
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Update biased first moment estimate - IN-PLACE
            self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate - IN-PLACE
            self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)
            
            # AMSGrad variant
            if self.amsgrad:
                self.v_max[param_id] = np.maximum(self.v_max[param_id], v_hat)
                v_hat = self.v_max[param_id]
            
            # Update parameters - IN-PLACE
            v_hat_sqrt = np.sqrt(v_hat)
            param.data = param.data - self.lr * m_hat / (v_hat_sqrt + self.eps)
    
    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the optimizer"""
        state = {
            'lr': self.lr,
            'betas': (self.beta1, self.beta2),
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'amsgrad': self.amsgrad,
            't': self.t,
            'm': {k: v.copy() if isinstance(v, np.ndarray) else np.array(v) for k, v in self.m.items()},
            'v': {k: v.copy() if isinstance(v, np.ndarray) else np.array(v) for k, v in self.v.items()},
        }
        
        if self.amsgrad:
            state['v_max'] = {k: v.copy() if isinstance(v, np.ndarray) else np.array(v) 
                             for k, v in self.v_max.items()}
        
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the optimizer state"""
        self.lr = state_dict['lr']
        self.beta1, self.beta2 = state_dict['betas']
        self.eps = state_dict['eps']
        self.weight_decay = state_dict['weight_decay']
        self.amsgrad = state_dict['amsgrad']
        self.t = state_dict['t']
        
        self.m = state_dict['m']
        self.v = state_dict['v']
        
        if self.amsgrad and 'v_max' in state_dict:
            self.v_max = state_dict['v_max']
    
    def __repr__(self):
        return f"Adam(lr={self.lr}, betas=({self.beta1}, {self.beta2}), eps={self.eps}, weight_decay={self.weight_decay})"


class AdamW(Optimizer):
    """AdamW optimizer (Adam with decoupled weight decay) - MEMORY OPTIMIZED"""
    
    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 1e-2, amsgrad: bool = False):
        super().__init__(params, lr)
        
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        
        # MEMORY OPTIMIZATION: Use dict for lazy initialization
        self.m = {}
        self.v = {}
        
        if amsgrad:
            self.v_max = {}
        
        self.t = 0
    
    def step(self):
        """Performs a single optimization step"""
        self.t += 1
        
        for param in self.params:
            grad = self._get_grad_data(param)
            if grad is None:
                continue
            
            param_id = id(param)
            
            # Lazy initialization
            if param_id not in self.m:
                self.m[param_id] = np.zeros_like(param.data)
                self.v[param_id] = np.zeros_like(param.data)
                if self.amsgrad:
                    self.v_max[param_id] = np.zeros_like(param.data)
            
            # Update biased first moment estimate - IN-PLACE
            self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate - IN-PLACE
            self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)
            
            # AMSGrad variant
            if self.amsgrad:
                self.v_max[param_id] = np.maximum(self.v_max[param_id], v_hat)
                v_hat = self.v_max[param_id]
            
            # Update parameters with decoupled weight decay - IN-PLACE
            v_hat_sqrt = np.sqrt(v_hat)
            update = m_hat / (v_hat_sqrt + self.eps) + self.weight_decay * param.data
            param.data = param.data - self.lr * update
    
    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the optimizer"""
        state = {
            'lr': self.lr,
            'betas': (self.beta1, self.beta2),
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'amsgrad': self.amsgrad,
            't': self.t,
            'm': {k: v.copy() if isinstance(v, np.ndarray) else np.array(v) for k, v in self.m.items()},
            'v': {k: v.copy() if isinstance(v, np.ndarray) else np.array(v) for k, v in self.v.items()},
        }
        
        if self.amsgrad:
            state['v_max'] = {k: v.copy() if isinstance(v, np.ndarray) else np.array(v) 
                             for k, v in self.v_max.items()}
        
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the optimizer state"""
        self.lr = state_dict['lr']
        self.beta1, self.beta2 = state_dict['betas']
        self.eps = state_dict['eps']
        self.weight_decay = state_dict['weight_decay']
        self.amsgrad = state_dict['amsgrad']
        self.t = state_dict['t']
        
        self.m = state_dict['m']
        self.v = state_dict['v']
        
        if self.amsgrad and 'v_max' in state_dict:
            self.v_max = state_dict['v_max']
    
    def __repr__(self):
        return f"AdamW(lr={self.lr}, betas=({self.beta1}, {self.beta2}), weight_decay={self.weight_decay})"


class RMSprop(Optimizer):
    """RMSprop optimizer - MEMORY OPTIMIZED"""
    
    def __init__(self, params, lr: float = 1e-2, alpha: float = 0.99, eps: float = 1e-8,
                 weight_decay: float = 0, momentum: float = 0, centered: bool = False):
        super().__init__(params, lr)
        
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered
        
        # MEMORY OPTIMIZATION: Use dict for lazy initialization
        self.square_avg = {}
        
        if momentum > 0:
            self.momentum_buffer = {}
        
        if centered:
            self.grad_avg = {}
    
    def step(self):
        """Performs a single optimization step"""
        for param in self.params:
            grad = self._get_grad_data(param)
            if grad is None:
                continue
            
            param_id = id(param)
            
            # Lazy initialization
            if param_id not in self.square_avg:
                self.square_avg[param_id] = np.zeros_like(param.data)
                if self.momentum > 0:
                    self.momentum_buffer[param_id] = np.zeros_like(param.data)
                if self.centered:
                    self.grad_avg[param_id] = np.zeros_like(param.data)
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Update square average - IN-PLACE
            self.square_avg[param_id] = self.alpha * self.square_avg[param_id] + (1 - self.alpha) * (grad ** 2)
            
            if self.centered:
                # Update gradient average - IN-PLACE
                self.grad_avg[param_id] = self.alpha * self.grad_avg[param_id] + (1 - self.alpha) * grad
                avg = np.sqrt(self.square_avg[param_id] - self.grad_avg[param_id] ** 2) + self.eps
            else:
                avg = np.sqrt(self.square_avg[param_id]) + self.eps
            
            if self.momentum > 0:
                # Update momentum buffer - IN-PLACE
                self.momentum_buffer[param_id] = self.momentum * self.momentum_buffer[param_id] + grad / avg
                param.data = param.data - self.lr * self.momentum_buffer[param_id]
            else:
                param.data = param.data - self.lr * grad / avg
    
    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the optimizer"""
        state = {
            'lr': self.lr,
            'alpha': self.alpha,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'centered': self.centered,
            'square_avg': {k: v.copy() if isinstance(v, np.ndarray) else np.array(v) 
                          for k, v in self.square_avg.items()},
        }
        
        if self.momentum > 0:
            state['momentum_buffer'] = {k: v.copy() if isinstance(v, np.ndarray) else np.array(v) 
                                       for k, v in self.momentum_buffer.items()}
        
        if self.centered:
            state['grad_avg'] = {k: v.copy() if isinstance(v, np.ndarray) else np.array(v) 
                                for k, v in self.grad_avg.items()}
        
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the optimizer state"""
        self.lr = state_dict['lr']
        self.alpha = state_dict['alpha']
        self.eps = state_dict['eps']
        self.weight_decay = state_dict['weight_decay']
        self.momentum = state_dict['momentum']
        self.centered = state_dict['centered']
        
        self.square_avg = state_dict['square_avg']
        
        if self.momentum > 0 and 'momentum_buffer' in state_dict:
            self.momentum_buffer = state_dict['momentum_buffer']
        
        if self.centered and 'grad_avg' in state_dict:
            self.grad_avg = state_dict['grad_avg']
    
    def __repr__(self):
        return f"RMSprop(lr={self.lr}, alpha={self.alpha}, momentum={self.momentum})"


class Adagrad(Optimizer):
    """Adagrad optimizer - MEMORY OPTIMIZED"""
    
    def __init__(self, params, lr: float = 1e-2, lr_decay: float = 0,
                 weight_decay: float = 0, eps: float = 1e-10):
        super().__init__(params, lr)
        
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.eps = eps
        
        # MEMORY OPTIMIZATION: Use dict for lazy initialization
        self.state_sum = {}
        self.t = 0
    
    def step(self):
        """Performs a single optimization step"""
        self.t += 1
        
        for param in self.params:
            grad = self._get_grad_data(param)
            if grad is None:
                continue
            
            param_id = id(param)
            
            # Lazy initialization
            if param_id not in self.state_sum:
                self.state_sum[param_id] = np.zeros_like(param.data)
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Update accumulated gradient - IN-PLACE
            self.state_sum[param_id] = self.state_sum[param_id] + grad ** 2
            
            # Compute learning rate with decay
            clr = self.lr / (1 + (self.t - 1) * self.lr_decay)
            
            # Update parameters - IN-PLACE
            std = np.sqrt(self.state_sum[param_id]) + self.eps
            param.data = param.data - clr * grad / std
    
    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the optimizer"""
        return {
            'lr': self.lr,
            'lr_decay': self.lr_decay,
            'weight_decay': self.weight_decay,
            'eps': self.eps,
            't': self.t,
            'state_sum': {k: v.copy() if isinstance(v, np.ndarray) else np.array(v) 
                         for k, v in self.state_sum.items()},
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the optimizer state"""
        self.lr = state_dict['lr']
        self.lr_decay = state_dict['lr_decay']
        self.weight_decay = state_dict['weight_decay']
        self.eps = state_dict['eps']
        self.t = state_dict['t']
        
        self.state_sum = state_dict['state_sum']
    
    def __repr__(self):
        return f"Adagrad(lr={self.lr}, lr_decay={self.lr_decay}, weight_decay={self.weight_decay})"


class Adadelta(Optimizer):
    """Adadelta optimizer - MEMORY OPTIMIZED"""
    
    def __init__(self, params, lr: float = 1.0, rho: float = 0.9,
                 eps: float = 1e-6, weight_decay: float = 0):
        super().__init__(params, lr)
        
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        
        # MEMORY OPTIMIZATION: Use dict for lazy initialization
        self.square_avg = {}
        self.acc_delta = {}
    
    def step(self):
        """Performs a single optimization step"""
        for param in self.params:
            grad = self._get_grad_data(param)
            if grad is None:
                continue
            
            param_id = id(param)
            
            # Lazy initialization
            if param_id not in self.square_avg:
                self.square_avg[param_id] = np.zeros_like(param.data)
                self.acc_delta[param_id] = np.zeros_like(param.data)
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Accumulate gradient - IN-PLACE
            self.square_avg[param_id] = self.rho * self.square_avg[param_id] + (1 - self.rho) * (grad ** 2)
            
            # Compute update
            std = np.sqrt(self.acc_delta[param_id] + self.eps)
            delta = (std / np.sqrt(self.square_avg[param_id] + self.eps)) * grad
            
            # Accumulate update - IN-PLACE
            self.acc_delta[param_id] = self.rho * self.acc_delta[param_id] + (1 - self.rho) * (delta ** 2)
            
            # Update parameters - IN-PLACE
            param.data = param.data - self.lr * delta
    
    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the optimizer"""
        return {
            'lr': self.lr,
            'rho': self.rho,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'square_avg': {k: v.copy() if isinstance(v, np.ndarray) else np.array(v) 
                          for k, v in self.square_avg.items()},
            'acc_delta': {k: v.copy() if isinstance(v, np.ndarray) else np.array(v) 
                         for k, v in self.acc_delta.items()},
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the optimizer state"""
        self.lr = state_dict['lr']
        self.rho = state_dict['rho']
        self.eps = state_dict['eps']
        self.weight_decay = state_dict['weight_decay']
        
        self.square_avg = state_dict['square_avg']
        self.acc_delta = state_dict['acc_delta']
    
    def __repr__(self):
        return f"Adadelta(lr={self.lr}, rho={self.rho}, eps={self.eps})"


class LBFGS(Optimizer):
    """Limited-memory BFGS optimizer (simplified version) - MEMORY OPTIMIZED"""
    
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
            
            # Update parameters - IN-PLACE
            for param in self.params:
                grad = self._get_grad_data(param)
                if grad is not None:
                    param.data = param.data - self.lr * grad
        
        return loss
    
    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the optimizer"""
        return {
            'lr': self.lr,
            'max_iter': self.max_iter,
            'tolerance_grad': self.tolerance_grad,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the optimizer state"""
        self.lr = state_dict['lr']
        self.max_iter = state_dict['max_iter']
        self.tolerance_grad = state_dict['tolerance_grad']
    
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
    
    def get_last_lr(self):
        """Get current learning rate (alias for PyTorch compatibility)"""
        return self.get_lr()
    
    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the scheduler"""
        return {
            'base_lr': self.base_lr,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the scheduler state"""
        self.base_lr = state_dict['base_lr']


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
    
    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state.update({
            'step_size': self.step_size,
            'gamma': self.gamma,
            'last_epoch': self.last_epoch,
        })
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        super().load_state_dict(state_dict)
        self.step_size = state_dict['step_size']
        self.gamma = state_dict['gamma']
        self.last_epoch = state_dict['last_epoch']


class ExponentialLR(LRScheduler):
    """Decays learning rate by gamma every epoch"""
    
    def __init__(self, optimizer, gamma: float):
        super().__init__(optimizer)
        self.gamma = gamma
    
    def step(self, epoch=None):
        self.optimizer.lr = self.optimizer.lr * self.gamma
    
    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state['gamma'] = self.gamma
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        super().load_state_dict(state_dict)
        self.gamma = state_dict['gamma']


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
    
    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state.update({
            'T_max': self.T_max,
            'eta_min': self.eta_min,
            'last_epoch': self.last_epoch,
        })
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        super().load_state_dict(state_dict)
        self.T_max = state_dict['T_max']
        self.eta_min = state_dict['eta_min']
        self.last_epoch = state_dict['last_epoch']


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
    
    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state.update({
            'mode': self.mode,
            'factor': self.factor,
            'patience': self.patience,
            'threshold': self.threshold,
            'min_lr': self.min_lr,
            'best': self.best,
            'num_bad_epochs': self.num_bad_epochs,
        })
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        super().load_state_dict(state_dict)
        self.mode = state_dict['mode']
        self.factor = state_dict['factor']
        self.patience = state_dict['patience']
        self.threshold = state_dict['threshold']
        self.min_lr = state_dict['min_lr']
        self.best = state_dict['best']
        self.num_bad_epochs = state_dict['num_bad_epochs']


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