import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                # Gradients are Tensors, so we need to zero out their data
                p.grad.data.fill(0)


class SGD(Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad.data

class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
    
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            # Skip parameters that don't require gradients or have invalid data
            if not p.requires_grad or np.any(np.isnan(p.data)) or np.any(np.isinf(p.data)):
                continue
                
            grad = p.grad.data
            
            # Skip if gradient is all zeros or contains NaN/Inf
            if np.all(grad == 0) or np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
                continue
            
            # Add L2 regularization (not decoupled weight decay)
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Compute denominator with numerical stability
            denom = np.sqrt(v_hat) + self.eps
            
            # Check for valid denominator
            if np.any(denom == 0) or np.any(np.isnan(denom)):
                continue
            
            # Update parameters with gradient clipping
            update = self.lr * m_hat / denom
            update = np.clip(update, -10.0, 10.0)
            p.data -= update


class AdamW(Optimizer):
    """
    AdamW optimizer with decoupled weight decay.
    
    The key difference from Adam is that weight decay is applied directly to the parameters
    rather than being added to the gradients. This decoupling fixes issues with L2 regularization
    in Adam and often leads to better generalization.
    
    Reference: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
    https://arxiv.org/abs/1711.05101
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.001)
        betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
        amsgrad: Whether to use AMSGrad variant (default: False)
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.t = 0
        
        # First moment (mean) of gradients
        self.m = [np.zeros_like(p.data) for p in self.params]
        
        # Second moment (uncentered variance) of gradients
        self.v = [np.zeros_like(p.data) for p in self.params]
        
        # Maximum of second moment (for AMSGrad)
        if self.amsgrad:
            self.v_max = [np.zeros_like(p.data) for p in self.params]
    
    def step(self):
        self.t += 1
        
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            # Skip parameters that don't require gradients or have invalid data
            if not p.requires_grad or np.any(np.isnan(p.data)) or np.any(np.isinf(p.data)):
                continue
                
            grad = p.grad.data
            
            # Skip if gradient is all zeros or contains NaN/Inf
            if np.all(grad == 0) or np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
                continue
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            if self.amsgrad:
                # Use maximum of past squared gradients
                self.v_max[i] = np.maximum(self.v_max[i], self.v[i])
                v_hat = self.v_max[i] / (1 - self.beta2 ** self.t)
            else:
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Compute denominator with numerical stability
            denom = np.sqrt(v_hat) + self.eps
            
            # Check for valid denominator
            if np.any(denom == 0) or np.any(np.isnan(denom)):
                continue
            
            # Update parameters with Adam step
            update = self.lr * m_hat / denom
            
            # Clip update to prevent exploding gradients
            update = np.clip(update, -10.0, 10.0)
            
            p.data -= update
            
            # Apply decoupled weight decay
            # This is the key difference from Adam: weight decay is applied directly
            # to parameters, not added to gradients
            if self.weight_decay != 0:
                p.data -= self.lr * self.weight_decay * p.data


class AdamWScheduleFree(AdamW):
    """
    AdamW with schedule-free learning rate adaptation.
    
    This variant adapts the learning rate automatically without requiring
    a learning rate schedule. Useful for quick experimentation.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Initial learning rate (default: 0.001)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Numerical stability term (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
        lr_decay: Learning rate decay factor per step (default: 0.9999)
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0.01, lr_decay=0.9999):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad=False)
        self.initial_lr = lr
        self.lr_decay = lr_decay
    
    def step(self):
        # Decay learning rate
        self.lr = self.initial_lr * (self.lr_decay ** self.t)
        
        # Call parent step
        super().step()

