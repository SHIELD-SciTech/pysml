

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

from .optimizer import Optimizer
from ..tensor import Tensor

class Adam(Optimizer):

    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):

        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if isinstance(grad, Tensor):
                    grad_data = grad.data
                else:
                    grad_data = grad
                
                backend = p._backend
                param_id = id(p)
                
                # Initialize state
                if 'step' not in self.state[param_id]:
                    self.state[param_id]['step'] = 0
                    self.state[param_id]['exp_avg'] = backend.zeros(p.data.shape, dtype=p.data.dtype)
                    self.state[param_id]['exp_avg_sq'] = backend.zeros(p.data.shape, dtype=p.data.dtype)
                    if amsgrad:
                        self.state[param_id]['max_exp_avg_sq'] = backend.zeros(p.data.shape, dtype=p.data.dtype)
                
                state = self.state[param_id]
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                
                state['step'] += 1
                step = state['step']
                
                # Apply weight decay (L2 regularization for Adam)
                if weight_decay != 0:
                    grad_data = backend.add(grad_data, backend.multiply(p.data, weight_decay))
                
                # Update biased first moment estimate
                exp_avg = backend.add(
                    backend.multiply(exp_avg, beta1),
                    backend.multiply(grad_data, 1 - beta1)
                )
                state['exp_avg'] = exp_avg
                
                # Update biased second raw moment estimate
                exp_avg_sq = backend.add(
                    backend.multiply(exp_avg_sq, beta2),
                    backend.multiply(backend.multiply(grad_data, grad_data), 1 - beta2)
                )
                state['exp_avg_sq'] = exp_avg_sq
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # Compute bias-corrected estimates
                exp_avg_corrected = backend.divide(exp_avg, bias_correction1)
                exp_avg_sq_corrected = backend.divide(exp_avg_sq, bias_correction2)
                
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    max_exp_avg_sq = backend.maximum(max_exp_avg_sq, exp_avg_sq_corrected)
                    state['max_exp_avg_sq'] = max_exp_avg_sq
                    denom = backend.add(backend.sqrt(max_exp_avg_sq), eps)
                else:
                    denom = backend.add(backend.sqrt(exp_avg_sq_corrected), eps)
                
                # Update parameters
                step_size = lr
                update = backend.divide(exp_avg_corrected, denom)
                p.data = backend.subtract(p.data, backend.multiply(update, step_size))
        
        return loss

class AdamW(Optimizer):

    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if isinstance(grad, Tensor):
                    grad_data = grad.data
                else:
                    grad_data = grad

                backend = p._backend
                param_id = id(p)

                # Initialize state
                if 'step' not in self.state[param_id]:
                    self.state[param_id]['step'] = 0
                    self.state[param_id]['exp_avg'] = backend.zeros(p.data.shape, dtype=p.data.dtype)
                    self.state[param_id]['exp_avg_sq'] = backend.zeros(p.data.shape, dtype=p.data.dtype)
                    if amsgrad:
                        self.state[param_id]['max_exp_avg_sq'] = backend.zeros(p.data.shape, dtype=p.data.dtype)

                state = self.state[param_id]
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                state['step'] += 1
                step = state['step']

                # Decoupled weight decay (applied directly to weights, not gradient)
                # In-place: p.data -= p.data * (lr * weight_decay)
                if weight_decay != 0:
                    backend.subtract(p.data, backend.multiply(p.data, lr * weight_decay), out=p.data)

                # Update biased first moment estimate in-place
                # exp_avg = beta1 * exp_avg + (1 - beta1) * grad_data
                backend.multiply(exp_avg, beta1, out=exp_avg)
                backend.add(exp_avg, backend.multiply(grad_data, 1 - beta1), out=exp_avg)

                # Update biased second raw moment estimate in-place
                # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad_data^2
                grad_sq = backend.multiply(grad_data, grad_data)
                backend.multiply(exp_avg_sq, beta2, out=exp_avg_sq)
                backend.add(exp_avg_sq, backend.multiply(grad_sq, 1 - beta2), out=exp_avg_sq)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Compute bias-corrected estimates (these need temps)
                exp_avg_corrected = backend.divide(exp_avg, bias_correction1)
                exp_avg_sq_corrected = backend.divide(exp_avg_sq, bias_correction2)

                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    backend.maximum(max_exp_avg_sq, exp_avg_sq_corrected, out=max_exp_avg_sq)
                    denom = backend.add(backend.sqrt(max_exp_avg_sq), eps)
                else:
                    denom = backend.add(backend.sqrt(exp_avg_sq_corrected), eps)

                # Update parameters in-place
                # p.data -= lr * exp_avg_corrected / denom
                update = backend.divide(exp_avg_corrected, denom)
                backend.subtract(p.data, backend.multiply(update, lr), out=p.data)

        return loss

__all__ = ['Adam', 'AdamW']
