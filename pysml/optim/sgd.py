

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

from .optimizer import Optimizer
from ..tensor import Tensor

class SGD(Optimizer):

    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):

        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            dampening = group['dampening']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if isinstance(grad, Tensor):
                    grad_data = grad.data
                else:
                    grad_data = grad
                
                backend = p._backend
                
                # Apply weight decay
                if weight_decay != 0:
                    grad_data = backend.add(grad_data, backend.multiply(p.data, weight_decay))
                
                # Apply momentum
                if momentum != 0:
                    param_id = id(p)
                    if 'momentum_buffer' not in self.state[param_id]:
                        buf = backend.copy(grad_data)
                        self.state[param_id]['momentum_buffer'] = buf
                    else:
                        buf = self.state[param_id]['momentum_buffer']
                        buf = backend.add(
                            backend.multiply(buf, momentum),
                            backend.multiply(grad_data, 1 - dampening)
                        )
                        self.state[param_id]['momentum_buffer'] = buf
                    
                    if nesterov:
                        grad_data = backend.add(grad_data, backend.multiply(buf, momentum))
                    else:
                        grad_data = buf
                
                # Update parameters
                p.data = backend.subtract(p.data, backend.multiply(grad_data, lr))
        
        return loss

__all__ = ['SGD']
