

from __future__ import annotations

from typing import Optional

from .module import Module
from ..tensor import Tensor
from .. import engine

class CrossEntropyLoss(Module):

    
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        ignore_index: int = -100,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        backend = input._backend
        
        # Get number of classes
        num_classes = input.shape[1] if input.ndim >= 2 else input.shape[0]
        
        # Compute log softmax
        log_probs = engine.log_softmax(input, axis=1)
        
        # Get target data
        target_data = target.data if isinstance(target, Tensor) else target
        
        # Handle different input shapes
        if input.ndim == 2:
            # (N, C) input, (N,) target
            batch_size = input.shape[0]
            
            # Create one-hot encoding
            target_int = backend.astype(target_data, backend.int64)
            
            # Gather log probabilities for target classes
            # Manual gather: select log_probs[i, target[i]]
            losses = []
            for i in range(batch_size):
                idx = int(target_int[i])
                if idx == self.ignore_index:
                    losses.append(0.0)
                else:
                    losses.append(-float(log_probs.data[i, idx]))
            
            loss_data = backend.asarray(losses)
            
            # Apply label smoothing if specified
            if self.label_smoothing > 0:
                smooth_loss = -backend.mean(log_probs.data, axis=1)
                loss_data = (1 - self.label_smoothing) * loss_data + self.label_smoothing * smooth_loss
            
            # Apply class weights if specified
            if self.weight is not None:
                weight_data = self.weight.data if isinstance(self.weight, Tensor) else self.weight
                for i in range(batch_size):
                    idx = int(target_int[i])
                    if idx != self.ignore_index:
                        loss_data[i] *= weight_data[idx]
        else:
            # Handle higher dimensional inputs
            # Flatten to (N, C, -1) then process
            loss_data = backend.zeros((input.shape[0],), dtype=log_probs.data.dtype)
            # Simplified: just use the 2D path on reshaped data
            input_flat = input.reshape(input.shape[0], num_classes, -1)
            target_flat = target.reshape(target.shape[0], -1)
            
            for i in range(input.shape[0]):
                for j in range(target_flat.shape[1]):
                    idx = int(target_flat[i, j])
                    if idx != self.ignore_index:
                        loss_data[i] -= float(log_probs.data[i, idx])
            
            loss_data = loss_data / target_flat.shape[1]
        
        loss = input._new_like(loss_data, requires_grad=input._requires_grad)
        
        # Apply reduction
        if self.reduction == 'mean':
            # Exclude ignored indices from mean
            if self.ignore_index >= 0:
                mask = backend.not_equal(target_data, self.ignore_index)
                n_valid = backend.sum(mask)
                if float(n_valid) > 0:
                    return engine.divide(engine.sum_with_grad(loss), 
                                        Tensor([float(n_valid)], dtype=input._dtype, device=input.active_device))
            return engine.mean_with_grad(loss)
        elif self.reduction == 'sum':
            return engine.sum_with_grad(loss)
        else:
            return loss
    
    def extra_repr(self) -> str:
        return f'reduction={self.reduction}, label_smoothing={self.label_smoothing}'

class NLLLoss(Module):

    
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        ignore_index: int = -100,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        backend = input._backend
        batch_size = input.shape[0]
        target_data = target.data if isinstance(target, Tensor) else target
        target_int = backend.astype(target_data, backend.int64)
        
        losses = []
        for i in range(batch_size):
            idx = int(target_int[i])
            if idx == self.ignore_index:
                losses.append(0.0)
            else:
                losses.append(-float(input.data[i, idx]))
        
        loss_data = backend.asarray(losses)
        loss = input._new_like(loss_data, requires_grad=input._requires_grad)
        
        if self.reduction == 'mean':
            return engine.mean_with_grad(loss)
        elif self.reduction == 'sum':
            return engine.sum_with_grad(loss)
        return loss

class MSELoss(Module):

    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        diff = engine.subtract(input, target)
        squared = engine.power(diff, 2)
        
        if self.reduction == 'mean':
            return engine.mean_with_grad(squared)
        elif self.reduction == 'sum':
            return engine.sum_with_grad(squared)
        return squared
    
    def extra_repr(self) -> str:
        return f'reduction={self.reduction}'

class L1Loss(Module):

    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        diff = engine.subtract(input, target)
        abs_diff = engine.abs(diff)
        
        if self.reduction == 'mean':
            return engine.mean_with_grad(abs_diff)
        elif self.reduction == 'sum':
            return engine.sum_with_grad(abs_diff)
        return abs_diff
    
    def extra_repr(self) -> str:
        return f'reduction={self.reduction}'

class SmoothL1Loss(Module):

    
    def __init__(self, reduction: str = 'mean', beta: float = 1.0):
        super().__init__()
        self.reduction = reduction
        self.beta = beta
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        backend = input._backend
        diff = engine.subtract(input, target)
        abs_diff = engine.abs(diff)
        
        # Smooth L1: use L2 when |diff| < beta, L1 otherwise
        l2_part = engine.multiply(engine.power(diff, 2), 0.5 / self.beta)
        l1_part = engine.subtract(abs_diff, Tensor([0.5 * self.beta], dtype=input._dtype, device=input.active_device))
        
        mask = backend.less(abs_diff.data, self.beta)
        loss_data = backend.where(mask, l2_part.data, l1_part.data)
        loss = input._new_like(loss_data, requires_grad=input._requires_grad)
        
        if self.reduction == 'mean':
            return engine.mean_with_grad(loss)
        elif self.reduction == 'sum':
            return engine.sum_with_grad(loss)
        return loss
    
    def extra_repr(self) -> str:
        return f'reduction={self.reduction}, beta={self.beta}'

class BCELoss(Module):

    
    def __init__(self, weight: Optional[Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        backend = input._backend
        eps = 1e-7
        
        # Clamp input for numerical stability
        input_clamped = engine.clip(input, eps, 1.0 - eps)
        
        # BCE: -[y * log(p) + (1-y) * log(1-p)]
        term1 = engine.multiply(target, engine.log(input_clamped))
        one_minus_target = engine.subtract(
            Tensor(backend.ones(target.shape), dtype=target._dtype, device=target.active_device),
            target
        )
        one_minus_input = engine.subtract(
            Tensor(backend.ones(input_clamped.shape), dtype=input._dtype, device=input.active_device),
            input_clamped
        )
        term2 = engine.multiply(one_minus_target, engine.log(one_minus_input))
        
        loss = engine.negative(engine.add(term1, term2))
        
        if self.weight is not None:
            loss = engine.multiply(loss, self.weight)
        
        if self.reduction == 'mean':
            return engine.mean_with_grad(loss)
        elif self.reduction == 'sum':
            return engine.sum_with_grad(loss)
        return loss
    
    def extra_repr(self) -> str:
        return f'reduction={self.reduction}'

class BCEWithLogitsLoss(Module):

    
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        reduction: str = 'mean',
        pos_weight: Optional[Tensor] = None
    ):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.pos_weight = pos_weight
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        backend = input._backend
        
        # Numerically stable BCE with logits:
        # max(x, 0) - x * target + log(1 + exp(-|x|))
        max_val = engine.relu(input)
        neg_abs = engine.negative(engine.abs(input))
        
        # log(1 + exp(-|x|))
        log_term = engine.log(
            engine.add(
                Tensor(backend.ones(input.shape), dtype=input._dtype, device=input.active_device),
                engine.exp(neg_abs)
            )
        )
        
        loss = engine.add(
            engine.subtract(max_val, engine.multiply(input, target)),
            log_term
        )
        
        if self.pos_weight is not None:
            # Apply positive weight
            pos_term = engine.multiply(
                engine.multiply(target, engine.log(engine.sigmoid(input))),
                self.pos_weight
            )
            # Recalculate with pos_weight
            pass  # Simplified for now
        
        if self.weight is not None:
            loss = engine.multiply(loss, self.weight)
        
        if self.reduction == 'mean':
            return engine.mean_with_grad(loss)
        elif self.reduction == 'sum':
            return engine.sum_with_grad(loss)
        return loss
    
    def extra_repr(self) -> str:
        return f'reduction={self.reduction}'

class KLDivLoss(Module):

    
    def __init__(self, reduction: str = 'mean', log_target: bool = False):
        super().__init__()
        self.reduction = reduction
        self.log_target = log_target
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        backend = input._backend
        
        if self.log_target:
            # target is log(P), input is log(Q)
            # KL = exp(log(P)) * (log(P) - log(Q)) = exp(log(P)) * log(P) - exp(log(P)) * log(Q)
            loss = engine.multiply(
                engine.exp(target),
                engine.subtract(target, input)
            )
        else:
            # target is P, input is log(Q)
            # KL = P * (log(P) - log(Q)) = P * log(P) - P * log(Q)
            # Since P * log(P) doesn't depend on Q, we often just compute -P * log(Q)
            loss = engine.multiply(target, engine.subtract(engine.log(target), input))
        
        if self.reduction == 'mean':
            return engine.mean_with_grad(loss)
        elif self.reduction == 'sum':
            return engine.sum_with_grad(loss)
        elif self.reduction == 'batchmean':
            batch_size = input.shape[0]
            return engine.divide(
                engine.sum_with_grad(loss),
                Tensor([float(batch_size)], dtype=input._dtype, device=input.active_device)
            )
        return loss
    
    def extra_repr(self) -> str:
        return f'reduction={self.reduction}'

__all__ = [
    'CrossEntropyLoss', 'NLLLoss', 'MSELoss', 'L1Loss', 
    'SmoothL1Loss', 'BCELoss', 'BCEWithLogitsLoss', 'KLDivLoss'
]
