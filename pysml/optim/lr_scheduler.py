

from __future__ import annotations

import math
from typing import List, Optional

from .optimizer import Optimizer

class LRScheduler:

    
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # Initialize
        self.step()
    
    def get_lr(self) -> List[float]:

        raise NotImplementedError
    
    def step(self, epoch: Optional[int] = None) -> None:

        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
    
    def state_dict(self):

        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
    
    def load_state_dict(self, state_dict):

        self.__dict__.update(state_dict)

class StepLR(LRScheduler):

    
    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1
    ):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]

class MultiStepLR(LRScheduler):

    
    def __init__(
        self,
        optimizer: Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        last_epoch: int = -1
    ):
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]

class ExponentialLR(LRScheduler):

    
    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float,
        last_epoch: int = -1
    ):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch == 0:
            return self.base_lrs
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]

class CosineAnnealingLR(LRScheduler):

    
    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0,
        last_epoch: int = -1
    ):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch == 0:
            return self.base_lrs
        
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]

class CosineAnnealingWarmRestarts(LRScheduler):

    
    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = -1
    ):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]
    
    def step(self, epoch: Optional[int] = None) -> None:
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, but got {epoch}")
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        
        self.last_epoch = math.floor(epoch)
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class LinearLR(LRScheduler):

    
    def __init__(
        self,
        optimizer: Optimizer,
        start_factor: float = 1.0 / 3,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = -1
    ):
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch == 0:
            return [base_lr * self.start_factor for base_lr in self.base_lrs]
        
        if self.last_epoch >= self.total_iters:
            return [base_lr * self.end_factor for base_lr in self.base_lrs]
        
        factor = self.start_factor + (self.end_factor - self.start_factor) * self.last_epoch / self.total_iters
        return [base_lr * factor for base_lr in self.base_lrs]

class ConstantLR(LRScheduler):

    
    def __init__(
        self,
        optimizer: Optimizer,
        factor: float = 1.0 / 3,
        total_iters: int = 5,
        last_epoch: int = -1
    ):
        self.factor = factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.total_iters:
            return [base_lr * self.factor for base_lr in self.base_lrs]
        return self.base_lrs

class OneCycleLR(LRScheduler):

    
    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = 'cos',
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1
    ):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.initial_lr = max_lr / div_factor
        self.min_lr = self.initial_lr / final_div_factor
        
        self.step_up = int(total_steps * pct_start)
        self.step_down = total_steps - self.step_up
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.step_up:
            # Increasing phase
            pct = self.last_epoch / self.step_up
            if self.anneal_strategy == 'cos':
                lr = self.initial_lr + (self.max_lr - self.initial_lr) * (1 - math.cos(math.pi * pct)) / 2
            else:
                lr = self.initial_lr + (self.max_lr - self.initial_lr) * pct
        else:
            # Decreasing phase
            pct = (self.last_epoch - self.step_up) / self.step_down
            if self.anneal_strategy == 'cos':
                lr = self.min_lr + (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * pct)) / 2
            else:
                lr = self.max_lr - (self.max_lr - self.min_lr) * pct
        
        return [lr] * len(self.base_lrs)

__all__ = [
    'LRScheduler', 'StepLR', 'MultiStepLR', 'ExponentialLR',
    'CosineAnnealingLR', 'CosineAnnealingWarmRestarts',
    'LinearLR', 'ConstantLR', 'OneCycleLR'
]
