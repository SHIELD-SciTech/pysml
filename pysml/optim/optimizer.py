

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from ..tensor import Tensor
from ..nn.module import Parameter

class Optimizer:

    
    def __init__(self, params, defaults: Dict[str, Any]):
        self.defaults = defaults
        self.state: Dict[int, Dict[str, Any]] = defaultdict(dict)
        self.param_groups: List[Dict[str, Any]] = []
        
        # Handle parameter groups
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
        
        for param_group in param_groups:
            self.add_param_group(param_group)
    
    def add_param_group(self, param_group: Dict[str, Any]) -> None:

        params = param_group['params']
        if isinstance(params, (Tensor, Parameter)):
            param_group['params'] = [params]
        else:
            param_group['params'] = list(params)
        
        # Apply defaults
        for name, default in self.defaults.items():
            if name not in param_group:
                param_group[name] = default
        
        self.param_groups.append(param_group)
    
    def zero_grad(self, set_to_none: bool = True) -> None:

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()
    
    def step(self, closure: Optional[Callable] = None):

        raise NotImplementedError
    
    def state_dict(self) -> Dict[str, Any]:

        # Pack state
        packed_state = {}
        for param_id, param_state in self.state.items():
            packed_state[param_id] = {}
            for key, value in param_state.items():
                if isinstance(value, Tensor):
                    packed_state[param_id][key] = value.clone().detach()
                else:
                    packed_state[param_id][key] = value
        
        return {
            'state': packed_state,
            'param_groups': [
                {k: v for k, v in group.items() if k != 'params'}
                for group in self.param_groups
            ]
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:

        # Load state
        for param_id, param_state in state_dict['state'].items():
            self.state[int(param_id)] = {}
            for key, value in param_state.items():
                if isinstance(value, Tensor):
                    self.state[int(param_id)][key] = value.clone()
                else:
                    self.state[int(param_id)][key] = value
        
        # Load param groups (excluding params)
        for group, saved_group in zip(self.param_groups, state_dict['param_groups']):
            group.update(saved_group)
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += '\n'
            format_string += f'Parameter Group {i}\n'
            for key, value in group.items():
                if key != 'params':
                    format_string += f'    {key}: {value}\n'
        format_string += ')'
        return format_string

__all__ = ['Optimizer']
