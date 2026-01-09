

from __future__ import annotations

import weakref
from collections import OrderedDict
from typing import Iterator, Optional, Tuple, Dict, Any, Callable, Set, List, Union

from ..tensor import Tensor
from ..dtype import bf16

class Parameter(Tensor):

    
    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            # Create empty parameter
            instance = super().__new__(cls)
            return instance
        
        if isinstance(data, Tensor):
            instance = super().__new__(cls)
            return instance
        
        instance = super().__new__(cls)
        return instance
    
    def __init__(self, data=None, requires_grad=True):
        if data is None:
            return
            
        if isinstance(data, Tensor):
            # Copy from existing tensor
            super().__init__(
                data.data,
                dtype=data._dtype,
                device=data.active_device,
                requires_grad=requires_grad
            )
        else:
            super().__init__(
                data,
                dtype=bf16(),
                device="cpu",
                requires_grad=requires_grad
            )
    
    def __repr__(self):
        return f"Parameter containing:\n{super().__repr__()}"

class Module:

    
    _version: int = 1
    training: bool
    
    def __init__(self):
        self._parameters: Dict[str, Optional[Parameter]] = OrderedDict()
        self._buffers: Dict[str, Optional[Tensor]] = OrderedDict()
        self._modules: Dict[str, Optional['Module']] = OrderedDict()
        self._hooks: Dict[int, Callable] = OrderedDict()
        self._forward_hooks: Dict[int, Callable] = OrderedDict()
        self._backward_hooks: Dict[int, Callable] = OrderedDict()
        self.training = True
        self._is_stateful = False
    
    def forward(self, *args, **kwargs):

        raise NotImplementedError(
            f"Module [{type(self).__name__}] is missing the required 'forward' function"
        )
    
    def __call__(self, *args, **kwargs):

        # Pre-forward hooks
        for hook in self._hooks.values():
            result = hook(self, args)
            if result is not None:
                if not isinstance(result, tuple):
                    result = (result,)
                args = result
        
        # Forward pass
        result = self.forward(*args, **kwargs)
        
        # Post-forward hooks
        for hook in self._forward_hooks.values():
            hook_result = hook(self, args, result)
            if hook_result is not None:
                result = hook_result
        
        return result
    
    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:

        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call"
            )
        
        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(
                f"cannot assign '{type(param).__name__}' object to parameter '{name}' "
                f"(Parameter or None required)"
            )
        else:
            self._parameters[name] = param
    
    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None:

        if '_buffers' not in self.__dict__:
            raise AttributeError(
                "cannot assign buffer before Module.__init__() call"
            )
        
        if tensor is not None and not isinstance(tensor, Tensor):
            raise TypeError(
                f"cannot assign '{type(tensor).__name__}' object to buffer '{name}' "
                f"(Tensor or None required)"
            )
        
        self._buffers[name] = tensor
    
    def add_module(self, name: str, module: Optional['Module']) -> None:

        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{type(module).__name__} is not a Module subclass")
        
        if hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute '{name}' already exists")
        
        self._modules[name] = module
    
    def __setattr__(self, name: str, value: Any) -> None:

        
        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]
        
        params = self.__dict__.get('_parameters')
        modules = self.__dict__.get('_modules')
        buffers = self.__dict__.get('_buffers')
        
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameter before Module.__init__() call"
                )
            remove_from(self.__dict__, modules, buffers)
            self.register_parameter(name, value)
        elif isinstance(value, Module):
            if modules is None:
                raise AttributeError(
                    "cannot assign module before Module.__init__() call"
                )
            remove_from(self.__dict__, params, buffers)
            modules[name] = value
        elif params is not None and name in params:
            if value is not None and not isinstance(value, Parameter):
                raise TypeError(
                    f"cannot assign '{type(value).__name__}' to parameter '{name}'"
                )
            self.register_parameter(name, value)
        else:
            object.__setattr__(self, name, value)
    
    def __getattr__(self, name: str) -> Any:

        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        
        if '_modules' in self.__dict__:
            _modules = self.__dict__['_modules']
            if name in _modules:
                return _modules[name]
        
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )
    
    def __delattr__(self, name: str) -> None:

        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)
    
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:

        for name, param in self.named_parameters(recurse=recurse):
            yield param
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:

        memo: Set[int] = set()
        
        for name, param in self._parameters.items():
            if param is None:
                continue
            if id(param) in memo:
                continue
            memo.add(id(param))
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, param
        
        if recurse:
            for module_name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = f"{prefix}.{module_name}" if prefix else module_name
                for name, param in module.named_parameters(prefix=submodule_prefix, recurse=recurse):
                    if id(param) in memo:
                        continue
                    memo.add(id(param))
                    yield name, param
    
    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:

        for name, buf in self.named_buffers(recurse=recurse):
            yield buf
    
    def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:

        memo: Set[int] = set()
        
        for name, buf in self._buffers.items():
            if buf is None:
                continue
            if id(buf) in memo:
                continue
            memo.add(id(buf))
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, buf
        
        if recurse:
            for module_name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = f"{prefix}.{module_name}" if prefix else module_name
                for name, buf in module.named_buffers(prefix=submodule_prefix, recurse=recurse):
                    if id(buf) in memo:
                        continue
                    memo.add(id(buf))
                    yield name, buf
    
    def children(self) -> Iterator['Module']:

        for name, module in self._modules.items():
            if module is not None:
                yield module
    
    def named_children(self) -> Iterator[Tuple[str, 'Module']]:

        for name, module in self._modules.items():
            if module is not None:
                yield name, module
    
    def modules(self) -> Iterator['Module']:

        for name, module in self.named_modules():
            yield module
    
    def named_modules(self, memo: Optional[Set[int]] = None, prefix: str = '') -> Iterator[Tuple[str, 'Module']]:

        if memo is None:
            memo = set()
        
        if id(self) not in memo:
            memo.add(id(self))
            yield prefix, self
            
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = f"{prefix}.{name}" if prefix else name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m
    
    def train(self, mode: bool = True) -> 'Module':

        self.training = mode
        for module in self.children():
            module.train(mode)
        return self
    
    def eval(self) -> 'Module':

        return self.train(False)
    
    def requires_grad_(self, requires_grad: bool = True) -> 'Module':

        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self
    
    def zero_grad(self, set_to_none: bool = True) -> None:

        for p in self.parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.zero_grad()
    
    def state_dict(self, destination: Optional[Dict] = None, prefix: str = '') -> Dict[str, Any]:

        if destination is None:
            destination = OrderedDict()
        
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param.clone().detach()
        
        for name, buf in self._buffers.items():
            if buf is not None:
                destination[prefix + name] = buf.clone().detach()
        
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + '.')
        
        return destination
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> Tuple[List[str], List[str]]:

        missing_keys: List[str] = []
        unexpected_keys: List[str] = list(state_dict.keys())
        
        def _get_data_for_param(value, param):
            """Get data in the correct format for the parameter's backend."""
            import numpy as np

            # First, get raw numpy data from the value
            if hasattr(value, 'data'):
                data = value.data
                if isinstance(data, memoryview):
                    np_data = np.array(data)
                elif hasattr(data, 'asnumpy'):
                    np_data = data.asnumpy()
                elif hasattr(data, 'copy'):
                    np_data = data.copy()
                else:
                    np_data = np.array(data)
            elif hasattr(value, 'asnumpy'):
                np_data = value.asnumpy()
            else:
                np_data = np.array(value)

            # Convert to the parameter's backend format
            if param is not None and hasattr(param, '_backend'):
                backend = param._backend
                if hasattr(backend, 'convert'):
                    return backend.convert(np_data, param._dtype, device=param.active_device)

            return np_data

        def load(module: Module, prefix: str = ''):
            for name, param in module._parameters.items():
                key = prefix + name
                if key in state_dict:
                    if key in unexpected_keys:
                        unexpected_keys.remove(key)
                    if param is not None:
                        param.data = _get_data_for_param(state_dict[key], param)
                elif strict:
                    missing_keys.append(key)

            for name, buf in module._buffers.items():
                key = prefix + name
                if key in state_dict:
                    if key in unexpected_keys:
                        unexpected_keys.remove(key)
                    if buf is not None:
                        buf.data = _get_data_for_param(state_dict[key], buf)
                elif strict:
                    missing_keys.append(key)
            
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        
        load(self)
        
        if strict:
            if missing_keys:
                raise RuntimeError(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                raise RuntimeError(f"Unexpected keys: {unexpected_keys}")
        
        return missing_keys, unexpected_keys
    
    def to(self, device: str = None, dtype = None) -> 'Module':

        for name, param in self._parameters.items():
            if param is not None:
                self._parameters[name] = Parameter(param.to(device=device, dtype=dtype))
        
        for name, buf in self._buffers.items():
            if buf is not None:
                self._buffers[name] = buf.to(device=device, dtype=dtype)
        
        for module in self.children():
            module.to(device=device, dtype=dtype)
        
        return self
    
    def cpu(self) -> 'Module':

        return self.to(device='cpu')
    
    def cuda(self, device: Optional[int] = None) -> 'Module':

        target = 'cuda' if device is None else f'cuda:{device}'
        return self.to(device=target)
    
    def xpu(self, device: Optional[int] = None) -> 'Module':

        target = 'xpu' if device is None else f'xpu:{device}'
        return self.to(device=target)
    
    def half(self) -> 'Module':

        from ..dtype import fp16
        return self.to(dtype=fp16())
    
    def float(self) -> 'Module':

        from ..dtype import fp32
        return self.to(dtype=fp32())
    
    def bfloat16(self) -> 'Module':

        from ..dtype import bf16
        return self.to(dtype=bf16())
    
    def apply(self, fn: Callable[['Module'], None]) -> 'Module':

        for module in self.children():
            module.apply(fn)
        fn(self)
        return self
    
    def register_forward_hook(self, hook: Callable) -> Any:

        handle = len(self._forward_hooks)
        self._forward_hooks[handle] = hook
        return handle
    
    def register_backward_hook(self, hook: Callable) -> Any:

        handle = len(self._backward_hooks)
        self._backward_hooks[handle] = hook
        return handle
    
    def register_forward_pre_hook(self, hook: Callable) -> Any:

        handle = len(self._hooks)
        self._hooks[handle] = hook
        return handle
    
    def extra_repr(self) -> str:

        return ''
    
    def __repr__(self) -> str:

        extra = self.extra_repr()
        extra_lines = extra.split('\n') if extra else []
        
        child_lines = []
        for name, module in self._modules.items():
            mod_str = repr(module)
            mod_str = '\n'.join('  ' + line for line in mod_str.split('\n'))
            child_lines.append(f'({name}): {mod_str.lstrip()}')
        
        lines = extra_lines + child_lines
        
        main_str = f'{self.__class__.__name__}('
        if lines:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        
        return main_str
    
    def _get_name(self) -> str:

        return self.__class__.__name__
    
    def num_parameters(self, only_trainable: bool = False) -> int:

        total = 0
        for p in self.parameters():
            if only_trainable and not p.requires_grad:
                continue
            total += p.numel()
        return total

__all__ = ['Module', 'Parameter']
