"""
PySML Neural Network Module Base Class
Provides base functionality for all neural network layers
"""

from typing import List, Iterator
from ..tensor import Tensor


class Module:
    def __init__(self):
        self._training = True
        self._modules = {}
        self._parameters = {}
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward()")
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def train(self, mode: bool = True):
        self._training = mode
        # Recursively set training mode for all submodules
        for module in self._modules.values():
            if isinstance(module, Module):
                module.train(mode)
        return self
    
    def eval(self):
        return self.train(False)
    
    @property
    def training(self) -> bool:
        return self._training
    
    def parameters(self) -> List[Tensor]:
        params = []
        
        # Add direct parameters
        for param in self._parameters.values():
            if param is not None:
                params.append(param)
        
        # Add parameters from submodules
        for module in self._modules.values():
            if isinstance(module, Module):
                params.extend(module.parameters())
        
        return params
    
    def named_parameters(self, prefix: str = '') -> Iterator[tuple]:
        for name, param in self._parameters.items():
            if param is not None:
                full_name = f"{prefix}.{name}" if prefix else name
                yield (full_name, param)
        
        for name, module in self._modules.items():
            if isinstance(module, Module):
                submodule_prefix = f"{prefix}.{name}" if prefix else name
                yield from module.named_parameters(submodule_prefix)
    
    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()
    
    def to(self, device: str):
        from .. import engine
        
        # Move direct parameters
        for name, param in self._parameters.items():
            if param is not None:
                self._parameters[name] = engine.to_device(param, device)
        
        # Move submodules
        for module in self._modules.values():
            if isinstance(module, Module):
                module.to(device)
        
        return self
    
    def __setattr__(self, name: str, value):
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Tensor) and hasattr(value, 'requires_grad') and value.requires_grad:
            self._parameters[name] = value
        
        # Use object's __setattr__ to actually set the attribute
        object.__setattr__(self, name, value)
    
    def __repr__(self):
        lines = [self.__class__.__name__ + '(']
        
        for name, module in self._modules.items():
            mod_str = repr(module)
            mod_str = '  ' + mod_str.replace('\n', '\n  ')
            lines.append(f'  ({name}): {mod_str}')
        
        lines.append(')')
        return '\n'.join(lines)
    
    def state_dict(self):
        state = {}
        for name, param in self._parameters.items():
            state[name] = param.data
        
        # Recursively get child module states
        for name, module in self._modules.items():
            state[name] = module.state_dict()
        
        return state

    def load_state_dict(self, state_dict):
        for name, value in state_dict.items():
            if name in self._parameters:
                self._parameters[name].data = value
            elif name in self._modules:
                self._modules[name].load_state_dict(value)

    def free_memory(self):
        """
        Free all intermediate activations and gradients
        Call this after backward pass when you don't need the computation graph
        """
        # Free parameter gradients
        for param in self.parameters():
            if param.grad is not None:
                del param.grad.data
                param.grad = None
            param._prev.clear()
            param._backward = lambda: None
        
        # Recursively free submodules
        for module in self._modules.values():
            if isinstance(module, Module):
                module.free_memory()


class Sequential(Module):
    """
    A sequential container for modules.
    
    Modules will be added in the order they are passed in the constructor.
    
    Example:
        >>> model = Sequential(
        ...     Linear(10, 20),
        ...     ReLU(),
        ...     Linear(20, 10)
        ... )
        >>> output = model(input)
    """
    
    def __init__(self, *args):
        """
        Initialize Sequential module.
        
        Args:
            *args: Variable number of modules to add
        """
        super().__init__()
        self.layers = []
        
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)
            self.layers.append(module)
    
    def add_module(self, name: str, module: Module):
        """Add a child module to the current module"""
        self._modules[name] = module
        setattr(self, name, module)
    
    def forward(self, x):
        """
        Forward pass through all modules in sequence.
        
        Args:
            x: Input tensor
        
        Returns:
            Output tensor after passing through all layers
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __repr__(self):
        """String representation"""
        lines = ['Sequential(']
        for idx, layer in enumerate(self.layers):
            layer_str = repr(layer)
            layer_str = '  ' + layer_str.replace('\n', '\n  ')
            lines.append(f'  ({idx}): {layer_str}')
        lines.append(')')
        return '\n'.join(lines)


class ModuleList(Module):
    """
    Holds submodules in a list.
    
    Can be indexed like a regular Python list, but modules it contains
    are properly registered.
    
    Example:
        >>> layers = ModuleList([Linear(10, 20) for _ in range(5)])
        >>> for layer in layers:
        ...     x = layer(x)
    """
    
    def __init__(self, modules=None):
        """
        Initialize ModuleList.
        
        Args:
            modules: Optional iterable of modules to add
        """
        super().__init__()
        self._module_list = []
        
        if modules is not None:
            for idx, module in enumerate(modules):
                self.add_module(str(idx), module)
                self._module_list.append(module)
    
    def add_module(self, name: str, module: Module):
        """Add a child module"""
        self._modules[name] = module
    
    def append(self, module: Module):
        """Append a module to the end of the list"""
        idx = len(self._module_list)
        self.add_module(str(idx), module)
        self._module_list.append(module)
    
    def __getitem__(self, idx):
        """Get module at index"""
        return self._module_list[idx]
    
    def __len__(self):
        """Return number of modules"""
        return len(self._module_list)
    
    def __iter__(self):
        """Iterate over modules"""
        return iter(self._module_list)
    
    def forward(self, *args, **kwargs):
        """ModuleList has no forward method"""
        raise NotImplementedError("ModuleList has no forward() method")


class ModuleDict(Module):
    """
    Holds submodules in a dictionary.
    
    Can be indexed like a regular Python dict, but modules it contains
    are properly registered.
    
    Example:
        >>> layers = ModuleDict({
        ...     'linear1': Linear(10, 20),
        ...     'linear2': Linear(20, 10)
        ... })
        >>> x = layers['linear1'](x)
    """
    
    def __init__(self, modules=None):
        """
        Initialize ModuleDict.
        
        Args:
            modules: Optional dict of modules to add
        """
        super().__init__()
        self._module_dict = {}
        
        if modules is not None:
            for name, module in modules.items():
                self.add_module(name, module)
                self._module_dict[name] = module
    
    def add_module(self, name: str, module: Module):
        """Add a child module"""
        self._modules[name] = module
        self._module_dict[name] = module
    
    def __getitem__(self, key):
        """Get module by key"""
        return self._module_dict[key]
    
    def __setitem__(self, key, module):
        """Set module by key"""
        self.add_module(key, module)
    
    def keys(self):
        """Return keys"""
        return self._module_dict.keys()
    
    def values(self):
        """Return values"""
        return self._module_dict.values()
    
    def items(self):
        """Return items"""
        return self._module_dict.items()
    
    def forward(self, *args, **kwargs):
        """ModuleDict has no forward method"""
        raise NotImplementedError("ModuleDict has no forward() method")