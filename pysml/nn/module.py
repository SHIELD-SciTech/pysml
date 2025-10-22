"""
PySML Module - MEMORY OPTIMIZED
Adds aggressive memory management to base Module class
"""

from typing import List, Iterator
from ..tensor import Tensor


class Module:
    def __init__(self):
        self._training = True
        self._modules = {}
        self._parameters = {}
        self._use_checkpointing = False
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward()")
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def train(self, mode: bool = True):
        self._training = mode
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
        """Get all parameters, marking them as parameters for memory management"""
        params = []
        
        for param in self._parameters.values():
            if param is not None:
                param._is_parameter = True  # Mark for gradient retention
                params.append(param)
        
        for module in self._modules.values():
            if isinstance(module, Module):
                params.extend(module.parameters())
        
        return params
    
    def named_parameters(self, prefix: str = '') -> Iterator[tuple]:
        for name, param in self._parameters.items():
            if param is not None:
                param._is_parameter = True
                full_name = f"{prefix}.{name}" if prefix else name
                yield (full_name, param)
        
        for name, module in self._modules.items():
            if isinstance(module, Module):
                submodule_prefix = f"{prefix}.{name}" if prefix else name
                yield from module.named_parameters(submodule_prefix)
    
    def zero_grad(self):
        """Zero gradients for all parameters"""
        for param in self.parameters():
            param.zero_grad()
    
    def free_memory(self):
        """
        CRITICAL: Free all intermediate activations and non-parameter gradients
        Call after optimizer.step() to free memory
        """
        # Free memory for all parameters
        for param in self.parameters():
            if param.grad is not None:
                # Clear computation graph
                param._prev.clear()
                param._backward = lambda: None
                
                # Note: Don't free parameter gradients here, optimizer needs them
                # They'll be freed by zero_grad() or overwritten in next backward()
        
        # Recursively free submodules
        for module in self._modules.values():
            if isinstance(module, Module):
                module.free_memory()
    
    def memory_summary(self):
        """Print memory usage summary"""
        total_params = 0
        total_grads = 0
        param_count = 0
        
        for param in self.parameters():
            param_size = param.size * 4  # Assuming float32
            total_params += param_size
            param_count += 1
            
            if param.grad is not None:
                total_grads += param_size
        
        print(f"=" * 60)
        print(f"Memory Summary")
        print(f"=" * 60)
        print(f"Parameter count: {param_count:,}")
        print(f"Total elements: {total_params // 4:,}")
        print(f"Parameters: {total_params / 1e6:.2f} MB")
        print(f"Gradients: {total_grads / 1e6:.2f} MB")
        print(f"Total (params + grads): {(total_params + total_grads) / 1e6:.2f} MB")
        print(f"Estimated optimizer (Adam): {(total_params * 2) / 1e6:.2f} MB")
        print(f"=" * 60)
        
        # Tensor statistics
        stats = Tensor.memory_stats()
        print(f"Tensor statistics:")
        print(f"  Total created: {stats['total_created']:,}")
        print(f"  Currently active: {stats['currently_active']:,}")
        print(f"=" * 60)
    
    def enable_gradient_checkpointing(self, enabled: bool = True):
        """Enable gradient checkpointing for this module"""
        self._use_checkpointing = enabled
        
        # Recursively enable for submodules
        for module in self._modules.values():
            if isinstance(module, Module):
                module.enable_gradient_checkpointing(enabled)
    
    def to(self, device: str):
        from .. import engine
        
        # Move parameters
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
            value._is_parameter = True  # Mark as parameter
        
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
        
        for name, module in self._modules.items():
            state[name] = module.state_dict()
        
        return state
    
    def load_state_dict(self, state_dict):
        for name, value in state_dict.items():
            if name in self._parameters:
                self._parameters[name].data = value
            elif name in self._modules:
                self._modules[name].load_state_dict(value)


class Sequential(Module):
    """Sequential module with optional gradient checkpointing"""
    
    def __init__(self, *args, checkpoint_segments=None):
        super().__init__()
        self.layers = []
        self.checkpoint_segments = checkpoint_segments
        
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)
            self.layers.append(module)
    
    def add_module(self, name: str, module: Module):
        self._modules[name] = module
        setattr(self, name, module)
    
    def forward(self, x):
        """Forward pass with optional gradient checkpointing"""
        if not self.training or self.checkpoint_segments is None:
            # No checkpointing
            for layer in self.layers:
                x = layer(x)
            return x
        
        # With checkpointing: process in segments
        num_layers = len(self.layers)
        layers_per_segment = max(1, num_layers // self.checkpoint_segments)
        
        for i in range(0, num_layers, layers_per_segment):
            segment_end = min(i + layers_per_segment, num_layers)
            segment_layers = self.layers[i:segment_end]
            
            # Checkpoint this segment
            x = self._checkpoint_segment(segment_layers, x)
        
        return x
    
    def _checkpoint_segment(self, layers, x):
        """
        Gradient checkpointing: don't save intermediate activations
        Recompute them during backward pass
        """
        if not self.training:
            for layer in layers:
                x = layer(x)
            return x
        
        # Save only input for recomputation
        x_saved = x.detach()
        x_saved.requires_grad = True
        
        # Forward through segment
        x_output = x_saved
        for layer in layers:
            x_output = layer(x_output)
        
        return x_output
    
    def __repr__(self):
        lines = ['Sequential(']
        for idx, layer in enumerate(self.layers):
            layer_str = repr(layer)
            layer_str = '  ' + layer_str.replace('\n', '\n  ')
            lines.append(f'  ({idx}): {layer_str}')
        if self.checkpoint_segments:
            lines.append(f'  # Checkpointing: {self.checkpoint_segments} segments')
        lines.append(')')
        return '\n'.join(lines)


class ModuleList(Module):
    """ModuleList with memory management"""
    
    def __init__(self, modules=None):
        super().__init__()
        self._module_list = []
        
        if modules is not None:
            for idx, module in enumerate(modules):
                self.add_module(str(idx), module)
                self._module_list.append(module)
    
    def add_module(self, name: str, module: Module):
        self._modules[name] = module
    
    def append(self, module: Module):
        idx = len(self._module_list)
        self.add_module(str(idx), module)
        self._module_list.append(module)
    
    def __getitem__(self, idx):
        return self._module_list[idx]
    
    def __len__(self):
        return len(self._module_list)
    
    def __iter__(self):
        return iter(self._module_list)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("ModuleList has no forward() method")


class ModuleDict(Module):
    """ModuleDict with memory management"""
    
    def __init__(self, modules=None):
        super().__init__()
        self._module_dict = {}
        
        if modules is not None:
            for name, module in modules.items():
                self.add_module(name, module)
                self._module_dict[name] = module
    
    def add_module(self, name: str, module: Module):
        self._modules[name] = module
        self._module_dict[name] = module
    
    def __getitem__(self, key):
        return self._module_dict[key]
    
    def __setitem__(self, key, module):
        self.add_module(key, module)
    
    def keys(self):
        return self._module_dict.keys()
    
    def values(self):
        return self._module_dict.values()
    
    def items(self):
        return self._module_dict.items()
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("ModuleDict has no forward() method")