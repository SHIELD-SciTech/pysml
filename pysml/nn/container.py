

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Iterator, Optional, Tuple, Union, List, Iterable

from .module import Module

class Sequential(Module):

    
    def __init__(self, *args):
        super().__init__()
        
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[Module, 'Sequential']:
        if isinstance(idx, slice):
            return Sequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return list(self._modules.values())[idx]
    
    def __setitem__(self, idx: int, module: Module) -> None:
        key = list(self._modules.keys())[idx]
        self._modules[key] = module
    
    def __delitem__(self, idx: int) -> None:
        key = list(self._modules.keys())[idx]
        del self._modules[key]
    
    def __len__(self) -> int:
        return len(self._modules)
    
    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())
    
    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x
    
    def append(self, module: Module) -> 'Sequential':

        self.add_module(str(len(self)), module)
        return self
    
    def extend(self, modules: Iterable[Module]) -> 'Sequential':

        for module in modules:
            self.append(module)
        return self
    
    def insert(self, index: int, module: Module) -> 'Sequential':

        items = list(self._modules.items())
        items.insert(index, (str(len(self)), module))
        self._modules = OrderedDict()
        for i, (_, mod) in enumerate(items):
            self._modules[str(i)] = mod
        return self

class ModuleList(Module):

    
    def __init__(self, modules: Optional[Iterable[Module]] = None):
        super().__init__()
        if modules is not None:
            self.extend(modules)
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[Module, 'ModuleList']:
        if isinstance(idx, slice):
            return ModuleList(list(self._modules.values())[idx])
        else:
            if idx < 0:
                idx = len(self) + idx
            return self._modules[str(idx)]
    
    def __setitem__(self, idx: int, module: Module) -> None:
        if idx < 0:
            idx = len(self) + idx
        self._modules[str(idx)] = module
    
    def __delitem__(self, idx: int) -> None:
        if idx < 0:
            idx = len(self) + idx
        del self._modules[str(idx)]
        # Re-index
        items = list(self._modules.values())
        self._modules = OrderedDict()
        for i, mod in enumerate(items):
            self._modules[str(i)] = mod
    
    def __len__(self) -> int:
        return len(self._modules)
    
    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())
    
    def __contains__(self, module: Module) -> bool:
        return module in self._modules.values()
    
    def append(self, module: Module) -> 'ModuleList':

        self.add_module(str(len(self)), module)
        return self
    
    def extend(self, modules: Iterable[Module]) -> 'ModuleList':

        for module in modules:
            self.append(module)
        return self
    
    def insert(self, index: int, module: Module) -> 'ModuleList':

        items = list(self._modules.values())
        items.insert(index, module)
        self._modules = OrderedDict()
        for i, mod in enumerate(items):
            self._modules[str(i)] = mod
        return self
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("ModuleList does not implement forward()")

class ModuleDict(Module):

    
    def __init__(self, modules: Optional[Dict[str, Module]] = None):
        super().__init__()
        if modules is not None:
            self.update(modules)
    
    def __getitem__(self, key: str) -> Module:
        return self._modules[key]
    
    def __setitem__(self, key: str, module: Module) -> None:
        self.add_module(key, module)
    
    def __delitem__(self, key: str) -> None:
        del self._modules[key]
    
    def __len__(self) -> int:
        return len(self._modules)
    
    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)
    
    def __contains__(self, key: str) -> bool:
        return key in self._modules
    
    def keys(self) -> Iterator[str]:
        return iter(self._modules.keys())
    
    def values(self) -> Iterator[Module]:
        return iter(self._modules.values())
    
    def items(self) -> Iterator[Tuple[str, Module]]:
        return iter(self._modules.items())
    
    def get(self, key: str, default: Optional[Module] = None) -> Optional[Module]:
        return self._modules.get(key, default)
    
    def pop(self, key: str) -> Module:
        module = self._modules[key]
        del self._modules[key]
        return module
    
    def update(self, modules: Dict[str, Module]) -> 'ModuleDict':

        for key, module in modules.items():
            self.add_module(key, module)
        return self
    
    def clear(self) -> None:

        self._modules.clear()
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("ModuleDict does not implement forward()")

__all__ = ['Sequential', 'ModuleList', 'ModuleDict']
