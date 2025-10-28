"""
PySML Module — v0.4.8-final
Optimized: lightweight module graph, parameter registration, state_dict.
"""

from ..tensor import Tensor

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True

    def parameters(self):
        for name, p in self._parameters.items():
            yield p
        for name, m in self._modules.items():
            yield from m.parameters()

    def add_module(self, name, module):
        self._modules[name] = module
        return module

    def __setattr__(self, name, value):
        if isinstance(value, Tensor):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self, mode=True):
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self):
        sd = {name: p.data for name, p in self._parameters.items()}
        for name, m in self._modules.items():
            sd[name] = m.state_dict()
        return sd

    def load_state_dict(self, state_dict):
        for name, val in state_dict.items():
            if name in self._parameters:
                self._parameters[name].data[...] = val
            elif name in self._modules:
                self._modules[name].load_state_dict(val)
