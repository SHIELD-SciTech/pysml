from typing import Iterator, Tuple, Optional, List, Dict, Any, Callable, Union
from collections import OrderedDict
import weakref


class Parameter:
	def __init__(self, data, requires_grad=True):
		from ..tensor import Tensor
		
		if not isinstance(data, Tensor):
			data = Tensor(data, requires_grad=requires_grad)
		else:
			data._requires_grad = requires_grad
		
		self.data = data
		self._requires_grad = requires_grad
	
	@property
	def shape(self):
		return self.data.shape
	
	@property
	def grad(self):
		return self.data.grad
	
	@grad.setter
	def grad(self, value):
		self.data.grad = value
	
	@property
	def requires_grad(self):
		return self._requires_grad
	
	def zero_grad(self):
		self.data.zero_grad()
	
	def to(self, device):
		self.data.to(device)
		return self
	
	def numpy(self):
		return self.data.numpy()
	
	def item(self):
		return self.data.item()
	
	def __repr__(self):
		return f"Parameter containing:\n{self.data}"
	
	def __str__(self):
		return self.__repr__()


class Module:
	def __init__(self):
		# Use object.__setattr__ to avoid recursion with __setattr__
		object.__setattr__(self, '_parameters', OrderedDict())
		object.__setattr__(self, '_modules', OrderedDict())
		object.__setattr__(self, '_buffers', OrderedDict())
		object.__setattr__(self, 'training', True)
		object.__setattr__(self, '_forward_hooks', [])
		object.__setattr__(self, '_backward_hooks', [])
	
	def forward(self, *args, **kwargs):
		raise NotImplementedError(
			f"{self.__class__.__name__} must implement forward() method"
		)
	
	def __call__(self, *args, **kwargs):
		# Run forward hooks (before forward pass)
		for hook in self._forward_hooks:
			hook(self, args, kwargs)
		
		# Execute forward pass
		output = self.forward(*args, **kwargs)
		
		# Run backward hooks (after forward pass)
		for hook in self._backward_hooks:
			hook(self, args, output)
		
		return output
	
	def __setattr__(self, name: str, value: Any) -> None:
		if not hasattr(self, '_parameters'):
			object.__setattr__(self, name, value)
			return

		params = self.__dict__['_parameters']
		modules = self.__dict__['_modules']
		buffers = self.__dict__['_buffers']

		if isinstance(value, Parameter):
			modules.pop(name, None)
			buffers.pop(name, None)
			params[name] = value
			object.__setattr__(self, name, value)
			return

		if isinstance(value, Module):
			params.pop(name, None)
			buffers.pop(name, None)
			modules[name] = value
			object.__setattr__(self, name, value)
			return

		if name in params:
			del params[name]
		if name in modules:
			del modules[name]
		if name in buffers:
			del buffers[name]

		object.__setattr__(self, name, value)
	def __getattr__(self, name: str) -> Any:
		if '_parameters' in self.__dict__:
			_parameters = self.__dict__['_parameters']
			if name in _parameters:
				return _parameters[name]
		
		if '_modules' in self.__dict__:
			_modules = self.__dict__['_modules']
			if name in _modules:
				return _modules[name]
		
		if '_buffers' in self.__dict__:
			_buffers = self.__dict__['_buffers']
			if name in _buffers:
				return _buffers[name]
		
		raise AttributeError(
			f"'{type(self).__name__}' object has no attribute '{name}'"
		)
	
	def __delattr__(self, name: str) -> None:
		if name in self._parameters:
			del self._parameters[name]
		elif name in self._modules:
			del self._modules[name]
		elif name in self._buffers:
			del self._buffers[name]
		else:
			object.__delattr__(self, name)
	
	def register_buffer(self, name: str, tensor, persistent: bool = True):
		from ..tensor import Tensor
		
		if not isinstance(tensor, Tensor):
			from ..tensor import Tensor
			tensor = Tensor(tensor, requires_grad=False)
		
		self._buffers[name] = tensor
	
	def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
		if param is None:
			if name in self._parameters:
				del self._parameters[name]
		elif not isinstance(param, Parameter):
			raise TypeError(
				f"register_parameter expects Parameter, got {type(param)}"
			)
		else:
			self._parameters[name] = param
	
	def add_module(self, name: str, module: Optional['Module']) -> None:
		if module is None:
			if name in self._modules:
				del self._modules[name]
		elif not isinstance(module, Module):
			raise TypeError(
				f"add_module expects Module, got {type(module)}"
			)
		else:
			self._modules[name] = module
	
	def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
		for name, param in self.named_parameters(recurse=recurse):
			yield param
	
	def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
		# Yield own parameters
		for name, param in self._parameters.items():
			full_name = f"{prefix}.{name}" if prefix else name
			yield full_name, param
		
		# Recursively yield submodule parameters
		if recurse:
			for name, module in self._modules.items():
				submodule_prefix = f"{prefix}.{name}" if prefix else name
				for sub_name, param in module.named_parameters(prefix=submodule_prefix, recurse=True):
					yield sub_name, param
	
	def buffers(self, recurse: bool = True) -> Iterator:
		for name, buf in self.named_buffers(recurse=recurse):
			yield buf
	
	def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Any]]:
		# Yield own buffers
		for name, buf in self._buffers.items():
			full_name = f"{prefix}.{name}" if prefix else name
			yield full_name, buf
		
		# Recursively yield submodule buffers
		if recurse:
			for name, module in self._modules.items():
				submodule_prefix = f"{prefix}.{name}" if prefix else name
				for sub_name, buf in module.named_buffers(prefix=submodule_prefix, recurse=True):
					yield sub_name, buf
	
	def children(self) -> Iterator['Module']:
		for name, module in self.named_children():
			yield module
	
	def named_children(self) -> Iterator[Tuple[str, 'Module']]:
		for name, module in self._modules.items():
			yield name, module
	
	def modules(self) -> Iterator['Module']:
		for name, module in self.named_modules():
			yield module
	
	def named_modules(self, memo: Optional[set] = None, prefix: str = '') -> Iterator[Tuple[str, 'Module']]:
		if memo is None:
			memo = set()
		
		if self not in memo:
			memo.add(self)
			yield prefix, self
			
			for name, module in self._modules.items():
				if module is None:
					continue
				submodule_prefix = f"{prefix}.{name}" if prefix else name
				for sub_name, sub_module in module.named_modules(memo=memo, prefix=submodule_prefix):
					yield sub_name, sub_module
	
	def train(self, mode: bool = True):
		self.training = mode
		
		# Recursively set submodules
		for module in self.children():
			module.train(mode)
		
		return self
	
	def eval(self):
		return self.train(False)
	
	def to(self, device: str):
		# Move parameters
		for param in self.parameters():
			param.to(device)
		
		# Move buffers
		for buf in self.buffers():
			buf.to(device)
		
		return self
	
	def cpu(self):
		return self.to('cpu')
	
	def cuda(self, device: int = 0):
		return self.to(f'cuda:{device}')
	
	def xpu(self, device: int = 0):
		return self.to(f'xpu:{device}')
	
	def zero_grad(self):
		for param in self.parameters():
			param.zero_grad()
	
	def state_dict(self, destination=None, prefix=''):
		if destination is None:
			destination = OrderedDict()
		
		# Save parameters
		for name, param in self._parameters.items():
			key = f"{prefix}{name}"
			destination[key] = param.data.numpy()  # Store as NumPy for portability
		
		# Save buffers
		for name, buf in self._buffers.items():
			key = f"{prefix}{name}"
			destination[key] = buf.numpy()
		
		# Recursively save submodules
		for name, module in self._modules.items():
			if module is not None:
				module.state_dict(destination, prefix=f"{prefix}{name}.")
		
		return destination
	
	def load_state_dict(self, state_dict, strict=True):
		from ..tensor import Tensor
		
		missing_keys = []
		unexpected_keys = []
		
		# Track which keys we've seen
		seen_keys = set()
		
		# Load parameters
		for name, param in self._parameters.items():
			key = name
			if key in state_dict:
				# Convert numpy back to Tensor
				data = state_dict[key]
				param.data.data = param.data._backend.asarray(data)
				seen_keys.add(key)
			else:
				missing_keys.append(key)
		
		# Load buffers
		for name, buf in self._buffers.items():
			key = name
			if key in state_dict:
				data = state_dict[key]
				buf.data = buf._backend.asarray(data)
				seen_keys.add(key)
			else:
				missing_keys.append(key)
		
		# Recursively load submodules
		for name, module in self._modules.items():
			if module is not None:
				# Extract submodule state
				prefix = f"{name}."
				submodule_state = OrderedDict()
				for key, value in state_dict.items():
					if key.startswith(prefix):
						subkey = key[len(prefix):]
						submodule_state[subkey] = value
						seen_keys.add(key)
				
				if submodule_state:
					module.load_state_dict(submodule_state, strict=strict)
		
		# Check for unexpected keys
		for key in state_dict.keys():
			if key not in seen_keys:
				unexpected_keys.append(key)
		
		# Report errors if strict mode
		if strict:
			error_msgs = []
			if missing_keys:
				error_msgs.append(f"Missing keys: {missing_keys}")
			if unexpected_keys:
				error_msgs.append(f"Unexpected keys: {unexpected_keys}")
			
			if error_msgs:
				raise RuntimeError(
					"Error(s) loading state_dict:\n" + "\n".join(error_msgs)
				)
	
	def apply(self, fn: Callable[['Module'], None]) -> 'Module':
		for module in self.children():
			module.apply(fn)
		fn(self)
		return self
	
	def register_forward_hook(self, hook: Callable) -> None:
		self._forward_hooks.append(hook)
	
	def register_backward_hook(self, hook: Callable) -> None:
		self._backward_hooks.append(hook)
	
	def __repr__(self):
		# Get submodules representation
		lines = []
		for key, module in self._modules.items():
			module_str = repr(module)
			module_str = self._add_indent(module_str, 2)
			lines.append(f"({key}): {module_str}")
		
		main_str = f"{self.__class__.__name__}("
		if lines:
			main_str += "\n  " + "\n  ".join(lines) + "\n"
		main_str += ")"
		
		return main_str
	
	def _add_indent(self, text: str, indent: int) -> str:
		lines = text.split('\n')
		if len(lines) == 1:
			return text
		first = lines[0]
		rest = '\n'.join(' ' * indent + line for line in lines[1:])
		return first + '\n' + rest
	
	def extra_repr(self) -> str:
		return ''


class Sequential(Module):
	def __init__(self, *args):
		super().__init__()
		
		if len(args) == 1 and isinstance(args[0], OrderedDict):
			# Sequential(OrderedDict([...]))
			for key, module in args[0].items():
				self.add_module(key, module)
		else:
			# Sequential(layer1, layer2, ...)
			for idx, module in enumerate(args):
				self.add_module(str(idx), module)
	
	def forward(self, x):
		for module in self._modules.values():
			x = module(x)
		return x
	
	def append(self, module: Module) -> 'Sequential':
		idx = len(self._modules)
		self.add_module(str(idx), module)
		return self
	
	def __getitem__(self, idx: Union[int, slice]) -> Union[Module, 'Sequential']:
		if isinstance(idx, slice):
			# Return new Sequential with sliced modules
			return Sequential(OrderedDict(list(self._modules.items())[idx]))
		else:
			# Return single module
			return list(self._modules.values())[idx]
	
	def __len__(self) -> int:
		return len(self._modules)
	
	def __iter__(self):
		return iter(self._modules.values())


class ModuleList(Module):
	def __init__(self, modules: Optional[List[Module]] = None):
		super().__init__()
		if modules is not None:
			self.extend(modules)
	
	def append(self, module: Module) -> 'ModuleList':
		idx = len(self._modules)
		self.add_module(str(idx), module)
		return self
	
	def extend(self, modules: List[Module]) -> 'ModuleList':
		for module in modules:
			self.append(module)
		return self
	
	def insert(self, index: int, module: Module) -> 'ModuleList':
		# Shift existing modules
		for i in range(len(self._modules), index, -1):
			self._modules[str(i)] = self._modules[str(i - 1)]
		self._modules[str(index)] = module
		return self
	
	def forward(self, *args, **kwargs):
		raise NotImplementedError(
			"ModuleList doesn't define forward(). "
			"Iterate over modules manually in your forward() method."
		)
	
	def __getitem__(self, idx: Union[int, slice]) -> Union[Module, 'ModuleList']:
		if isinstance(idx, slice):
			return ModuleList(list(self._modules.values())[idx])
		else:
			return list(self._modules.values())[idx]
	
	def __setitem__(self, idx: int, module: Module) -> None:
		self._modules[str(idx)] = module
	
	def __delitem__(self, idx: int) -> None:
		del self._modules[str(idx)]
		# Re-index remaining modules
		modules = list(self._modules.values())
		self._modules.clear()
		for i, module in enumerate(modules):
			self._modules[str(i)] = module
	
	def __len__(self) -> int:
		return len(self._modules)
	
	def __iter__(self):
		return iter(self._modules.values())


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
	
	def __iter__(self):
		return iter(self._modules.keys())
	
	def __contains__(self, key: str) -> bool:
		return key in self._modules
	
	def keys(self):
		return self._modules.keys()
	
	def values(self):
		return self._modules.values()
	
	def items(self):
		return self._modules.items()
	
	def update(self, modules: Dict[str, Module]) -> 'ModuleDict':
		for key, module in modules.items():
			self[key] = module
		return self
	
	def forward(self, *args, **kwargs):
		raise NotImplementedError(
			"ModuleDict doesn't define forward(). "
			"Access modules by key in your forward() method."
		)


class Identity(Module):
	def __init__(self):
		super().__init__()
	
	def forward(self, x):
		return x
	
	def __repr__(self):
		return f"{self.__class__.__name__}()"


# Utility Functions

def get_parameter_count(module: Module) -> Dict[str, int]:
	total = 0
	trainable = 0
	
	for param in module.parameters():
		param_count = 1
		for dim in param.shape:
			param_count *= dim
		
		total += param_count
		if param.requires_grad:
			trainable += param_count
	
	return {
		'total': total,
		'trainable': trainable,
		'non_trainable': total - trainable
	}


def freeze_module(module: Module) -> None:
	# Freeze all parameters (disable gradient computation).
	for param in module.parameters():
		param._requires_grad = False
		param.data._requires_grad = False


def unfreeze_module(module: Module) -> None:
	#Unfreeze all parameters (enable gradient computation).
	for param in module.parameters():
		param._requires_grad = True
		param.data._requires_grad = True


__all__ = [
	'Module',
	'Parameter',
	'Sequential',
	'ModuleList',
	'ModuleDict',
	'Identity',
	'get_parameter_count',
	'freeze_module',
	'unfreeze_module',
]
