from .tensor import Tensor
from . import dtype
from .engine import (
	# Arithmetic operations
	add, subtract, multiply, divide, power,
	negative, positive, floor_divide, remainder, mod,
	abs, absolute, sign,
	
	# Matrix operations
	matmul, dot, outer, inner,
	
	# Tensor operations
	transpose, reshape, squeeze, expand_dims,
	
	# Trigonometric functions
	sin, cos, tan, arcsin, arccos, arctan,
	sinh, cosh, tanh,
	
	# Exponential and logarithmic functions
	exp, log, log10, log2, sqrt, square,
	
	# Reduction operations
	sum, mean, max, min,
	
	# Comparison operations
	maximum, minimum, equal, greater, less,
	
	# Concatenation and stacking
	concatenate, stack,
	
	# Utility functions
	clip, where,
	
	# NEW - Modern deep learning operations
	softmax, log_softmax, gelu, silu,
	layer_norm, rms_norm, batch_norm, group_norm,
	dropout, embedding,
	permute, unsqueeze, split,
	gather, masked_fill
)

from . import utils
from . import distributed

from . import xpu
from . import cuda
from . import cpu

from .save_load import (
	save, load,
	save_state_dict, load_state_dict,
	save_checkpoint, load_checkpoint,
	get_model_size, save_model_info
)

__all__ = [
	'Tensor', 'dtype',
	'add', 'subtract', 'multiply', 'divide', 'power',
	'negative', 'positive', 'floor_divide', 'remainder', 'mod',
	'abs', 'absolute', 'sign',
	'matmul', 'dot', 'outer', 'inner',
	'transpose', 'reshape', 'squeeze', 'expand_dims',
	'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
	'sinh', 'cosh', 'tanh',
	'exp', 'log', 'log10', 'log2', 'sqrt', 'square',
	'sum', 'mean', 'max', 'min',
	'maximum', 'minimum', 'equal', 'greater', 'less',
	'concatenate', 'stack',
	'clip', 'where',
	# NEW operations
	'softmax', 'log_softmax', 'gelu', 'silu',
	'layer_norm', 'rms_norm', 'batch_norm', 'group_norm',
	'dropout', 'embedding',
        'permute', 'unsqueeze', 'split',
        'gather', 'masked_fill',
        'xpu', 'cuda', 'cpu', 'distributed'
]

__version__ = '0.5.3'
__all__.append('__version__')