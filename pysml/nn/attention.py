from .module import Module, Parameter
from .linear import Linear
from .dropout import Dropout
import math


class MultiHeadAttention(Module):
	
	def __init__(self, d_model, num_heads, dropout=0.0, bias=True, 
				 add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
		super().__init__()
		self.d_model = d_model
		self.num_heads = num_heads
		self.dropout_p = dropout
		
		# Key and value dimensions (default to d_model)
		self.kdim = kdim if kdim is not None else d_model
		self.vdim = vdim if vdim is not None else d_model
		
		# Head dimension
		assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
		self.head_dim = d_model // num_heads
		self.scale = 1.0 / math.sqrt(self.head_dim)
		
		# Linear projections for Q, K, V
		self.q_proj = Linear(d_model, d_model, bias=bias)
		self.k_proj = Linear(self.kdim, d_model, bias=bias)
		self.v_proj = Linear(self.vdim, d_model, bias=bias)
		
		# Output projection
		self.out_proj = Linear(d_model, d_model, bias=bias)
		
		# Dropout
		if dropout > 0:
			self.dropout = Dropout(dropout)
		else:
			self.dropout = None
		
		# Optional bias for K, V
		self.add_bias_kv = add_bias_kv
		if add_bias_kv:
			self.bias_k = Parameter(self._initialize_bias())
			self.bias_v = Parameter(self._initialize_bias())
		
		self.add_zero_attn = add_zero_attn
	
	def _initialize_bias(self):
		from .. import Tensor
		import numpy as np
		data = np.zeros((1, 1, self.d_model))
		return Tensor(data, requires_grad=True)
	
	def forward(self, query, key=None, value=None, attn_mask=None, 
				key_padding_mask=None, need_weights=False):
		# query: (batch, seq_len, d_model)
		# key: (batch, src_len, kdim) - if None, uses query (self-attention)
		# value: (batch, src_len, vdim) - if None, uses query
		# attn_mask: (seq_len, src_len) or (batch, seq_len, src_len)
		# key_padding_mask: (batch, src_len) - True for positions to mask
		
		from .. import engine
		
		# Use query for key/value if not provided (self-attention)
		if key is None:
			key = query
		if value is None:
			value = query
		
		batch_size = query.shape[0]
		seq_len = query.shape[1]
		src_len = key.shape[1]
		
		# Linear projections
		Q = self.q_proj(query)  # (batch, seq_len, d_model)
		K = self.k_proj(key)	# (batch, src_len, d_model)
		V = self.v_proj(value)  # (batch, src_len, d_model)
		
		# Reshape for multi-head attention
		# (batch, seq_len, d_model) -> (batch, seq_len, num_heads, head_dim)
		# -> (batch, num_heads, seq_len, head_dim)
		Q = self._split_heads(Q, batch_size, seq_len)
		K = self._split_heads(K, batch_size, src_len)
		V = self._split_heads(V, batch_size, src_len)
		
		# Compute attention scores
		# (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, src_len)
		# = (batch, num_heads, seq_len, src_len)
		scores = engine.matmul(Q, K.T())
		
		# Scale scores
		scores = engine.multiply(scores, self.scale)
		
		# Apply attention mask if provided
		if attn_mask is not None:
			scores = self._apply_attention_mask(scores, attn_mask)
		
		# Apply key padding mask if provided
		if key_padding_mask is not None:
			scores = self._apply_key_padding_mask(scores, key_padding_mask)
		
		# Apply softmax to get attention weights
		attn_weights = engine.softmax(scores, axis=-1)
		
		# Apply dropout to attention weights
		if self.dropout is not None:
			attn_weights = self.dropout(attn_weights)
		
		# Apply attention to values
		# (batch, num_heads, seq_len, src_len) @ (batch, num_heads, src_len, head_dim)
		# = (batch, num_heads, seq_len, head_dim)
		attn_output = engine.matmul(attn_weights, V)
		
		# Reshape back to (batch, seq_len, d_model)
		attn_output = self._combine_heads(attn_output, batch_size, seq_len)
		
		# Output projection
		output = self.out_proj(attn_output)
		
		if need_weights:
			return output, attn_weights
		return output
	
	def _split_heads(self, x, batch_size, seq_len):
		# (batch, seq, d_model) -> (batch, seq, num_heads, head_dim)
		# -> (batch, num_heads, seq, head_dim)
		
		backend = x._backend
		
		# Reshape
		x_reshaped = x.reshape((batch_size, seq_len, self.num_heads, self.head_dim))
		
		# Transpose: (batch, seq, num_heads, head_dim) -> (batch, num_heads, seq, head_dim)
		from .. import engine
		return engine.permute(x_reshaped, (0, 2, 1, 3))
	
	def _combine_heads(self, x, batch_size, seq_len):
		# (batch, num_heads, seq, head_dim) -> (batch, seq, d_model)
		
		from .. import engine
		
		# Transpose: (batch, num_heads, seq, head_dim) -> (batch, seq, num_heads, head_dim)
		x_transposed = engine.permute(x, (0, 2, 1, 3))
		
		# Reshape
		return x_transposed.reshape((batch_size, seq_len, self.d_model))
	
	def _apply_attention_mask(self, scores, mask):
		# Apply attention mask (usually for causal attention)
		from .. import engine
		backend = scores._backend
		
		# Expand mask to match scores shape if needed
		if len(mask.shape) == 2:
			# (seq_len, src_len) -> (1, 1, seq_len, src_len)
			mask_data = backend.expand_dims(backend.expand_dims(mask.data, 0), 0)
		else:
			mask_data = mask.data
		
		# Create large negative value for masked positions
		# This ensures softmax gives ~0 probability
		from .. import Tensor
		mask_tensor = Tensor.__new__(Tensor)
		mask_tensor._backend = backend
		mask_tensor._dtype = scores._dtype
		mask_tensor.device = scores.device
		mask_tensor.active_device = scores.active_device
		mask_tensor.data = mask_data
		mask_tensor._requires_grad = False
		mask_tensor._grad = None
		
		# Where mask is True, replace with large negative value
		large_neg = -1e9
		return engine.where(mask_tensor, large_neg, scores)
	
	def _apply_key_padding_mask(self, scores, key_padding_mask):
		# key_padding_mask: (batch, src_len) - True for positions to ignore
		from .. import engine
		backend = scores._backend
		
		# Expand to (batch, 1, 1, src_len)
		mask_expanded = backend.expand_dims(
			backend.expand_dims(key_padding_mask.data, 1), 1
		)
		
		from .. import Tensor
		mask_tensor = Tensor.__new__(Tensor)
		mask_tensor._backend = backend
		mask_tensor._dtype = scores._dtype
		mask_tensor.device = scores.device
		mask_tensor.active_device = scores.active_device
		mask_tensor.data = mask_expanded
		mask_tensor._requires_grad = False
		mask_tensor._grad = None
		
		# Mask out padded positions
		large_neg = -1e9
		return engine.where(mask_tensor, large_neg, scores)
	
	def extra_repr(self):
		return f"d_model={self.d_model}, num_heads={self.num_heads}, dropout={self.dropout_p}"


class MultiHeadSelfAttention(Module):
	
	def __init__(self, d_model, num_heads, dropout=0.0, bias=True):
		super().__init__()
		self.attention = MultiHeadAttention(
			d_model, num_heads, dropout=dropout, bias=bias
		)
	
	def forward(self, x, attn_mask=None, key_padding_mask=None, need_weights=False):
		# Self-attention: query, key, value all come from x
		return self.attention(x, x, x, attn_mask, key_padding_mask, need_weights)
	
	def extra_repr(self):
		return self.attention.extra_repr()


class CrossAttention(Module):
	
	def __init__(self, d_model, num_heads, dropout=0.0, bias=True, kdim=None, vdim=None):
		super().__init__()
		self.attention = MultiHeadAttention(
			d_model, num_heads, dropout=dropout, bias=bias,
			kdim=kdim, vdim=vdim
		)
	
	def forward(self, query, key, value, attn_mask=None, 
				key_padding_mask=None, need_weights=False):
		# Cross-attention: query from decoder, key/value from encoder
		return self.attention(query, key, value, attn_mask, 
							  key_padding_mask, need_weights)
	
	def extra_repr(self):
		return self.attention.extra_repr()


# Utility functions

def create_causal_mask(seq_len, device='cpu'):
	# Create causal mask for autoregressive attention
	# Upper triangular matrix of True values
	import numpy as np
	from .. import Tensor
	
	mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
	
	tensor = Tensor(mask, requires_grad=False)
	tensor.to(device)
	
	return tensor


def create_padding_mask(lengths, max_len=None, device='cpu'):
	# Create padding mask from sequence lengths
	# lengths: (batch,) actual lengths of each sequence
	import numpy as np
	from .. import Tensor
	
	batch_size = len(lengths)
	if max_len is None:
		max_len = int(np.max(lengths))
	
	# Create mask: True for padding positions
	mask = np.zeros((batch_size, max_len), dtype=bool)
	for i, length in enumerate(lengths):
		if length < max_len:
			mask[i, length:] = True
	
	tensor = Tensor(mask, requires_grad=False)
	tensor.to(device)
	
	return tensor


__all__ = [
	'MultiHeadAttention',
	'MultiHeadSelfAttention',
	'CrossAttention',
	'create_causal_mask',
	'create_padding_mask',
]