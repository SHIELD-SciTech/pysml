from .module import Module, Parameter
import math


class Embedding(Module):
	
	def __init__(self, num_embeddings, embedding_dim, padding_idx=None, 
				 max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
		super().__init__()
		self.num_embeddings = num_embeddings
		self.embedding_dim = embedding_dim
		self.padding_idx = padding_idx
		self.max_norm = max_norm
		self.norm_type = norm_type
		self.scale_grad_by_freq = scale_grad_by_freq
		self.sparse = sparse
		
		# Initialize embedding table
		self.weight = Parameter(self._initialize_weight())
		
		# Zero out padding embedding if specified
		if padding_idx is not None:
			backend = self.weight.data._backend
			self.weight.data.data[padding_idx] = backend.zeros(embedding_dim)
	
	def _initialize_weight(self):
		# Normal initialization with std = 1.0
		from .. import Tensor
		import numpy as np
		
		data = np.random.randn(self.num_embeddings, self.embedding_dim)
		return Tensor(data, requires_grad=True)
	
	def forward(self, indices):
		# indices: (batch, seq_len) or any shape
		# output: (*indices.shape, embedding_dim)
		
		from .. import engine
		
		# Use optimized backend embedding lookup (zero-copy!)
		output = engine.embedding(self.weight.data, indices, self.padding_idx)
		
		# Apply max_norm constraint if specified
		if self.max_norm is not None:
			output = self._renorm_embeddings(output, indices)
		
		return output
	
	def _renorm_embeddings(self, embeddings, indices):
		# Constrain embedding norms to max_norm
		from .. import engine
		backend = embeddings._backend
		
		# Compute norms
		norms = backend.sqrt(
			backend.sum(backend.square(embeddings.data), axis=-1, keepdims=True)
		)
		
		# Clip norms
		scale = backend.minimum(norms, self.max_norm) / (norms + 1e-7)
		
		# Apply scaling
		from .. import Tensor
		scale_tensor = Tensor.__new__(Tensor)
		scale_tensor._backend = backend
		scale_tensor._dtype = embeddings._dtype
		scale_tensor.device = embeddings.device
		scale_tensor.active_device = embeddings.active_device
		scale_tensor.data = scale
		scale_tensor._requires_grad = False
		scale_tensor._grad = None
		
		return engine.multiply(embeddings, scale_tensor)
	
	def extra_repr(self):
		s = f"{self.num_embeddings}, {self.embedding_dim}"
		if self.padding_idx is not None:
			s += f", padding_idx={self.padding_idx}"
		if self.max_norm is not None:
			s += f", max_norm={self.max_norm}"
		if self.norm_type != 2.0:
			s += f", norm_type={self.norm_type}"
		if self.scale_grad_by_freq:
			s += ", scale_grad_by_freq=True"
		if self.sparse:
			s += ", sparse=True"
		return s
	
	@classmethod
	def from_pretrained(cls, embeddings, freeze=True, padding_idx=None, 
						max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
		# Create embedding from pretrained weights
		from .. import Tensor
		
		if not isinstance(embeddings, Tensor):
			embeddings = Tensor(embeddings, requires_grad=not freeze)
		
		num_embeddings, embedding_dim = embeddings.shape
		
		embedding = cls(
			num_embeddings, embedding_dim, padding_idx=padding_idx,
			max_norm=max_norm, norm_type=norm_type, 
			scale_grad_by_freq=scale_grad_by_freq, sparse=sparse
		)
		
		# Replace weights
		embedding.weight = Parameter(embeddings)
		
		# Freeze if requested
		if freeze:
			embedding.weight._requires_grad = False
			embedding.weight.data._requires_grad = False
		
		return embedding


class EmbeddingBag(Module):
	
	def __init__(self, num_embeddings, embedding_dim, max_norm=None, 
				 norm_type=2.0, scale_grad_by_freq=False, mode='mean', 
				 sparse=False, include_last_offset=False, padding_idx=None):
		super().__init__()
		self.num_embeddings = num_embeddings
		self.embedding_dim = embedding_dim
		self.max_norm = max_norm
		self.norm_type = norm_type
		self.scale_grad_by_freq = scale_grad_by_freq
		self.mode = mode  # 'sum', 'mean', or 'max'
		self.sparse = sparse
		self.include_last_offset = include_last_offset
		self.padding_idx = padding_idx
		
		# Initialize embedding table
		self.weight = Parameter(self._initialize_weight())
		
		# Zero out padding embedding if specified
		if padding_idx is not None:
			backend = self.weight.data._backend
			self.weight.data.data[padding_idx] = backend.zeros(embedding_dim)
	
	def _initialize_weight(self):
		from .. import Tensor
		import numpy as np
		
		data = np.random.randn(self.num_embeddings, self.embedding_dim)
		return Tensor(data, requires_grad=True)
	
	def forward(self, indices, offsets=None, per_sample_weights=None):
		# indices: (total_indices,) flattened indices
		# offsets: (batch,) start index for each bag
		# per_sample_weights: (total_indices,) optional weights for each embedding
		# output: (batch, embedding_dim)
		
		from .. import engine
		
		# Get embeddings for all indices
		embeddings = engine.embedding(self.weight.data, indices, self.padding_idx)
		
		# Apply per-sample weights if provided
		if per_sample_weights is not None:
			backend = embeddings._backend
			# Expand weights to match embedding dimension
			weights_expanded = backend.expand_dims(per_sample_weights.data, axis=-1)
			
			from .. import Tensor
			weights_tensor = Tensor.__new__(Tensor)
			weights_tensor._backend = backend
			weights_tensor._dtype = embeddings._dtype
			weights_tensor.device = embeddings.device
			weights_tensor.active_device = embeddings.active_device
			weights_tensor.data = weights_expanded
			weights_tensor._requires_grad = False
			weights_tensor._grad = None
			
			embeddings = engine.multiply(embeddings, weights_tensor)
		
		# Pool embeddings according to offsets
		if offsets is not None:
			output = self._pool_by_offsets(embeddings, offsets)
		else:
			# No offsets: treat as single bag
			output = self._pool_all(embeddings)
		
		return output
	
	def _pool_by_offsets(self, embeddings, offsets):
		# Pool embeddings into bags using offsets
		from .. import Tensor
		backend = embeddings._backend
		
		# Convert offsets to indices
		batch_size = len(offsets.data)
		if self.include_last_offset:
			batch_size -= 1
		
		# Create output tensor
		import numpy as np
		if backend.BACKEND_NAME == 'cpu':
			output_data = np.zeros((batch_size, self.embedding_dim))
		else:
			output_data = backend.zeros((batch_size, self.embedding_dim))
		
		# Pool each bag
		embeddings_np = embeddings.numpy()
		offsets_np = offsets.numpy() if hasattr(offsets, 'numpy') else offsets.data
		
		for i in range(batch_size):
			start = int(offsets_np[i])
			end = int(offsets_np[i + 1]) if i + 1 < len(offsets_np) else len(embeddings_np)
			
			bag_embeddings = embeddings_np[start:end]
			
			if len(bag_embeddings) == 0:
				continue
			
			if self.mode == 'sum':
				output_data[i] = np.sum(bag_embeddings, axis=0)
			elif self.mode == 'mean':
				output_data[i] = np.mean(bag_embeddings, axis=0)
			elif self.mode == 'max':
				output_data[i] = np.max(bag_embeddings, axis=0)
		
		# Convert back to tensor
		output = Tensor.__new__(Tensor)
		output._backend = backend
		output._dtype = embeddings._dtype
		output.device = embeddings.device
		output.active_device = embeddings.active_device
		output.data = backend.asarray(output_data)
		output._requires_grad = embeddings._requires_grad
		output._grad = None
		
		return output
	
	def _pool_all(self, embeddings):
		# Pool all embeddings into single vector
		from .. import engine
		
		if self.mode == 'sum':
			return engine.sum_with_grad(embeddings, axis=0, keepdims=False)
		elif self.mode == 'mean':
			return engine.mean_with_grad(embeddings, axis=0, keepdims=False)
		elif self.mode == 'max':
			backend = embeddings._backend
			from .. import Tensor
			output = Tensor.__new__(Tensor)
			output._backend = backend
			output._dtype = embeddings._dtype
			output.device = embeddings.device
			output.active_device = embeddings.active_device
			output.data = backend.max(embeddings.data, axis=0, keepdims=False)
			output._requires_grad = embeddings._requires_grad
			output._grad = None
			return output
	
	def extra_repr(self):
		s = f"{self.num_embeddings}, {self.embedding_dim}"
		s += f", mode='{self.mode}'"
		if self.max_norm is not None:
			s += f", max_norm={self.max_norm}"
		if self.padding_idx is not None:
			s += f", padding_idx={self.padding_idx}"
		return s


# Utility functions for embeddings

def _no_grad_embedding_renorm_(embeddings, norm, max_norm):
	# Renormalize embeddings to have max norm
	# Used internally, not exposed
	from .. import Tensor
	import numpy as np
	
	# Compute current norms
	backend = embeddings._backend
	norms = backend.sqrt(
		backend.sum(backend.square(embeddings), axis=-1, keepdims=True)
	)
	
	# Find embeddings that exceed max_norm
	mask = norms > max_norm
	
	# Scale down those embeddings
	scale = max_norm / (norms + 1e-7)
	embeddings[mask] = embeddings[mask] * scale[mask]


__all__ = [
	'Embedding',
	'EmbeddingBag',
]