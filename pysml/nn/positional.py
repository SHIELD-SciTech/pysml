from .module import Module, Parameter
import math


class SinusoidalPositionalEncoding(Module):
    
    def __init__(self, d_model, max_len=5000, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        if dropout > 0:
            from .dropout import Dropout
            self.dropout = Dropout(dropout)
        else:
            self.dropout = None
        
        # Pre-compute sinusoidal encodings
        self.register_buffer('pe', self._create_sinusoidal_encodings())
    
    def _create_sinusoidal_encodings(self):
        # Create fixed sinusoidal position encodings
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        from .. import Tensor
        import numpy as np
        
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len, dtype=np.float32).reshape(-1, 1)
        
        # Compute division term
        div_term = np.exp(
            np.arange(0, self.d_model, 2, dtype=np.float32) * 
            -(math.log(10000.0) / self.d_model)
        )
        
        # Apply sin to even indices
        pe[:, 0::2] = np.sin(position * div_term)
        
        # Apply cos to odd indices
        if self.d_model % 2 == 0:
            pe[:, 1::2] = np.cos(position * div_term)
        else:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        
        return Tensor(pe, requires_grad=False)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model) or (seq_len, batch, d_model)
        from .. import engine
        
        seq_len = x.shape[1] if len(x.shape) == 3 else x.shape[0]
        
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_len}"
            )
        
        # Get positional encodings for this sequence
        backend = x._backend
        pe_slice = self.pe.data[:seq_len]
        
        # Expand to match batch size if needed
        if len(x.shape) == 3:
            # (batch, seq, d_model)
            pe_expanded = backend.expand_dims(pe_slice, axis=0)
            # Broadcast to batch size
            from .. import Tensor
            pe_tensor = Tensor.__new__(Tensor)
            pe_tensor._backend = backend
            pe_tensor._dtype = x._dtype
            pe_tensor.device = x.device
            pe_tensor.active_device = x.active_device
            pe_tensor.data = pe_expanded
            pe_tensor._requires_grad = False
            pe_tensor._grad = None
        else:
            # (seq, batch, d_model) - unsupported for now
            pe_tensor = self.pe[:seq_len]
        
        # Add positional encoding to input
        output = engine.add(x, pe_tensor)
        
        # Apply dropout if present
        if self.dropout is not None:
            output = self.dropout(output)
        
        return output
    
    def extra_repr(self):
        return f"d_model={self.d_model}, max_len={self.max_len}"


class LearnedPositionalEmbedding(Module):
    
    def __init__(self, max_positions, embedding_dim, padding_idx=None):
        super().__init__()
        self.max_positions = max_positions
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        # Learnable position embeddings
        from .embedding import Embedding
        self.embedding = Embedding(max_positions, embedding_dim, padding_idx=padding_idx)
    
    def forward(self, x, positions=None):
        # x: (batch, seq_len, d_model)
        # positions: (batch, seq_len) optional position indices
        
        from .. import Tensor
        import numpy as np
        
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        if positions is None:
            # Create default positions: 0, 1, 2, ..., seq_len-1
            positions = np.arange(seq_len)
            positions = np.tile(positions, (batch_size, 1))
            
            backend = x._backend
            positions_tensor = Tensor.__new__(Tensor)
            positions_tensor._backend = backend
            positions_tensor._dtype = x._dtype
            positions_tensor.device = x.device
            positions_tensor.active_device = x.active_device
            positions_tensor.data = backend.asarray(positions)
            positions_tensor._requires_grad = False
            positions_tensor._grad = None
        else:
            positions_tensor = positions
        
        # Get position embeddings
        pos_embeddings = self.embedding(positions_tensor)
        
        # Add to input
        from .. import engine
        return engine.add(x, pos_embeddings)
    
    def extra_repr(self):
        return f"max_positions={self.max_positions}, embedding_dim={self.embedding_dim}"


class RotaryPositionalEmbedding(Module):
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Pre-compute rotation matrices
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        self.register_buffer('inv_freq', Tensor(inv_freq, requires_grad=False))
        
        # Cache for cos/sin values
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
    
    def _update_cos_sin_cache(self, seq_len):
        # Update cached cos/sin values if sequence length changed
        if seq_len != self._seq_len_cached or self._cos_cached is None:
            import numpy as np
            from .. import Tensor
            
            self._seq_len_cached = seq_len
            
            # Compute position indices
            t = np.arange(seq_len, dtype=np.float32)
            
            # Compute frequencies
            inv_freq_np = self.inv_freq.numpy()
            freqs = np.outer(t, inv_freq_np)  # (seq_len, dim//2)
            
            # Duplicate for both parts of each dimension pair
            emb = np.concatenate([freqs, freqs], axis=-1)  # (seq_len, dim)
            
            # Compute cos and sin
            backend = self.inv_freq._backend
            self._cos_cached = Tensor(np.cos(emb), requires_grad=False)
            self._sin_cached = Tensor(np.sin(emb), requires_grad=False)
            
            # Move to same device as inv_freq
            self._cos_cached.to(self.inv_freq.active_device)
            self._sin_cached.to(self.inv_freq.active_device)
    
    def rotate_half(self, x):
        # Rotate half the hidden dims of the input
        # Split x into two halves and swap with negation
        backend = x._backend
        
        # x: (..., dim)
        x1 = x.data[..., :self.dim // 2]
        x2 = x.data[..., self.dim // 2:]
        
        # Concatenate [-x2, x1]
        from .. import Tensor
        import numpy as np
        
        x_np = x.numpy()
        x1_np = x_np[..., :self.dim // 2]
        x2_np = x_np[..., self.dim // 2:]
        
        rotated = np.concatenate([-x2_np, x1_np], axis=-1)
        
        output = Tensor.__new__(Tensor)
        output._backend = backend
        output._dtype = x._dtype
        output.device = x.device
        output.active_device = x.active_device
        output.data = backend.asarray(rotated)
        output._requires_grad = x._requires_grad
        output._grad = None
        
        return output
    
    def forward(self, q, k, seq_len=None):
        # Apply rotary embeddings to query and key
        # q, k: (batch, seq_len, num_heads, head_dim)
        
        if seq_len is None:
            seq_len = q.shape[1]
        
        # Update cos/sin cache
        self._update_cos_sin_cache(seq_len)
        
        from .. import engine
        
        # Get cos and sin for this sequence length
        cos = self._cos_cached[:seq_len]
        sin = self._sin_cached[:seq_len]
        
        # Expand dimensions to match q, k
        # cos/sin: (seq_len, dim) -> (1, seq_len, 1, dim)
        backend = q._backend
        cos_expanded = backend.expand_dims(backend.expand_dims(cos.data, 0), 2)
        sin_expanded = backend.expand_dims(backend.expand_dims(sin.data, 0), 2)
        
        from .. import Tensor
        cos_tensor = Tensor.__new__(Tensor)
        cos_tensor._backend = backend
        cos_tensor._dtype = q._dtype
        cos_tensor.device = q.device
        cos_tensor.active_device = q.active_device
        cos_tensor.data = cos_expanded
        cos_tensor._requires_grad = False
        cos_tensor._grad = None
        
        sin_tensor = Tensor.__new__(Tensor)
        sin_tensor._backend = backend
        sin_tensor._dtype = q._dtype
        sin_tensor.device = q.device
        sin_tensor.active_device = q.active_device
        sin_tensor.data = sin_expanded
        sin_tensor._requires_grad = False
        sin_tensor._grad = None
        
        # Apply rotation: x * cos + rotate_half(x) * sin
        q_embed = engine.add(
            engine.multiply(q, cos_tensor),
            engine.multiply(self.rotate_half(q), sin_tensor)
        )
        k_embed = engine.add(
            engine.multiply(k, cos_tensor),
            engine.multiply(self.rotate_half(k), sin_tensor)
        )
        
        return q_embed, k_embed
    
    def extra_repr(self):
        return f"dim={self.dim}, max_position_embeddings={self.max_position_embeddings}, base={self.base}"


class ALiBiPositionalBias(Module):
    
    def __init__(self, num_heads, max_seq_len=2048):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Compute ALiBi slopes
        slopes = self._get_alibi_slopes(num_heads)
        self.register_buffer('slopes', Tensor(slopes, requires_grad=False))
        
        # Cache bias matrix
        self._cached_bias = None
        self._cached_seq_len = 0
    
    def _get_alibi_slopes(self, num_heads):
        # Compute slopes for ALiBi
        # For n heads, slopes are: 2^(-8/n), 2^(-16/n), ..., 2^(-8)
        import numpy as np
        
        def get_slopes_power_of_2(n):
            start = 2 ** (-8)
            ratio = start ** (1.0 / n)
            return np.array([start * (ratio ** i) for i in range(n)])
        
        # Handle non-power-of-2 number of heads
        if (num_heads & (num_heads - 1)) == 0:  # Check if power of 2
            return get_slopes_power_of_2(num_heads)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            
            # Interpolate for remaining heads
            extra = num_heads - closest_power_of_2
            extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)[::2][:extra]
            slopes = np.concatenate([slopes, extra_slopes])
            
            return slopes
    
    def _build_alibi_bias(self, seq_len):
        # Build ALiBi bias matrix: -slopes * |i - j|
        import numpy as np
        
        # Create position difference matrix
        positions = np.arange(seq_len)
        pos_diff = np.abs(positions[:, None] - positions[None, :])  # (seq, seq)
        
        # Apply slopes
        slopes_np = self.slopes.numpy()
        bias = -slopes_np[:, None, None] * pos_diff[None, :, :]  # (heads, seq, seq)
        
        return bias
    
    def forward(self, attention_scores):
        # attention_scores: (batch, num_heads, seq_len, seq_len)
        # Returns: attention_scores + alibi_bias
        
        seq_len = attention_scores.shape[-1]
        
        # Update cache if needed
        if seq_len != self._cached_seq_len or self._cached_bias is None:
            self._cached_seq_len = seq_len
            bias = self._build_alibi_bias(seq_len)
            
            from .. import Tensor
            self._cached_bias = Tensor(bias, requires_grad=False)
            self._cached_bias.to(attention_scores.active_device)
        
        # Add bias to attention scores
        from .. import engine
        backend = attention_scores._backend
        
        # Expand bias to match batch size
        bias_expanded = backend.expand_dims(self._cached_bias.data, 0)
        
        from .. import Tensor
        bias_tensor = Tensor.__new__(Tensor)
        bias_tensor._backend = backend
        bias_tensor._dtype = attention_scores._dtype
        bias_tensor.device = attention_scores.device
        bias_tensor.active_device = attention_scores.active_device
        bias_tensor.data = bias_expanded
        bias_tensor._requires_grad = False
        bias_tensor._grad = None
        
        return engine.add(attention_scores, bias_tensor)
    
    def extra_repr(self):
        return f"num_heads={self.num_heads}, max_seq_len={self.max_seq_len}"


class AbsolutePositionalEmbedding(Module):
    
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # Simple learned absolute positions
        self.weight = Parameter(self._initialize_weight())
    
    def _initialize_weight(self):
        from .. import Tensor
        import numpy as np
        
        # Normal initialization
        data = np.random.randn(self.max_seq_len, self.d_model) * 0.02
        return Tensor(data, requires_grad=True)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        from .. import engine
        
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max {self.max_seq_len}")
        
        # Get position embeddings
        backend = x._backend
        pos_emb = self.weight.data[:seq_len]
        
        # Expand to batch dimension
        pos_emb_expanded = backend.expand_dims(pos_emb, 0)
        
        from .. import Tensor
        pos_tensor = Tensor.__new__(Tensor)
        pos_tensor._backend = backend
        pos_tensor._dtype = x._dtype
        pos_tensor.device = x.device
        pos_tensor.active_device = x.active_device
        pos_tensor.data = pos_emb_expanded
        pos_tensor._requires_grad = False
        pos_tensor._grad = None
        
        return engine.add(x, pos_tensor)
    
    def extra_repr(self):
        return f"max_seq_len={self.max_seq_len}, d_model={self.d_model}"


import numpy as np


__all__ = [
    'SinusoidalPositionalEncoding',
    'LearnedPositionalEmbedding',
    'RotaryPositionalEmbedding',
    'ALiBiPositionalBias',
    'AbsolutePositionalEmbedding',
]