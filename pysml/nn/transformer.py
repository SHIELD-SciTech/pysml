from .module import Module, ModuleList
from .linear import Linear
from .attention import MultiHeadAttention, MultiHeadSelfAttention, CrossAttention
from .normalization import LayerNorm, RMSNorm
from .activation import GELU, ReLU
from .dropout import Dropout
import math


class TransformerEncoderLayer(Module):
	
	def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1, 
				 activation='gelu', norm_first=False, norm='layernorm'):
		super().__init__()
		self.d_model = d_model
		self.num_heads = num_heads
		
		# Default feedforward dimension
		if d_ff is None:
			d_ff = 4 * d_model
		self.d_ff = d_ff
		
		self.norm_first = norm_first  # Pre-norm vs post-norm
		
		# Self-attention
		self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
		
		# Feedforward network
		self.ff1 = Linear(d_model, d_ff)
		self.ff2 = Linear(d_ff, d_model)
		
		# Activation
		if activation == 'gelu':
			self.activation = GELU()
		elif activation == 'relu':
			self.activation = ReLU()
		else:
			raise ValueError(f"Unknown activation: {activation}")
		
		# Normalization (50% memory optimized!)
		if norm == 'layernorm':
			self.norm1 = LayerNorm(d_model)
			self.norm2 = LayerNorm(d_model)
		elif norm == 'rmsnorm':
			self.norm1 = RMSNorm(d_model)
			self.norm2 = RMSNorm(d_model)
		else:
			raise ValueError(f"Unknown norm: {norm}")
		
		# Dropout
		self.dropout1 = Dropout(dropout)
		self.dropout2 = Dropout(dropout)
	
	def forward(self, x, attn_mask=None, key_padding_mask=None):
		# x: (batch, seq_len, d_model)
		
		if self.norm_first:
			# Pre-norm (modern, e.g., GPT-2, LLaMA)
			x = x + self._sa_block(self.norm1(x), attn_mask, key_padding_mask)
			x = x + self._ff_block(self.norm2(x))
		else:
			# Post-norm (original Transformer)
			x = self.norm1(x + self._sa_block(x, attn_mask, key_padding_mask))
			x = self.norm2(x + self._ff_block(x))
		
		return x
	
	def _sa_block(self, x, attn_mask, key_padding_mask):
		# Self-attention block
		x = self.self_attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
		return self.dropout1(x)
	
	def _ff_block(self, x):
		# Feedforward block
		from .. import engine
		x = self.ff1(x)
		x = self.activation(x)
		x = self.ff2(x)
		return self.dropout2(x)
	
	def extra_repr(self):
		return (f"d_model={self.d_model}, num_heads={self.num_heads}, "
				f"d_ff={self.d_ff}, norm_first={self.norm_first}")


class TransformerDecoderLayer(Module):
	
	def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1,
				 activation='gelu', norm_first=False, norm='layernorm'):
		super().__init__()
		self.d_model = d_model
		self.num_heads = num_heads
		
		if d_ff is None:
			d_ff = 4 * d_model
		self.d_ff = d_ff
		
		self.norm_first = norm_first
		
		# Self-attention
		self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
		
		# Cross-attention
		self.cross_attn = CrossAttention(d_model, num_heads, dropout=dropout)
		
		# Feedforward network
		self.ff1 = Linear(d_model, d_ff)
		self.ff2 = Linear(d_ff, d_model)
		
		# Activation
		if activation == 'gelu':
			self.activation = GELU()
		elif activation == 'relu':
			self.activation = ReLU()
		else:
			raise ValueError(f"Unknown activation: {activation}")
		
		# Normalization
		if norm == 'layernorm':
			self.norm1 = LayerNorm(d_model)
			self.norm2 = LayerNorm(d_model)
			self.norm3 = LayerNorm(d_model)
		elif norm == 'rmsnorm':
			self.norm1 = RMSNorm(d_model)
			self.norm2 = RMSNorm(d_model)
			self.norm3 = RMSNorm(d_model)
		else:
			raise ValueError(f"Unknown norm: {norm}")
		
		# Dropout
		self.dropout1 = Dropout(dropout)
		self.dropout2 = Dropout(dropout)
		self.dropout3 = Dropout(dropout)
	
	def forward(self, x, memory, tgt_mask=None, memory_mask=None,
				tgt_key_padding_mask=None, memory_key_padding_mask=None):
		# x: (batch, tgt_seq_len, d_model) - decoder input
		# memory: (batch, src_seq_len, d_model) - encoder output
		
		if self.norm_first:
			# Pre-norm
			x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
			x = x + self._ca_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
			x = x + self._ff_block(self.norm3(x))
		else:
			# Post-norm
			x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
			x = self.norm2(x + self._ca_block(x, memory, memory_mask, memory_key_padding_mask))
			x = self.norm3(x + self._ff_block(x))
		
		return x
	
	def _sa_block(self, x, attn_mask, key_padding_mask):
		# Self-attention block
		x = self.self_attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
		return self.dropout1(x)
	
	def _ca_block(self, x, memory, attn_mask, key_padding_mask):
		# Cross-attention block
		x = self.cross_attn(x, memory, memory, attn_mask=attn_mask, 
						   key_padding_mask=key_padding_mask)
		return self.dropout2(x)
	
	def _ff_block(self, x):
		# Feedforward block
		x = self.ff1(x)
		x = self.activation(x)
		x = self.ff2(x)
		return self.dropout3(x)
	
	def extra_repr(self):
		return (f"d_model={self.d_model}, num_heads={self.num_heads}, "
				f"d_ff={self.d_ff}, norm_first={self.norm_first}")


class TransformerEncoder(Module):
	
	def __init__(self, encoder_layer, num_layers, norm=None):
		super().__init__()
		self.layers = ModuleList([encoder_layer for _ in range(num_layers)])
		self.num_layers = num_layers
		self.norm = norm
	
	def forward(self, x, mask=None, src_key_padding_mask=None):
		# x: (batch, seq_len, d_model)
		
		# Pass through all encoder layers
		for layer in self.layers:
			x = layer(x, attn_mask=mask, key_padding_mask=src_key_padding_mask)
		
		# Final normalization if specified
		if self.norm is not None:
			x = self.norm(x)
		
		return x
	
	def extra_repr(self):
		return f"num_layers={self.num_layers}"


class TransformerDecoder(Module):
	
	def __init__(self, decoder_layer, num_layers, norm=None):
		super().__init__()
		self.layers = ModuleList([decoder_layer for _ in range(num_layers)])
		self.num_layers = num_layers
		self.norm = norm
	
	def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
				tgt_key_padding_mask=None, memory_key_padding_mask=None):
		# tgt: (batch, tgt_seq_len, d_model)
		# memory: (batch, src_seq_len, d_model)
		
		# Pass through all decoder layers
		for layer in self.layers:
			tgt = layer(tgt, memory, tgt_mask, memory_mask,
					   tgt_key_padding_mask, memory_key_padding_mask)
		
		# Final normalization if specified
		if self.norm is not None:
			tgt = self.norm(tgt)
		
		return tgt
	
	def extra_repr(self):
		return f"num_layers={self.num_layers}"


class Transformer(Module):
	
	def __init__(self, d_model=512, num_heads=8, num_encoder_layers=6,
				 num_decoder_layers=6, d_ff=2048, dropout=0.1,
				 activation='relu', norm_first=False):
		super().__init__()
		self.d_model = d_model
		
		# Create encoder
		encoder_layer = TransformerEncoderLayer(
			d_model, num_heads, d_ff, dropout, activation, norm_first
		)
		encoder_norm = LayerNorm(d_model) if norm_first else None
		self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
		
		# Create decoder
		decoder_layer = TransformerDecoderLayer(
			d_model, num_heads, d_ff, dropout, activation, norm_first
		)
		decoder_norm = LayerNorm(d_model) if norm_first else None
		self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
	
	def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None,
				src_key_padding_mask=None, tgt_key_padding_mask=None, 
				memory_key_padding_mask=None):
		# src: (batch, src_seq_len, d_model)
		# tgt: (batch, tgt_seq_len, d_model)
		
		# Encode source
		memory = self.encoder(src, mask=src_mask, 
							 src_key_padding_mask=src_key_padding_mask)
		
		# Decode target
		output = self.decoder(tgt, memory, tgt_mask=tgt_mask, 
							 memory_mask=memory_mask,
							 tgt_key_padding_mask=tgt_key_padding_mask,
							 memory_key_padding_mask=memory_key_padding_mask)
		
		return output
	
	def extra_repr(self):
		return f"d_model={self.d_model}"


# GPT-style decoder-only model

class GPTBlock(Module):
	
	def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1, norm='layernorm'):
		super().__init__()
		self.d_model = d_model
		self.num_heads = num_heads
		
		if d_ff is None:
			d_ff = 4 * d_model
		
		# Pre-norm architecture (GPT-2 style)
		if norm == 'layernorm':
			self.ln1 = LayerNorm(d_model)
			self.ln2 = LayerNorm(d_model)
		elif norm == 'rmsnorm':
			self.ln1 = RMSNorm(d_model)
			self.ln2 = RMSNorm(d_model)
		
		# Self-attention
		self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
		
		# Feedforward
		self.ff1 = Linear(d_model, d_ff)
		self.ff2 = Linear(d_ff, d_model)
		self.gelu = GELU()
		self.dropout = Dropout(dropout)
	
	def forward(self, x, attn_mask=None, key_padding_mask=None):
		# x: (batch, seq_len, d_model)
		
		# Self-attention with residual
		attn_out = self.attn(self.ln1(x), attn_mask=attn_mask, 
							key_padding_mask=key_padding_mask)
		x = x + attn_out
		
		# Feedforward with residual
		ff_out = self.ff1(self.ln2(x))
		ff_out = self.gelu(ff_out)
		ff_out = self.ff2(ff_out)
		ff_out = self.dropout(ff_out)
		x = x + ff_out
		
		return x


class GPTModel(Module):
	
	def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12,
				 d_ff=3072, dropout=0.1, max_seq_len=1024):
		super().__init__()
		self.d_model = d_model
		self.vocab_size = vocab_size
		
		# Token embeddings
		from .embedding import Embedding
		self.token_embedding = Embedding(vocab_size, d_model)
		
		# Position embeddings (learned)
		from .positional import LearnedPositionalEmbedding
		self.position_embedding = LearnedPositionalEmbedding(max_seq_len, d_model)
		
		# Transformer blocks
		self.blocks = ModuleList([
			GPTBlock(d_model, num_heads, d_ff, dropout)
			for _ in range(num_layers)
		])
		
		# Final layer norm
		self.ln_f = LayerNorm(d_model)
		
		# Output head
		self.lm_head = Linear(d_model, vocab_size, bias=False)
		
		# Tie weights
		self.lm_head.weight = self.token_embedding.weight
		
		self.dropout = Dropout(dropout)
	
	def forward(self, input_ids, attention_mask=None):
		# input_ids: (batch, seq_len)
		
		# Get embeddings
		token_emb = self.token_embedding(input_ids)
		x = self.position_embedding(token_emb)
		x = self.dropout(x)
		
		# Create causal mask
		seq_len = input_ids.shape[1]
		from .attention import create_causal_mask
		causal_mask = create_causal_mask(seq_len, device=x.active_device)
		
		# Pass through transformer blocks
		for block in self.blocks:
			x = block(x, attn_mask=causal_mask, key_padding_mask=attention_mask)
		
		# Final norm
		x = self.ln_f(x)
		
		# Project to vocabulary
		logits = self.lm_head(x)
		
		return logits


# LLaMA-style decoder-only model (uses RMSNorm and RoPE)

class LLaMABlock(Module):
	
	def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
		super().__init__()
		self.d_model = d_model
		self.num_heads = num_heads
		
		if d_ff is None:
			# LLaMA uses SwiGLU, which needs 2/3 * 4 * d_model
			d_ff = int(2 * 4 * d_model / 3)
			d_ff = ((d_ff + 255) // 256) * 256  # Round to nearest 256
		self.d_ff = d_ff
		
		# RMSNorm (more efficient than LayerNorm)
		self.ln1 = RMSNorm(d_model)
		self.ln2 = RMSNorm(d_model)
		
		# Self-attention
		self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
		
		# Feedforward with SwiGLU
		from .activation import SwiGLU
		self.ff_gate = Linear(d_model, d_ff * 2)  # For SwiGLU split
		self.ff_down = Linear(d_ff, d_model)
		self.swiglu = SwiGLU()
		
		# RoPE (applied in attention, stored here for reference)
		from .positional import RotaryPositionalEmbedding
		self.rope = RotaryPositionalEmbedding(d_model // num_heads)
	
	def forward(self, x, attn_mask=None):
		# x: (batch, seq_len, d_model)
		
		# Self-attention with residual and RMSNorm
		normed = self.ln1(x)
		attn_out = self.attn(normed, attn_mask=attn_mask)
		x = x + attn_out
		
		# Feedforward with SwiGLU and residual
		normed = self.ln2(x)
		ff_out = self.ff_gate(normed)
		ff_out = self.swiglu(ff_out)  # SwiGLU activation
		ff_out = self.ff_down(ff_out)
		x = x + ff_out
		
		return x


__all__ = [
	'TransformerEncoderLayer',
	'TransformerDecoderLayer',
	'TransformerEncoder',
	'TransformerDecoder',
	'Transformer',
	'GPTBlock',
	'GPTModel',
	'LLaMABlock',
]