# Simple RWKV Model for PySML
# Lightweight implementation for quick experiments

from __future__ import annotations
import math
from typing import Optional, Tuple, List
import numpy as np

import pysml
from pysml import Tensor, nn
from pysml.nn import Module, Parameter
import pysml.dtype as dtype


class TimeMixing(Module):
    # Time mixing block - handles sequence interactions
    
    def __init__(self, d_model: int, layer_id: int, n_layers: int, device: str = 'cpu'):
        super().__init__()
        self.d_model = d_model
        
        ratio = layer_id / max(n_layers - 1, 1)
        mix = np.ones(d_model, dtype=np.float32) * (0.5 + 0.3 * ratio)
        
        self.time_mix_k = Parameter(Tensor(mix.copy(), dtype=dtype.fp32(), device=device, requires_grad=True))
        self.time_mix_v = Parameter(Tensor(mix.copy(), dtype=dtype.fp32(), device=device, requires_grad=True))
        self.time_mix_r = Parameter(Tensor(np.ones(d_model, dtype=np.float32) * 0.5, dtype=dtype.fp32(), device=device, requires_grad=True))
        
        self.key = nn.Linear(d_model, d_model, bias=False, device=device)
        self.value = nn.Linear(d_model, d_model, bias=False, device=device)
        self.receptance = nn.Linear(d_model, d_model, bias=False, device=device)
        self.output = nn.Linear(d_model, d_model, bias=False, device=device)
    
    def forward(self, x: Tensor, state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        batch, seq_len, d_model = x.shape
        backend = x._backend
        
        if seq_len > 1:
            zeros = Tensor(backend.zeros((batch, 1, d_model)), dtype=x._dtype, device=x.active_device)
            x_prev = pysml.concatenate([zeros, x[:, :-1, :]], axis=1)
        else:
            x_prev = state if state is not None else Tensor(backend.zeros((batch, 1, d_model)), dtype=x._dtype, device=x.active_device)
        
        ones = Tensor(backend.ones(self.time_mix_k.shape), dtype=x._dtype, device=x.active_device)
        
        xk = pysml.add(pysml.multiply(x, self.time_mix_k), pysml.multiply(x_prev, pysml.subtract(ones, self.time_mix_k)))
        xv = pysml.add(pysml.multiply(x, self.time_mix_v), pysml.multiply(x_prev, pysml.subtract(ones, self.time_mix_v)))
        xr = pysml.add(pysml.multiply(x, self.time_mix_r), pysml.multiply(x_prev, pysml.subtract(ones, self.time_mix_r)))
        
        k = self.key(xk)
        v = self.value(xv)
        r = pysml.sigmoid(self.receptance(xr))
        
        kv = pysml.multiply(k, v)
        out = self.output(pysml.multiply(r, kv))
        
        return out, x[:, -1:, :]


class ChannelMixing(Module):
    # Channel mixing block - FFN with squared ReLU
    
    def __init__(self, d_model: int, layer_id: int, n_layers: int, device: str = 'cpu'):
        super().__init__()
        self.d_model = d_model
        hidden = d_model * 4
        
        ratio = layer_id / max(n_layers - 1, 1)
        mix = np.ones(d_model, dtype=np.float32) * (0.5 + 0.3 * ratio)
        
        self.time_mix_k = Parameter(Tensor(mix.copy(), dtype=dtype.fp32(), device=device, requires_grad=True))
        self.time_mix_r = Parameter(Tensor(np.ones(d_model, dtype=np.float32) * 0.5, dtype=dtype.fp32(), device=device, requires_grad=True))
        
        self.key = nn.Linear(d_model, hidden, bias=False, device=device)
        self.receptance = nn.Linear(d_model, d_model, bias=False, device=device)
        self.value = nn.Linear(hidden, d_model, bias=False, device=device)
    
    def forward(self, x: Tensor, state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        batch, seq_len, d_model = x.shape
        backend = x._backend
        
        if seq_len > 1:
            zeros = Tensor(backend.zeros((batch, 1, d_model)), dtype=x._dtype, device=x.active_device)
            x_prev = pysml.concatenate([zeros, x[:, :-1, :]], axis=1)
        else:
            x_prev = state if state is not None else Tensor(backend.zeros((batch, 1, d_model)), dtype=x._dtype, device=x.active_device)
        
        ones = Tensor(backend.ones(self.time_mix_k.shape), dtype=x._dtype, device=x.active_device)
        
        xk = pysml.add(pysml.multiply(x, self.time_mix_k), pysml.multiply(x_prev, pysml.subtract(ones, self.time_mix_k)))
        xr = pysml.add(pysml.multiply(x, self.time_mix_r), pysml.multiply(x_prev, pysml.subtract(ones, self.time_mix_r)))
        
        k = pysml.relu(self.key(xk))
        k = pysml.multiply(k, k)
        kv = self.value(k)
        
        r = pysml.sigmoid(self.receptance(xr))
        out = pysml.multiply(r, kv)
        
        return out, x[:, -1:, :]


class RWKVBlock(Module):
    # Single RWKV block
    
    def __init__(self, d_model: int, layer_id: int, n_layers: int, device: str = 'cpu'):
        super().__init__()
        self.ln1 = nn.RMSNorm(d_model, device=device)
        self.ln2 = nn.RMSNorm(d_model, device=device)
        self.time_mix = TimeMixing(d_model, layer_id, n_layers, device=device)
        self.channel_mix = ChannelMixing(d_model, layer_id, n_layers, device=device)
    
    def forward(self, x: Tensor, state: Optional[Tuple] = None):
        tm_state = state[0] if state else None
        cm_state = state[1] if state else None
        
        dx, tm_state = self.time_mix(self.ln1(x), tm_state)
        x = pysml.add(x, dx)
        
        dx, cm_state = self.channel_mix(self.ln2(x), cm_state)
        x = pysml.add(x, dx)
        
        return x, (tm_state, cm_state)


class RWKV(Module):
    # Simple RWKV language model
    
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, device: str = 'cpu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.device = device
        
        self.embedding = nn.Embedding(vocab_size, d_model, device=device)
        self.blocks = nn.ModuleList([RWKVBlock(d_model, i, n_layers, device=device) for i in range(n_layers)])
        self.ln_out = nn.RMSNorm(d_model, device=device)
        self.head = nn.Linear(d_model, vocab_size, bias=False, device=device)
    
    def forward(self, idx: Tensor, state: Optional[List] = None) -> Tuple[Tensor, List]:
        x = self.embedding(idx)
        
        new_states = []
        for i, block in enumerate(self.blocks):
            block_state = state[i] if state else None
            x, new_state = block(x, block_state)
            new_states.append(new_state)
        
        x = self.ln_out(x)
        logits = self.head(x)
        
        return logits, new_states
    
    def generate(self, idx: Tensor, max_new_tokens: int, temperature: float = 1.0) -> Tensor:
        backend = idx._backend
        state = None
        
        for _ in range(max_new_tokens):
            input_idx = idx[:, -1:] if state else idx
            logits, state = self(input_idx, state)
            
            logits = pysml.divide(logits[:, -1, :], temperature)
            probs = pysml.softmax(logits, axis=-1)
            
            probs_data = probs.data
            if hasattr(backend, 'asnumpy'):
                probs_data = backend.asnumpy(probs_data)
            probs_np = np.asarray(probs_data)
            next_id = np.argmax(probs_np, axis=-1)
            
            next_tensor = Tensor(next_id.reshape(-1, 1).astype(np.float32), dtype=idx._dtype, device=idx.active_device)
            idx = pysml.concatenate([idx, next_tensor], axis=1)
        
        return idx


def create_rwkv_small(vocab_size: int = 50257, device: str = 'cpu'):
    return RWKV(vocab_size=vocab_size, d_model=256, n_layers=6, device=device)


def create_rwkv_base(vocab_size: int = 50257, device: str = 'cpu'):
    return RWKV(vocab_size=vocab_size, d_model=512, n_layers=12, device=device)


def create_rwkv_large(vocab_size: int = 50257, device: str = 'cpu'):
    return RWKV(vocab_size=vocab_size, d_model=1024, n_layers=24, device=device)


__all__ = ['TimeMixing', 'ChannelMixing', 'RWKVBlock', 'RWKV', 'create_rwkv_small', 'create_rwkv_base', 'create_rwkv_large']
