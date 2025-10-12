import numpy as np
from pysml.tensor import Tensor, dtype
from pysml.nn import functional as F
from pysml.nn.linear import Linear
from pysml.nn.module import Module
import pysml.operations as ops
import math


class MultiHeadSelfAttention(Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_linear = Linear(d_model, d_model)
        self.v_linear = Linear(d_model, d_model)
        self.k_linear = Linear(d_model, d_model)
        self.out = Linear(d_model, d_model)

    def forward(self, x, mask=None):
        bs = x.shape[0]

        # Perform linear op.
        k = self.k_linear(x).view(bs, -1, self.n_heads, self.d_k)
        q = self.q_linear(x).view(bs, -1, self.n_heads, self.d_k)
        v = self.v_linear(x).view(bs, -1, self.n_heads, self.d_k)

        # Transpose
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = scores.softmax(axis=-1)
        
        output = attention_weights @ v

        output = output.transpose(1, 2).view(bs, -1, self.d_model)
        output = self.out(output)
        return output


class PositionalEncoding(Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = ops.zeros(max_len, d_model).data
        position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2).astype(np.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = Tensor(pe, requires_grad=False)

    def forward(self, x):
        # x is expected to be of shape (batch_size, seq_len, d_model)
        return x + self.pe[:x.shape[1], :]
    
