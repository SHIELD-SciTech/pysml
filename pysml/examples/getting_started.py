#!/usr/bin/env python3
# PySML Getting Started
# Basic usage examples for the framework

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pysml
from pysml import Tensor, nn, optim
import pysml.dtype as dtype


def example_tensors():
    print('=== Basic Tensors ===')
    
    a = Tensor([1.0, 2.0, 3.0], dtype=dtype.fp32())
    b = Tensor([4.0, 5.0, 6.0], dtype=dtype.fp32())
    
    c = pysml.add(a, b)
    d = pysml.multiply(a, b)
    
    print(f'a = {a.data}')
    print(f'b = {b.data}')
    print(f'a + b = {c.data}')
    print(f'a * b = {d.data}')
    print()


def example_autograd():
    print('=== Autograd ===')
    
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype.fp32(), requires_grad=True)
    w = Tensor([[0.5, 0.5], [0.5, 0.5]], dtype=dtype.fp32(), requires_grad=True)
    
    y = pysml.matmul(x, w)
    z = pysml.mean(y)
    
    print(f'x = {x.data}')
    print(f'w = {w.data}')
    print(f'y = matmul(x, w) = {y.data}')
    print(f'z = mean(y) = {z.item():.4f}')
    
    z.backward()
    
    print(f'x.grad = {x.grad.data}')
    print(f'w.grad = {w.grad.data}')
    print()


def example_model():
    print('=== Simple MLP ===')
    
    class MLP(nn.Module):
        def __init__(self, in_dim, hidden, out_dim):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden)
            self.fc2 = nn.Linear(hidden, out_dim)
        
        def forward(self, x):
            x = pysml.relu(self.fc1(x))
            return self.fc2(x)
    
    model = MLP(10, 32, 2)
    print(f'Parameters: {model.num_parameters()}')
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print('Training...')
    for epoch in range(100):
        x = Tensor(np.random.randn(8, 10).astype(np.float32), dtype=dtype.fp32())
        y = Tensor(np.random.randn(8, 2).astype(np.float32), dtype=dtype.fp32())
        
        pred = model(x)
        loss = criterion(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f'  Epoch {epoch + 1}: Loss = {loss.item():.4f}')
    print()


def example_attention():
    print('=== Multi-Head Attention ===')
    
    d_model, n_heads = 64, 4
    batch_size, seq_len = 2, 8
    
    attn = nn.MultiheadAttention(d_model, n_heads)
    x = Tensor(np.random.randn(batch_size, seq_len, d_model).astype(np.float32), dtype=dtype.fp32())
    
    output, weights = attn(x, x, x)
    
    print(f'Input: {x.shape}')
    print(f'Output: {output.shape}')
    print(f'Attention weights: {weights.shape}')
    print()


def example_sequential():
    print('=== Sequential ===')
    
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.GELU(),
        nn.Linear(32, 10),
    )
    
    x = Tensor(np.random.randn(4, 10).astype(np.float32), dtype=dtype.fp32())
    y = model(x)
    
    print(f'Input: {x.shape}')
    print(f'Output: {y.shape}')
    print(f'Parameters: {model.num_parameters()}')
    print()


def example_save_load():
    print('=== Save/Load ===')
    
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))
    state = model.state_dict()
    print(f'State dict keys: {list(state.keys())}')
    
    model2 = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))
    model2.load_state_dict(state)
    print('Model loaded successfully')
    print()


def main():
    print('PySML - Getting Started')
    print('=' * 40)
    print()
    
    example_tensors()
    example_autograd()
    example_model()
    example_attention()
    example_sequential()
    example_save_load()
    
    print('All examples complete!')


if __name__ == '__main__':
    main()
