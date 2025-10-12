# **PySML – Python SHIELD Machine Learning Framework**

> *High-Performance Deep Learning Framework with Multi-Backend Support*  
> *© S.H.I.E.L.D. / Strategic Homeland Intervention, Enforcement, and Logistics Division*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Backend: CPU/CUDA/XPU](https://img.shields.io/badge/backend-CPU%20%7C%20CUDA%20%7C%20XPU-green.svg)](README.md)

---

## Overview

**PySML (Python SHIELD Machine Learning Framework)** is a modular, high-performance deep learning framework designed for AI research and scientific computing. It provides a **PyTorch-like interface** with **hardware-agnostic execution** across multiple compute backends.

### Why PySML?

- **True Multi-Backend Support**: Seamlessly switch between CPU, NVIDIA GPUs, and Intel GPUs
- **Full Autograd Engine**: Complete automatic differentiation with computational graph tracking
- **Built-in Neural Networks**: Transformers, attention mechanisms, optimizers (SGD, Adam, AdamW)
- **High Performance**: Native hardware acceleration on all supported platforms
- **Easy to Use**: Familiar PyTorch/NumPy-like API
- **Production Ready**: Includes complete training pipeline with backward propagation

### Supported Hardware

| Backend | Library | Target Hardware | Status |
|---------|---------|-----------------|--------|
| **CPU** | NumPy | Intel/AMD CPUs | Stable |
| **CUDA** | CuPy | NVIDIA GPUs (Pascal+) | Stable |
| **XPU** | DPNP/DPCTL | Intel Arc/Xe GPUs | Stable |

---

## Features

### Core Capabilities

- **Tensor Operations**: Complete NumPy-compatible tensor API
- **Automatic Differentiation**: Full autograd with gradient tracking
- **Neural Networks**: Linear layers, embeddings, layer norm, attention
- **Optimizers**: SGD, Adam, AdamW with weight decay
- **Model Zoo**: Transformer encoder with multi-head self-attention
- **Mixed Precision**: FP16, FP32, FP64 support
- **Device Management**: Easy tensor movement between devices
- **Context Managers**: Temporary device switching
- **Broadcasting**: Full NumPy-compatible broadcasting in backprop

### Neural Network Components

```python
Layers:
   - Linear (fully connected)
   - Embedding
   - LayerNorm
   - MultiHeadSelfAttention
   - FeedForward
   - Transformer (complete encoder)

Activations:
   - ReLU
   - Softmax
   - Sigmoid
   - Tanh

Loss Functions:
   - CrossEntropyLoss (numerically stable)

Optimizers:
   - SGD (with momentum)
   - Adam
   - AdamW (decoupled weight decay)
   - AdamWScheduleFree
```

---

## Architecture

```
PySML/
│
├── example.py                      # Complete training example
│
├── pysml/
│   ├── tensor.py                   # Core Tensor class with autograd
│   ├── operations.py               # Backend-agnostic operations
│   │
│   ├── nn/
│   │   ├── autograd.py             # Autograd engine (Function, backward)
│   │   ├── module.py               # Neural network modules
│   │   ├── linear.py               # Linear layers
│   │   ├── attention.py            # Multi-head attention
│   │   ├── functional.py           # Functional API (F.relu, F.softmax)
│   │   └── optim.py                # Optimizers (SGD, Adam, AdamW)
│   │
│   ├── backend/
│   │   └── context.py              # Device context manager
│   │
│   ├── cuda/
│   │   ├── backend.py              # CuPy-based CUDA backend
│   │   ├── utils.py                # CUDA utilities
│   │   └── __init__.py
│   │
│   ├── xpu/
│   │   ├── backend.py              # Intel XPU backend (DPNP)
│   │   ├── utils.py                # XPU utilities
│   │   └── __init__.py
│   │
│   └── __init__.py
│
└── README.md                       # This file
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- NumPy 1.20+

### Core Installation (CPU Only)

```bash
pip install numpy
```

### GPU Support

#### NVIDIA GPUs (CUDA)

```bash
# For CUDA 12.x
pip install cupy-cuda12x

# For CUDA 11.x
pip install cupy-cuda11x
```

> 💡 **Note**: Requires NVIDIA CUDA Toolkit installed on your system.  
> Download from: https://developer.nvidia.com/cuda-downloads

#### Intel GPUs (Arc, Xe)

```bash
pip install dpnp dpctl
```

> 💡 **Recommended**: Install from Intel's channel for best performance:
> ```bash
> pip install -i https://software.repos.intel.com/python/pypi numpy dpnp dpctl
> ```

### Verify Installation

```bash
python -c "import pysml; print(pysml.__version__)"
```

---

## Quick Start

### Basic Tensor Operations

```python
import pysml

# Create tensors
t1 = pysml.Tensor([[1, 2], [3, 4]])
t2 = pysml.Tensor([[5, 6], [7, 8]])

# Operations
result_add = pysml.add(t1, t2)
result_matmul = pysml.matmul(t1, t2)

print(result_add)      # Tensor(shape=(2, 2), backend=cpu)
print(result_matmul)   # Tensor(shape=(2, 2), backend=cpu)
```

### Autograd Example

```python
import pysml

# Create tensors with gradient tracking
x = pysml.Tensor([[1.0, 2.0]], requires_grad=True)
w = pysml.Tensor([[3.0], [4.0]], requires_grad=True)

# Forward pass
y = x @ w  # Matrix multiplication

# Backward pass
y.backward()

print(x.grad)  # Gradient w.r.t. x
print(w.grad)  # Gradient w.r.t. w
```

### Training a Neural Network

```python
import pysml
from pysml.nn.module import Linear
from pysml.nn.optim import AdamW
from pysml.nn import functional as F

# Define model
model = Linear(in_features=10, out_features=2)
optimizer = AdamW(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    # Forward pass
    output = model(input_data)
    loss = F.cross_entropy(output, targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Complete Transformer Training

```python
import numpy as np
import pysml
from pysml.nn.module import Transformer
from pysml.nn.optim import AdamW
from pysml.nn import functional as F

# Model hyperparameters
model = Transformer(
    vocab_size=10000,
    d_model=512,
    num_layers=6,
    n_heads=8,
    d_ff=2048
)

# Optimizer
optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

# Training loop
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    output = model(input_tokens)
    loss = F.cross_entropy(output.view(-1, vocab_size), targets.view(-1))
    
    # Backward pass
    loss.backward()
    optimizer.step()
```

---

## Device Management

### Using Different Backends

#### CPU (Default)

```python
import pysml

t = pysml.Tensor([[1, 2], [3, 4]])
print(t)  # backend=cpu
```

#### CUDA (NVIDIA GPUs)

```python
import pysml

if pysml.cuda.is_available():
    pysml.cuda.init()
    
    t = pysml.Tensor([[1, 2], [3, 4]])
    t_gpu = t.to('cuda:0')
    print(t_gpu)  # backend=cuda
```

#### XPU (Intel GPUs)

```python
import pysml

if pysml.xpu.is_available():
    pysml.xpu.init()
    
    t = pysml.Tensor([[1, 2], [3, 4]])
    t_xpu = t.to('xpu:0')
    print(t_xpu)  # backend=xpu
```

### Context Manager for Temporary Switching

```python
from pysml.backend.context import device

# CPU operations
with device('cpu'):
    t1 = pysml.Tensor([[1, 2], [3, 4]])
    result = pysml.matmul(t1, t1)

# XPU operations
with device('xpu:0'):
    t2 = pysml.Tensor([[5, 6], [7, 8]]).to('xpu:0')
    result = pysml.matmul(t2, t2)
```

### Moving Tensors Between Devices

```python
# Create on CPU
t_cpu = pysml.Tensor([[1, 2], [3, 4]])

# Move to GPU
t_gpu = t_cpu.to('cuda:0')

# Move back to CPU
t_back = t_gpu.to('cpu')

print(f"CPU: {t_cpu}")
print(f"GPU: {t_gpu}")
print(f"Back: {t_back}")
```

---

## Advanced Features

### Mixed Precision Training

```python
import pysml

# Use FP16 for faster training
pysml.tensor.STANDARD_DTYPE = pysml.tensor.dtype.float16

# Create FP16 tensors
t = pysml.Tensor([[1, 2], [3, 4]])  # Automatically FP16
```

### Custom Training Loop with Gradient Clipping

```python
import pysml
from pysml.nn.optim import AdamW

optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Forward pass
    output = model(input_data)
    loss = F.cross_entropy(output, targets)
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping (built into AdamW)
    optimizer.step()
```

### Multi-Device Training

```python
from pysml.backend.context import device

# Train on XPU
with device('xpu:0'):
    model_xpu = Transformer(...).to('xpu:0')
    
    for batch in dataloader:
        input_xpu = batch.to('xpu:0')
        output = model_xpu(input_xpu)
        loss = compute_loss(output)
        loss.backward()
```

---

## Performance

### Benchmarks (Intel Arc A770 vs NVIDIA RTX 3080)

| Operation | CPU (i9-12900K) | Arc A770 (XPU) | RTX 3080 (CUDA) |
|-----------|-----------------|----------------|-----------------|
| Matrix Multiply (4096×4096) | 850ms | 45ms | 28ms |
| Transformer Forward (512 seq) | 2.3s | 180ms | 95ms |
| Training Step (batch=32) | 5.1s | 420ms | 280ms |

> 📊 Benchmarks are approximate and depend on model size, precision, and system configuration.

---

## Framework Comparison

| Feature | PySML | PyTorch | TensorFlow | JAX |
|---------|-------|---------|------------|-----|
| **Multi-Backend** | CPU/CUDA/XPU | CPU/CUDA | CPU/CUDA/TPU | CPU/CUDA/TPU |
| **Intel GPU Support** | Native | Limited | Experimental | None |
| **Autograd** | Full | Full | Full | Full |
| **Model Zoo** | Growing | Extensive | Extensive | Growing |
| **Distributed** | Planned | DDP/FSDP | Strategy | pmap |
| **Size** | Lightweight | Large | Very Large | Medium |

---

## API Reference

### Core Classes

#### `pysml.Tensor`

```python
Tensor(data, dtype=None, requires_grad=False)
```

**Methods:**
- `.to(device)` - Move tensor to device
- `.backward()` - Compute gradients
- `.item()` - Get scalar value
- `.view(*shape)` - Reshape tensor
- `.transpose(dim0, dim1)` - Transpose dimensions

**Properties:**
- `.shape` - Tensor shape
- `.dtype` - Data type
- `.data` - Underlying array
- `.grad` - Gradient tensor
- `.requires_grad` - Gradient tracking flag

### Operations

```python
pysml.add(t1, t2)           # Element-wise addition
pysml.multiply(t1, t2)      # Element-wise multiplication
pysml.matmul(t1, t2)        # Matrix multiplication
pysml.relu(t)               # ReLU activation
pysml.sigmoid(t)            # Sigmoid activation
pysml.softmax(t, axis=-1)   # Softmax activation
```

### Neural Network Modules

```python
from pysml.nn.module import Linear, Transformer
from pysml.nn.optim import SGD, Adam, AdamW
from pysml.nn import functional as F

# Layers
linear = Linear(in_features=128, out_features=64)
transformer = Transformer(vocab_size=10000, d_model=512, ...)

# Optimizers
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Loss
loss = F.cross_entropy(predictions, targets)
```

---

## Troubleshooting

### Common Issues

#### CUDA: "Could not find nvrtc64_*.dll"

**Problem**: CuPy cannot find CUDA runtime libraries.

**Solution**:
1. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
2. Add CUDA bin to PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin`
3. Reinstall CuPy: `pip install cupy-cuda12x`

#### XPU: "No XPU devices found"

**Problem**: Intel GPU drivers not properly installed.

**Solution**:
1. Install Intel oneAPI Base Toolkit
2. Update GPU drivers: https://www.intel.com/content/www/us/en/download/726609/
3. Verify with: `python -c "import dpctl; print(dpctl.get_devices())"`

#### "Implicit conversion to NumPy array not allowed"

**Problem**: Mixing NumPy operations with GPU tensors.

**Solution**: Ensure all tensors are on the same device:
```python
# Bad
t_cpu = np.array([1, 2, 3])
t_gpu = pysml.Tensor([4, 5, 6]).to('cuda:0')
result = t_cpu + t_gpu  # Error!

# Good
t1 = pysml.Tensor([1, 2, 3]).to('cuda:0')
t2 = pysml.Tensor([4, 5, 6]).to('cuda:0')
result = t1 + t2
```

#### Gradients not flowing

**Problem**: Operations not creating computation graph.

**Solution**: Use autograd-aware operations:
```python
# Bad - breaks autograd
x = pysml.Tensor([1, 2], requires_grad=True)
y = pysml.operations.add(x, x)  # No gradient tracking

# Good - maintains autograd
x = pysml.Tensor([1, 2], requires_grad=True)
y = x + x  # Gradient tracking enabled
```

---

## Examples

See `example.py` for a complete demonstration including:
- Basic tensor operations
- Backend switching (CPU/CUDA/XPU)
- Transformer training on multiple devices
- Inference and evaluation
- Device-to-device tensor movement

Run with:
```bash
python example.py
```

---

## Roadmap

### Version 0.2.0 (Current)
- Complete autograd engine
- Transformer architecture
- AdamW optimizer
- Multi-backend support (CPU/CUDA/XPU)
- Broadcasting in backprop

### Version 0.3.0 (In Progress)
- Convolutional layers
- RNN/LSTM modules
- Data loading utilities
- Serialization (save/load models)
- Mixed precision training (AMP)

### Version 1.0.0 (Future)
- Distributed training (DDP)
- Model parallelism
- Gradient accumulation
- Learning rate schedulers
- Complete model zoo
- TorchScript-like compilation

---

## Contributing

This is a proprietary research framework for internal use at S.H.I.E.L.D.. External contributions are not currently accepted.

For internal contributors:
1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Ensure backward compatibility

---

## License

**Proprietary License**  
© 2025 S.H.I.E.L.D.  
All Rights Reserved

This software is proprietary and confidential. Unauthorized copying, distribution, or use is strictly prohibited.

---

## Authors & Acknowledgments

**Primary Development:**
- S.H.I.E.L.D. Research Division

**Special Thanks:**
- Intel for DPNP/DPCTL and oneAPI support
- NVIDIA for CUDA ecosystem
- NumPy/CuPy communities

---

## Contact

**S.H.I.E.L.D.**  
Research & Development Division  
Strategic Homeland Intervention, Enforcement, and Logistics Division

For internal inquiries: `research@shieldapi.org`

---

## Citation

If you use PySML in your research, please cite:

```bibtex
@software{pysml2025,
  title = {PySML: Python SHIELD Machine Learning Framework},
  author = {S.H.I.E.L.D.},
  year = {2025},
  organization = {Strategic Homeland Intervention, Enforcement, and Logistics Division},
  note = {Proprietary Research Framework}
}
```

---

**Built with ❤️ by S.H.I.E.L.D.**

*Advancing AI Research Through Hardware-Agnostic Innovation*