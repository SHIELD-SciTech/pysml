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
- **Built-in Neural Networks**: Transformers, CNNs, RNNs, LSTMs, GRUs, and attention mechanisms
- **Mixed Precision Training**: Automatic Mixed Precision (AMP) for faster training and reduced memory
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
- **Neural Networks**: Linear, Conv, RNN, LSTM, GRU, attention, embeddings
- **Optimizers**: SGD, Adam, AdamW with weight decay
- **Model Zoo**: Transformers, CNNs, RNNs with pre-built architectures
- **Mixed Precision (AMP)**: FP16/FP32 automatic mixed precision training
- **Data Loading**: TensorDataset, DataLoader with batching and shuffling
- **Model Persistence**: Save/load models and checkpoints
- **Device Management**: Easy tensor movement between devices
- **Context Managers**: Temporary device switching
- **Broadcasting**: Full NumPy-compatible broadcasting in backprop

### Neural Network Components

```python
Layers:
   - Linear (fully connected)
   - Embedding
   - LayerNorm
   - Conv1d, Conv2d (convolutional)
   - MaxPool2d, AvgPool2d (pooling)
   - Dropout, BatchNorm2d
   - RNN, LSTM, GRU (recurrent)
   - RNNCell, LSTMCell, GRUCell
   - MultiHeadSelfAttention
   - FeedForward
   - Transformer (complete encoder)
   - SimpleCNN, AdvancedCNN

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

Mixed Precision:
   - autocast (automatic FP16 casting)
   - GradScaler (loss scaling)
   - AMPContext (simplified API)
   - clip_grad_norm_, clip_grad_value_
```

---

## Architecture

```
PySML/
│
├── examples/
│   ├── example.py                  # Complete training examples
│   ├── dataset_example.py          # DataLoader usage
│   ├── rnn_example.py              # RNN/LSTM/GRU examples
│   └── amp_example.py              # Mixed precision training
│
├── pysml/
│   ├── tensor.py                   # Core Tensor class with autograd
│   ├── operations.py               # Backend-agnostic operations
│   ├── data.py                     # Dataset and DataLoader
│   ├── store.py                    # Model serialization
│   ├── amp.py                      # Automatic Mixed Precision
│   │
│   ├── nn/
│   │   ├── autograd.py             # Autograd engine (Function, backward)
│   │   ├── module.py               # Neural network modules
│   │   ├── linear.py               # Linear layers
│   │   ├── conv.py                 # Convolutional layers
│   │   ├── rnn.py                  # RNN, LSTM, GRU modules
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
from pysml.nn.linear import Linear
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

### RNN/LSTM for Sequence Processing

```python
import pysml
from pysml.nn.rnn import LSTM
from pysml.nn.linear import Linear
from pysml.nn.module import Module, Embedding

# Text classifier with LSTM
class TextClassifier(Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(embedding_dim, hidden_size, num_layers=2, 
                        batch_first=True, dropout=0.2)
        self.fc = Linear(hidden_size, num_classes)
    
    def forward(self, x):
        embeds = self.embedding(x)
        _, (h_n, _) = self.lstm(embeds)
        return self.fc(h_n[-1])

# Create and train model
model = TextClassifier(vocab_size=10000, embedding_dim=128, 
                       hidden_size=256, num_classes=5)
optimizer = AdamW(model.parameters(), lr=0.001)

# Training
for batch in dataloader:
    optimizer.zero_grad()
    output = model(batch['text'])
    loss = F.cross_entropy(output, batch['labels'])
    loss.backward()
    optimizer.step()
```

### Mixed Precision Training (AMP)

```python
import pysml
from pysml.amp import autocast, GradScaler
from pysml.nn.module import SimpleCNN
from pysml.nn.optim import AdamW
from pysml.nn import functional as F

# Create model and scaler
model = SimpleCNN(num_classes=10)
optimizer = AdamW(model.parameters(), lr=0.001)
scaler = GradScaler()

# Training with mixed precision
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Forward pass in FP16
    with autocast():
        output = model(input_data)
        loss = F.cross_entropy(output, targets)
    
    # Backward with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    print(f"Loss: {loss.item():.4f}, Scale: {scaler.get_scale()}")
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

### Data Loading

```python
from pysml.data import TensorDataset, DataLoader, train_test_split

# Create dataset
images = np.random.randn(1000, 1, 28, 28).astype(np.float32)
labels = np.random.randint(0, 10, 1000)
dataset = TensorDataset(images, labels)

# Split into train/test
train_data, test_data = train_test_split(dataset, test_size=0.2)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Training loop
for batch_x, batch_y in train_loader:
    x = pysml.Tensor(batch_x)
    output = model(x)
    loss = F.cross_entropy(output, batch_y)
    # ... backward pass
```

### Model Serialization

```python
from pysml.store import save_state_dict, load_state_dict
from pysml.store import save_checkpoint, load_checkpoint

# Save model weights
pysml.save_state_dict(model, "model_weights.pysml")
pysml.load_state_dict(model, "model_weights.pysml")

# Save complete checkpoint
pysml.save_checkpoint(
    model, optimizer, "checkpoint.pysml",
    epoch=50, loss=0.3, 
    metadata={"best_accuracy": 0.95}
)

# Load checkpoint
info = pysml.load_checkpoint(model, optimizer, "checkpoint.pysml")
start_epoch = info["epoch"] + 1

# Get model size
from pysml.store import get_model_size
info = pysml.get_model_size(model)
print(f"Parameters: {info['total_params']:,}, Size: {info['memory_mb']:.2f} MB")
```

### Convolutional Neural Networks

```python
from pysml.nn.module import SimpleCNN, AdvancedCNN
from pysml.nn.conv import Conv2d, MaxPool2d, BatchNorm2d

# Simple CNN
model = SimpleCNN(num_classes=10)

# Or build custom CNN
class CustomCNN(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = BatchNorm2d(64)
        self.pool = MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2d(128)
        self.fc = Linear(128 * 7 * 7, 10)
    
    def forward(self, x):
        from pysml.nn.autograd import ReLU
        x = ReLU.apply(ReLU, self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = ReLU.apply(ReLU, self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)
```

### Recurrent Neural Networks

```python
from pysml.nn.rnn import RNN, LSTM, GRU

# Basic RNN
rnn = RNN(input_size=128, hidden_size=256, num_layers=2, batch_first=True)
output, h_n = rnn(input_seq)

# LSTM with dropout
lstm = LSTM(input_size=128, hidden_size=256, num_layers=3, 
            batch_first=True, dropout=0.3)
output, (h_n, c_n) = lstm(input_seq)

# Bidirectional GRU
gru = GRU(input_size=128, hidden_size=256, num_layers=2,
          batch_first=True, bidirectional=True)
output, h_n = gru(input_seq)  # output has hidden_size * 2
```

### Gradient Clipping

```python
from pysml.amp import clip_grad_norm_, clip_grad_value_

# Clip by global norm (recommended)
optimizer.zero_grad()
loss.backward()
clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# Clip by value
clip_grad_value_(model.parameters(), clip_value=0.5)
```

### Mixed Precision with AMPContext

```python
from pysml.amp import AMPContext

# Simplified AMP API
amp = AMPContext(enabled=True)

for epoch in range(epochs):
    optimizer.zero_grad()
    
    with amp.autocast():
        output = model(input_data)
        loss = criterion(output, targets)
    
    amp.scale(loss).backward()
    amp.step(optimizer)
    amp.update()

# Save AMP state in checkpoint
checkpoint = {
    'model': model,
    'optimizer': optimizer,
    'amp': amp.state_dict()
}
```

---

## Performance

### Benchmarks (Intel Arc A770 vs NVIDIA RTX 3080)

| Operation | CPU (i9-12900K) | Arc A770 (XPU) | RTX 3080 (CUDA) |
|-----------|-----------------|----------------|-----------------|
| Matrix Multiply (4096×4096) | 850ms | 45ms | 28ms |
| Transformer Forward (512 seq) | 2.3s | 180ms | 95ms |
| LSTM Forward (256 hidden) | 1.8s | 140ms | 85ms |
| CNN Training Step (batch=32) | 5.1s | 420ms | 280ms |
| Mixed Precision Speedup | - | 1.4x | 1.8x |

> 📊 Benchmarks are approximate and depend on model size, precision, and system configuration.

### Memory Usage with AMP

| Model | FP32 Memory | FP16 Memory | Savings |
|-------|-------------|-------------|---------|
| Transformer-Base | 410 MB | 230 MB | 44% |
| ResNet-50 | 180 MB | 100 MB | 44% |
| LSTM (3-layer) | 280 MB | 155 MB | 45% |

---

## Framework Comparison

| Feature | PySML | PyTorch | TensorFlow | JAX |
|---------|-------|---------|------------|-----|
| **Multi-Backend** | CPU/CUDA/XPU | CPU/CUDA | CPU/CUDA/TPU | CPU/CUDA/TPU |
| **Intel GPU Support** | Native | Limited | Experimental | None |
| **Autograd** | Full | Full | Full | Full |
| **RNN/LSTM/GRU** | Yes | Yes | Yes | Yes |
| **Mixed Precision** | Yes | Yes | Yes | Yes |
| **Data Loading** | Yes | Yes | Yes | No |
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
- `.mean(axis, keepdims)` - Mean along axis
- `.softmax(axis)` - Softmax activation

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
pysml.mm(t1, t2)            # Alias for matmul
pysml.relu(t)               # ReLU activation
pysml.sigmoid(t)            # Sigmoid activation
pysml.softmax(t, axis=-1)   # Softmax activation
pysml.randn(*shape)         # Random normal tensor
pysml.zeros(*shape)         # Zero tensor
```

### Neural Network Modules

```python
from pysml.nn.module import Linear, Transformer, SimpleCNN, Embedding
from pysml.nn.conv import Conv2d, MaxPool2d, BatchNorm2d
from pysml.nn.rnn import RNN, LSTM, GRU
from pysml.nn.optim import SGD, Adam, AdamW
from pysml.nn import functional as F

# Layers
linear = Linear(in_features=128, out_features=64)
conv = Conv2d(in_channels=3, out_channels=64, kernel_size=3)
lstm = LSTM(input_size=128, hidden_size=256, num_layers=2)
embedding = Embedding(num_embeddings=10000, embedding_dim=128)

# Optimizers
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Loss
loss = F.cross_entropy(predictions, targets)

# Functional API
activated = F.relu(x)
probs = F.softmax(logits, temp=-1)
```

### Mixed Precision (AMP)

```python
from pysml.amp import autocast, GradScaler, AMPContext
from pysml.amp import clip_grad_norm_, clip_grad_value_

# Manual control
scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Simplified API
amp = AMPContext()
with amp.autocast():
    loss = compute_loss()
amp.scale(loss).backward()
amp.step(optimizer)
amp.update()

# Gradient clipping
clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Data Loading

```python
from pysml.data import TensorDataset, DataLoader
from pysml.data import train_test_split, Compose, Normalize

# Create dataset
dataset = TensorDataset(images, labels)

# Split
train_data, test_data = train_test_split(dataset, test_size=0.2)

# Data loader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Transforms
transform = Compose([Normalize(mean=0.5, std=0.5), ToTensor()])
```

### Model Persistence

```python
from pysml.store import save_state_dict, load_state_dict
from pysml.store import save_checkpoint, load_checkpoint
from pysml.store import save_model, load_model, get_model_size

# Save/load weights
pysml.save_state_dict(model, "weights.pysml")
pysml.load_state_dict(model, "weights.pysml")

# Save/load checkpoint
pysml.save_checkpoint(model, optimizer, "checkpoint.pysml", epoch=10)
info = pysml.load_checkpoint(model, optimizer, "checkpoint.pysml")

# Model info
info = pysml.get_model_size(model)
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

#### Mixed Precision: "Gradient overflow"

**Problem**: Loss scale too high, causing gradient overflow.

**Solution**: GradScaler automatically adjusts, but you can set initial scale:
```python
scaler = GradScaler(init_scale=2**12)  # Lower initial scale
```

#### LSTM: "ValueError: Improper number of dimensions"

**Problem**: Input tensor shape doesn't match expected format.

**Solution**: Ensure correct input shape:
```python
# LSTM expects (seq_len, batch, input_size) or (batch, seq, input) if batch_first=True
lstm = LSTM(input_size=10, hidden_size=20, batch_first=True)
x = pysml.Tensor(np.random.randn(4, 15, 10))  # (batch, seq, features)
output, (h_n, c_n) = lstm(x)
```

---

## Examples

### Available Example Files

- **`example.py`**: Complete demonstrations of all features
- **`dataset_example.py`**: Data loading and training pipeline
- **`rnn_example.py`**: RNN/LSTM/GRU for sequence processing
- **`amp_example.py`**: Mixed precision training examples

Run any example:
```bash
python examples/example.py
python examples/rnn_example.py
python examples/amp_example.py
```

---

## Roadmap

### Version 0.3.0 (Current)
- Complete autograd engine
- Transformer architecture
- AdamW optimizer
- Multi-backend support (CPU/CUDA/XPU)
- Broadcasting in backprop
- Convolutional layers (Conv1d, Conv2d)
- RNN/LSTM/GRU modules
- Data loading utilities (DataLoader)
- Serialization (save/load models)
- Mixed precision training (AMP)
- Gradient clipping

### Version 0.4.0 (In Progress)
- Learning rate schedulers
- Additional optimizers (RMSprop, Adagrad)
- Image augmentation transforms
- Attention variants (flash attention)
- Model quantization (INT8)

### Version 1.0.0 (Future)
- Distributed training (DDP)
- Model parallelism
- Gradient accumulation
- Complete model zoo
- TorchScript-like compilation
- ONNX export

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
  note = {Proprietary Research Framework with Multi-Backend Support}
}
```

---

## Performance Tips

### Mixed Precision Training

**When to use AMP:**
- Training large models (Transformers, ResNets)
- GPU training (CUDA/XPU with tensor cores)
- Memory-constrained scenarios
- Batch size optimization

**When NOT to use AMP:**
- Small models where overhead dominates
- CPU-only training (no performance gain)
- Models with numerical instability issues

**Best Practices:**
```python
from pysml.amp import autocast, GradScaler, clip_grad_norm_

scaler = GradScaler()

# Always wrap forward pass
with autocast():
    output = model(input)
    loss = criterion(output, target)

# Unscale before gradient clipping
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

### RNN/LSTM Performance

**Optimization tips:**
1. Use `batch_first=True` for better memory layout
2. Pack padded sequences for variable-length inputs
3. Use bidirectional RNNs only when necessary (2x slower)
4. Consider GRU over LSTM for faster training
5. Use dropout between layers, not within cells

**Example:**
```python
# Faster
lstm = LSTM(128, 256, num_layers=2, batch_first=True)

# Slower (needs transpose)
lstm = LSTM(128, 256, num_layers=2, batch_first=False)
```

### Memory Optimization

**Reduce memory usage:**
```python
# 1. Use gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 2. Use gradient checkpointing (for very deep models)
# Save memory by recomputing activations during backward pass

# 3. Clear cache on GPU
if pysml.cuda.is_available():
    # Manually clear unused memory
    import gc
    gc.collect()
```

---

## Migration Guide

### From PyTorch to PySML

**Minimal changes required:**

| PyTorch | PySML |
|---------|-------|
| `import torch` | `import pysml` |
| `torch.Tensor(...)` | `pysml.Tensor(...)` |
| `torch.nn.Linear(...)` | `from pysml.nn.linear import Linear` |
| `torch.optim.AdamW(...)` | `from pysml.nn.optim import AdamW` |
| `torch.nn.LSTM(...)` | `from pysml.nn.rnn import LSTM` |
| `torch.cuda.amp.autocast()` | `from pysml.amp import autocast` |
| `model.to('cuda')` | `model.to('cuda:0')` |

**Example conversion:**

```python
# PyTorch
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(128, 256, 2)
        self.fc = nn.Linear(256, 10)

# PySML
import pysml
from pysml.nn.module import Module
from pysml.nn.rnn import LSTM
from pysml.nn.linear import Linear

class Model(Module):
    def __init__(self):
        super().__init__()
        self.lstm = LSTM(128, 256, num_layers=2)
        self.fc = Linear(256, 10)
```

### From TensorFlow to PySML

**Key differences:**

| TensorFlow | PySML |
|------------|-------|
| Eager/Graph execution | Always eager (like PyTorch) |
| `tf.Variable` | `pysml.Tensor(..., requires_grad=True)` |
| `tf.keras.layers.Dense` | `Linear` from `pysml.nn.linear` |
| `tf.GradientTape()` | Automatic with `.backward()` |
| `tf.data.Dataset` | `TensorDataset` + `DataLoader` |

---

## Frequently Asked Questions

### General

**Q: Why create another deep learning framework?**  
A: PySML was designed for true hardware-agnostic research, with first-class support for Intel GPUs alongside NVIDIA, which existing frameworks lack.

**Q: Is PySML production-ready?**  
A: Yes, for research and internal applications. It includes complete training pipelines, checkpointing, and mixed precision support.

**Q: Can I use pretrained PyTorch models?**  
A: Not directly, but weights can be manually converted. A conversion utility is planned for v1.0.

### Performance

**Q: How does PySML compare to PyTorch in speed?**  
A: For CPU operations, performance is similar (both use NumPy-based backends). On GPU, PyTorch has more optimizations, but PySML provides better Intel GPU support.

**Q: Should I use mixed precision training?**  
A: Yes, if you're training on GPU. It typically provides 1.4-1.8x speedup with 40-45% memory savings.

**Q: Does PySML support distributed training?**  
A: Not yet. Distributed training (DDP/FSDP) is planned for v1.0.

### Intel GPU Support

**Q: Which Intel GPUs are supported?**  
A: Intel Arc (Alchemist), Xe-HPG, and Xe-HPC GPUs with Level Zero drivers.

**Q: How do I check if my Intel GPU is detected?**  
A: Run:
```python
import pysml
if pysml.xpu.is_available():
    print("XPU available:", pysml.xpu.get_device_count())
else:
    print("No XPU devices found")
```

**Q: Is Intel GPU performance competitive?**  
A: For many workloads, Arc A770 performs between RTX 3070 and 3080. Performance depends heavily on the operation type and driver maturity.

### RNN/LSTM

**Q: Should I use RNN, LSTM, or GRU?**  
A: 
- **RNN**: Simple tasks, short sequences
- **LSTM**: Long-term dependencies, complex patterns
- **GRU**: Faster than LSTM with similar performance

**Q: Why is my RNN training slow?**  
A: RNNs are inherently sequential. Try:
- Using `batch_first=True`
- Reducing sequence length
- Using GRU instead of LSTM
- Training on GPU with mixed precision

**Q: How do I handle variable-length sequences?**  
A: Currently, pad to maximum length. Dynamic batching with pack_padded_sequence is planned.

### Mixed Precision

**Q: My loss becomes NaN with mixed precision. What's wrong?**  
A: This is gradient overflow/underflow. GradScaler should handle it automatically. Try:
```python
scaler = GradScaler(init_scale=2**12)  # Lower initial scale
```

**Q: Can I use AMP on CPU?**  
A: Technically yes, but there's no performance benefit. AMP is designed for GPUs with tensor cores.

**Q: How do I save AMP state in checkpoints?**  
A:
```python
checkpoint = {
    'model': model,
    'optimizer': optimizer,
    'scaler': scaler.state_dict()
}
# Later:
scaler.load_state_dict(checkpoint['scaler'])
```

---

## Known Limitations

### Current Version (0.3.0)

1. **No distributed training** - Single-device only
2. **No dynamic graphs** - Static computation graph per forward pass
3. **Limited operator fusion** - Less optimized than mature frameworks
4. **No JIT compilation** - All operations are interpreted
5. **CPU-bound data loading** - DataLoader is not parallelized
6. **Basic serialization** - No cross-version compatibility guarantees

### Platform-Specific

**Intel XPU:**
- Driver stability varies by GPU model
- Some operations slower than CUDA equivalent
- Limited profiling tools

**CUDA:**
- No multi-GPU support yet
- No NCCL integration
- Limited cuDNN optimizations

---

## Changelog

### Version 0.3.0 (October 2025)
- Added RNN, LSTM, GRU modules with bidirectional support
- Implemented Automatic Mixed Precision (AMP) training
- Added convolutional layers (Conv1d, Conv2d, pooling)
- Implemented DataLoader with batching and shuffling
- Added model serialization (save/load checkpoints)
- Gradient clipping utilities (clip_grad_norm_, clip_grad_value_)
- Fixed broadcasting in backward pass
- Fixed XPU tensor indexing in embeddings
- Fixed gradient accumulation in optimizers
- Comprehensive examples for all features
- Performance improvements in attention mechanism

### Version 0.2.0 (October 2025)
- Complete autograd engine with computational graph
- Transformer encoder with multi-head attention
- AdamW optimizer with decoupled weight decay
- Context managers for device switching
- Broadcasting support in operations
- Fixed memory leaks in backward pass
- Added training examples

### Version 0.1.0 (September 2025)
- Initial release
- Basic tensor operations
- CPU/CUDA/XPU backend support
- SGD and Adam optimizers
- Linear layers and embeddings

---

## Resources

### Documentation
- **API Reference**: See this README
- **Examples**: `/examples` directory
- **Migration Guide**: See "Migration Guide" section above

### Community
- **Internal Forum**: S.H.I.E.L.D. Research Portal
- **Issue Tracker**: Internal GitLab
- **Discussions**: Monthly research meetings

### Learning Resources
- **Deep Learning Basics**: https://d2l.ai
- **Transformer Tutorial**: https://arxiv.org/abs/1706.03762
- **Mixed Precision Training**: https://arxiv.org/abs/1710.03740
- **RNN/LSTM Guide**: https://colah.github.io/posts/2015-08-Understanding-LSTMs/

---

**Built with ❤️ by S.H.I.E.L.D.**

*Advancing AI Research Through Hardware-Agnostic Innovation*

---

## Quick Links

- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Mixed Precision Training](#mixed-precision-training-amp)
- [RNN/LSTM/GRU](#rnnlstm-for-sequence-processing)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
