# **PySML v0.5-alpha1 – Python SHIELD Machine Learning Framework**

> *Enterprise-Grade Distributed AI with Multi-Backend Support (CPU, CUDA, XPU)*  
> *© S.H.I.E.L.D. / Strategic Homeland Intervention, Enforcement, and Logistics Division*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Backend: CPU/CUDA/XPU](https://img.shields.io/badge/backend-CPU%20%7C%20CUDA%20%7C%20XPU-green.svg)](README.md)
[![Version: 0.5-alpha1](https://img.shields.io/badge/version-0.5--alpha1-brightgreen.svg)](README.md)

---

## What's New in v0.5-alpha1

**Major Upgrade — October 2025**

PySML v0.5 introduces a refined distributed deep learning core with AMP (Automatic Mixed Precision), RWKV-based transformer alternatives, and enhanced backend optimizations.

### Core Improvements
- **RWKV Model Support** – Production-ready recurrent-transformer architecture for LLMs  
- **Diffusion Models** – UNet-based diffusion templates for image generation  
- **Audio Models** – WaveNet, Whisper-like encoder-decoders for audio tasks  
- **AdamW Optimizer (AMP-aware)** – Improved stability and convergence  
- **AMP Integration** – Automatic mixed precision for CUDA & XPU  
- **Enhanced DDP** – Unified data/pipeline parallelism with adaptive sync  

### Backend Enhancements
- Optimized tensor memory layout for multi-device systems  
- Adaptive kernel dispatch for Intel XPU (oneAPI) and NVIDIA CUDA  
- Reduced host-device synchronization overhead  

### Developer Experience
- Unified API design for tensors, modules, and optimizers  
- Expanded examples and model presets  
- Extended functional API parity with PyTorch  

---

## Overview

**PySML (Python SHIELD Machine Learning Framework)** provides a flexible, distributed deep learning foundation for multi-device and multi-backend research. Designed for high performance and hardware agnostic scalability.

Key features include:
- Distributed data and pipeline parallel training
- Unified CPU/CUDA/XPU backend API
- AMP-compatible autograd engine
- RWKV, diffusion, and transformer-based templates
- Lightweight, NumPy-compatible syntax

---

## Quick Installation

```bash
pip install numpy
pip install cupy-cuda12x     # For NVIDIA GPUs (CUDA 12.x)
pip install dpnp dpctl       # For Intel Arc / Xe GPUs
```

Verify installation:

```bash
python -c "import pysml; print(pysml.__version__)"
```

---

## Quick Start Examples

### Tensor Operations

```python
import pysml

a = pysml.randn(4, 4, requires_grad=True)
b = pysml.ones(4, 4)

c = pysml.add(a, b)
d = pysml.matmul(c, b.T)
d.backward()

print(a.grad)
```

### Building a Model

```python
import pysml.nn as nn
import pysml.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

model = SimpleMLP()
optimizer = nn.AdamW(model.parameters(), lr=0.001)

for epoch in range(10):
    out = model(x)
    loss = F.cross_entropy(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### AMP Training

```python
from pysml.cuda import backend as cuda_backend
from pysml.nn.optim import AdamW

model = nn.TransformerLM.from_preset("SMALL")
optimizer = AdamW(model.parameters(), lr=0.0001)

with cuda_backend.autocast(enabled=True):
    out = model(x)
    loss = F.cross_entropy(out, y)
    loss.backward()
    optimizer.step()
```

### Distributed Training

```python
from pysml.ddp import DataParallelModel
import pysml.nn as nn

devices = ['cuda:0', 'xpu:0']
model = nn.TransformerLM.from_preset('MEDIUM')
dp_model = DataParallelModel(model, devices)

loss = dp_model.forward_and_backward(inputs, targets)
dp_model.optimizer_step()
```

---

## Backends

| Backend | Library | Hardware | AMP Support | Status |
|----------|----------|-----------|--------------|---------|
| **CPU** | NumPy | Intel / AMD CPUs | No | Stable |
| **CUDA** | CuPy | NVIDIA GPUs | Yes | Stable |
| **XPU** | DPNP / DPCTL | Intel Arc / Xe GPUs | Yes | Stable |

---

## Architecture

```
PySML/
│
├── pysml/
│   ├── tensor.py
│   ├── engine.py
│   ├── nn/
│   │   ├── module.py
│   │   ├── functional.py
│   │   ├── optim.py
│   │   ├── models.py
│   │   └── activations.py
│   ├── ddp/
│   │   ├── data_parallel.py
│   │   ├── pipeline_parallel.py
│   │   ├── device_manager.py
│   │   └── strategies.py
│   ├── cuda/
│   ├── xpu/
│   └── cpu/
└── README.md
```

---

## Example Models

```python
from pysml.nn.models import RWKVModel, DiffusionUNet, AudioEncoderDecoder

# RWKV LLM
rwkv = RWKVModel.from_preset("RWKV-1B")

# Image diffusion
diffusion = DiffusionUNet.from_preset("512x512")

# Audio encoder-decoder
audio_model = AudioEncoderDecoder.from_preset("BaseSpeech")
```

---

## Citation

```bibtex
@software{pysml2025,
  title = {PySML: Python SHIELD Machine Learning Framework},
  author = {S.H.I.E.L.D.},
  year = {2025},
  version = {0.5-alpha1},
  organization = {Strategic Homeland Intervention, Enforcement, and Logistics Division},
  note = {Enterprise Deep Learning Framework with Distributed Training and AMP Support}
}
```

---

## Contact

**S.H.I.E.L.D.**  
Research & Development Division  
Strategic Homeland Intervention, Enforcement, and Logistics Division  

For internal inquiries: `research@shieldapi.org`

---

*Built with ❤️ by the S.H.I.E.L.D. Research Team*

PySML represents years of dedication to making deep learning more accessible, flexible, and powerful. Whether training on Intel Arc, NVIDIA CUDA, or CPUs, PySML ensures consistent performance and usability. Our mission is to advance AI through distributed, hardware-agnostic innovation.

*Advancing AI through hardware-agnostic innovation and distributed training*
