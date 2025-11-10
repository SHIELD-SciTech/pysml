# **PySML – Python SHIELD Machine Learning Framework**

> *High-Performance Deep Learning Framework with Multi-Backend Support & Memory Optimization*
> *© S.H.I.E.L.D. / Strategic Homeland Intervention, Enforcement, and Logistics Division*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Version 0.5.1](https://img.shields.io/badge/version-0.5.1-green.svg)](README.md)
[![License: Proprietary](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Backend: CPU/CUDA/XPU](https://img.shields.io/badge/backend-CPU%20%7C%20CUDA%20%7C%20XPU-green.svg)](README.md)
[![Memory: Optimized](https://img.shields.io/badge/memory-50%25%20optimized-brightgreen.svg)](README.md)

---

## Overview

**PySML (Python SHIELD Machine Learning Framework)** is a modular deep learning stack
engineered for research workloads that must span CPUs, NVIDIA CUDA GPUs, and Intel XPU
devices. Release **v0.5.1** focuses on tightening the building blocks that power
Transformers, RWKV-style mixers, and diffusion U-Nets so that the same attention core
and dense projection layers can be reused across radically different architectures.

### What's New in v0.5.1

```
ATTENTION & PROJECTION STABILITY
================================
- Multi-head attention now validates head geometry, applies key/value bias slots,
  and accepts additive or boolean masks for Transformers, RWKV hybrids, and diffusion
  cross-attention.
- Linear, Bilinear, and LazyLinear share a consolidated initialization routine that
  keeps fp16/bf16 projections numerically stable on every backend.

USAGE PLAYBOOK
==============
- New ready-to-run example scripts:
    * pysml/examples/transformer.py  → encoder classifier training loop
    * pysml/examples/rwkv.py         → RWKV-inspired recurrent classifier
    * pysml/examples/diffusion.py    → diffusion-style noise predictor
- README refreshed with quick navigation, setup guidance, and release notes.
```

*Personal note: Getting attention masks working cleanly across the Transformer and RWKV
grid finally makes the codebase feel cohesive rather than a collection of special cases.*

---

## Why PySML?

### Unique Advantages

| Feature | PySML v0.5.1 | PyTorch | TensorFlow | JAX |
|---------|--------------|---------|------------|-----|
| **Memory Efficiency** | 50% optimized | Standard | Standard | Standard |
| **Intel GPU (XPU)** | Native & Fast | Limited | Experimental | None |
| **NVIDIA GPU (CUDA)** | Full Support | Excellent | Full | Full |
| **Multi-Backend** | CPU/CUDA/XPU | CPU/CUDA | CPU/CUDA/TPU | CPU/CUDA/TPU |
| **Transformers** | Production-ready | Extensive | Extensive | Growing |
| **Diffusion Models** | Plug-and-play | Excellent | Good | Growing |
| **RWKV/Mixers** | Supported | Custom | Custom | Custom |
| **Framework Size** | Lightweight | Large | Very Large | Medium |
| **Learning Curve** | PyTorch-like | Medium | Steep | Steep |

### Core Strengths

- **Hardware Agnostic**: Swap between CPU, CUDA, and XPU backends with the same code.
- **Memory Optimized**: LayerNorm, RMSNorm, and view operations avoid redundant buffers.
- **Research Friendly**: PyTorch-like `Module` patterns with explicit parameter control.
- **Example Driven**: Transformer, RWKV, and diffusion walkthroughs ship with the repo.

*Personal note: These examples were written to mirror the mental model I use when porting
architectures between PySML and PyTorch—no hidden helpers, just raw modules.*

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
# CUDA 12.x (Ada, Hopper)
pip install cupy-cuda12x

# CUDA 11.x (Ampere, Turing)
pip install cupy-cuda11x
```

> Requires the NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-downloads

#### Intel GPUs (Arc, Flex, Max)

```bash
pip install dpnp dpctl
```

> Recommended mirror for best performance:
> ```bash
> pip install -i https://software.repos.intel.com/python/pypi numpy dpnp dpctl
> ```

### Verify Installation

```bash
python -c "import pysml; print(f'PySML {pysml.__version__}')"
# Output: PySML 0.5.1

python -c "import pysml; print('CUDA:', pysml.cuda.is_available()); print('XPU:', pysml.xpu.is_available())"
```

*Personal note: If Intel's runtime fails to load, double-check that oneAPI is on your PATH
and that Secure Boot is configured to allow unsigned GPU modules.*

---

## Quick Start

### Hello World: Basic Operations

```python
import pysml

x = pysml.Tensor([[1, 2, 3], [4, 5, 6]])
y = pysml.Tensor([[7, 8, 9], [10, 11, 12]])
print(pysml.add(x, y))
```

### Autograd in Action

```python
import pysml

x = pysml.Tensor([[1.0, 2.0]], requires_grad=True)
w = pysml.Tensor([[3.0], [4.0]], requires_grad=True)
y = x @ w
y.backward()
print('x.grad:', x.grad)
print('w.grad:', w.grad)
```

### Training Skeleton

```python
import pysml
from pysml.nn import Linear, CrossEntropyLoss, AdamW

model = Linear(16, 4)
criterion = CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=3e-4)

for _ in range(10):
    inputs = pysml.Tensor([[0.1] * 16])
    targets = pysml.Tensor([0], requires_grad=False)
    logits = model(inputs)
    loss = criterion(logits, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Model Recipes

### Transformer Encoder Classifier
- File: `pysml/examples/transformer.py`
- Highlights: Embedding + sinusoidal positions + `TransformerEncoder` stack + AdamW loop.

### RWKV-Style Recurrent Classifier
- File: `pysml/examples/rwkv.py`
- Highlights: Time-averaged mixing, per-step state updates, scalar-friendly batching.

### Diffusion Noise Predictor
- File: `pysml/examples/diffusion.py`
- Highlights: Residual U-Net core with bilinear upsampling and MSE noise regression.

---

## Repository Layout

```
pysml/
├── tensor.py              # Core tensor with autograd
├── engine.py              # Backend dispatch + autograd glue
├── nn/                    # Layer zoo (attention, transformer, conv, rwkv helpers)
└── examples/              # Reference scripts for v0.5.1 architectures
```

---

## Release Notes

### v0.5.1
- Attention masks now broadcast correctly for additive and boolean formats.
- Dense layer initialization normalized across Linear, Bilinear, and LazyLinear.
- Added ready-to-run Transformer, RWKV, and diffusion examples with documentation refresh.

*Personal note: The jump from ad-hoc notebooks to polished examples marks the point where I
would trust someone new to the project to build on PySML without a guided tour.*

---

Built with ❤️ by S.H.I.E.L.D.
