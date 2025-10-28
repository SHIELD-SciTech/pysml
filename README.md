# **PySML – Python SHIELD Machine Learning Framework**

> *Enterprise‑Grade Deep Learning with Multi‑Device Training and Full Backend Support*  
> *© S.H.I.E.L.D. / Strategic Homeland Intervention, Enforcement, and Logistics Division*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Backend: CPU/CUDA/XPU](https://img.shields.io/badge/backend-CPU%20%7C%20CUDA%20%7C%20XPU-green.svg)](README.md)
[![Version: 0.4.8](https://img.shields.io/badge/version-0.4.8-brightgreen.svg)](README.md)

---

## What's New in v0.4.8

**Targeted Optimizations – Memory & Speed**

This release focuses on **two primary optimizations** across the core stack:

### 🚀 Speed
- **In‑place optimizer updates (SGD/Adam)**: parameter arrays updated without allocating temporaries (‑**1.3× fewer allocations** on inner loops).
- **Faster autograd**: smaller closures store **only minimal ctx** (data/shape flags), with broadcasting‑aware reducers.
- **Vectorized softmax backward** and streamlined element‑wise grads (ReLU/Sigmoid/Tanh/Exp/Log).
- **Backend‑first math**: avoids Python round‑trips; favors ufuncs on NumPy/CuPy/DPNP wherever available.

### 🧠 Memory
- **Compact `Tensor` objects** using `__slots__` (reduces per‑tensor overhead).
- **Aggressive graph freeing** after backward; `no_grad` context for inference/initialization.
- **Fewer copies**: dtype casts and `.clone()` avoid redundant allocations; broadcasting reducers reuse shapes.
- **Device‑neutral `.to()`** uses host‑only hop one time (numpy ↔ backend) to avoid chain conversions.

> Expect **~10–25% faster training steps** and **notable memory savings** in typical MLP/Transformer blocks (numbers vary by backend and model size).

---

## Overview

PySML provides a **PyTorch‑like API** with **multi‑device** backends:
- **CPU** (NumPy), **CUDA** (CuPy), **XPU** (DPNP/DPCTL)
- Data & pipeline primitives (DDP folder) for multi‑device scaling

---

## Highlights

- **Full Autograd** with minimal‑ctx, broadcasting‑aware grads
- **Optimizers** (SGD, Adam) rewritten for in‑place, allocation‑free updates
- **Stability**: numerically stable softmax / cross‑entropy path
- **Interoperability**: easy `.to("cpu"|"cuda:0"|"xpu:0")` and `.numpy()`
- **Dev Ergonomics**: `no_grad()` for init/inference, simple training loops

---

## Quick Example

```python
import pysml
import pysml.nn.functional as F

# device can be "cpu", "cuda:0", or "xpu:0"
pysml.engine.set_device("cpu")

w = pysml.randn(128, 64, requires_grad=True)
x = pysml.randn(16, 128)
y = pysml.randn(16, 64)

out = pysml.matmul(x, w)
loss = F.mse_loss(out, y)

loss.backward()        # builds + frees graph
opt = pysml.nn.optim.Adam([w], lr=1e-3)
opt.step()             # in-place update
opt.zero_grad()
```

---

## Installation (Backends)

- **CPU**: `pip install numpy`
- **CUDA (NVIDIA)**: `pip install cupy-cuda12x` (or appropriate CUDA build)
- **XPU (Intel Arc/Xe)**: `pip install dpnp dpctl`  
  *Recommended channel for best perf:*  
  `pip install -i https://software.repos.intel.com/python/pypi numpy dpnp dpctl`

---

## API Notes

### Autograd & Memory
- Use `with pysml.no_grad():` around weight init / eval paths to **skip graph building**.
- After `.backward()`, PySML **clears node parents** to release memory; call `tensor.zero_grad()` between steps.

### Optimizers
- **SGD** supports momentum/Nesterov; **Adam** supports AMSGrad state via `amsgrad=True`.
- Both perform **in‑place parameter updates** to avoid excess allocations.

### Devices
- `engine.get_available_devices()` lists detected devices (e.g., `["cpu","cuda:0","xpu:0"]`).  
- `tensor.to("cuda:0")` or `module.to("xpu:0")` migrates arrays using a single host hop.

---

## Changelog (0.4.6 → 0.4.8)

- Reworked `Tensor` (`__slots__`, `no_grad`, safer `.astype`, faster `.clone`)  
- Leaner `engine` element‑wise ops and reducers  
- Optimizers rewritten for **in‑place updates** and **fewer temporaries**  
- Stability fixes in softmax/log/exp paths  
- README refreshed; version bump to **0.4.8**

---

## Examples

Check out comprehensive examples in the repository:

- **Distributed training examples**: Multi-GPU training workflows
- **Preset model examples**: Using ready-to-use configurations
- **Custom model examples**: Building models from scratch
- **Optimization examples**: Learning rate scheduling and advanced techniques

---

## Contributing

This is a proprietary research framework for internal use at S.H.I.E.L.D. External contributions are not currently accepted.

For internal contributors:
1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Ensure backward compatibility
5. Test on all supported backends (CPU/CUDA/XPU)

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
  version = {0.4.6},
  organization = {Strategic Homeland Intervention, Enforcement, and Logistics Division},
  note = {Enterprise Deep Learning Framework with Distributed Training}
}
```

---

*Built with ❤️ by the S.H.I.E.L.D. Research Team*

PySML represents years of dedication to making deep learning more accessible, flexible, and powerful. We believe that groundbreaking AI research shouldn't be limited by hardware constraints or framework lock-in. Whether you're training on a laptop CPU, a cutting-edge NVIDIA GPU, or Intel's Arc graphics cards, PySML provides the same elegant API and robust performance. Our mission is to empower researchers and engineers to focus on what matters most—pushing the boundaries of what's possible with machine learning. Thank you for being part of this journey with us.

*Advancing AI through hardware-agnostic innovation and distributed training*
