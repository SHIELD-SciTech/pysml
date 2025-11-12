# Backend Abstraction

PySML keeps front-end code device agnostic by routing every tensor operation through a backend-specific implementation. Three backends ship with the framework.

## CPU (NumPy)
- Always available and used as the fallback when other accelerators are missing. Helper functions report device metadata such as processor name, architecture, and core count.【F:pysml/cpu/__init__.py†L1-L55】

## NVIDIA CUDA (CuPy)
- Wraps CuPy arrays when available and falls back to NumPy otherwise. Conversion routines respect explicit device indices (e.g., `cuda:1`) and map common math primitives to CuPy implementations.【F:pysml/cuda/backend.py†L1-L84】
- The public module exports `is_available()`, device enumeration, and synchronization helpers, mirroring PyTorch’s CUDA utilities.【F:pysml/cuda/__init__.py†L1-L5】

## Intel XPU (dpnp/dpctl)
- Targets Intel GPUs through dpnp/dpctl with an interface parallel to the CUDA adapter. The module exposes availability checks and device helpers so code can switch to `tensor.xpu()` seamlessly.【F:pysml/xpu/__init__.py†L1-L5】

## Backend Selection
- Forward operators consult `_backend(*tensors)` to choose the highest-priority device among the inputs (CPU → XPU → CUDA). You can override this behavior by calling `.to(device)` or constructing tensors directly on the desired backend.【F:pysml/engine.py†L1-L32】【F:pysml/tensor.py†L58-L118】
