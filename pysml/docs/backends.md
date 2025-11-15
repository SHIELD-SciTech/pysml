# Backend Abstraction

PySML keeps front-end code device agnostic by routing every tensor operation through backend adapters. Use the following cheat sheet to decide which backend to enable and how each maps to PyTorch/TensorFlow counterparts.

## CPU Backend (`pysml.cpu`)

- **Reference**: Implemented in `pysml/cpu/__init__.py`; always available.【F:pysml/cpu/__init__.py†L1-L55】
- **What it does**: Wraps NumPy arrays, exposes device metadata (architecture, core count), and serves as the fallback when accelerators are absent.
- **Used in scripts**: Default execution target for every example; e.g., running `pysml/examples/rwkv.py` without flags executes entirely on CPU.【F:pysml/examples/rwkv.py†L1-L255】
- **Typical usage**:
  ```python
  from pysml import Tensor

  tensor = Tensor([[1, 2], [3, 4]], device="cpu")
  ```
- **Equivalent APIs**: PyTorch CPU tensors (`device="cpu"`), TensorFlow running on host via `/CPU:0`.

## NVIDIA CUDA Backend (`pysml.cuda`)

- **Reference**: Adapter defined in `pysml/cuda/backend.py` with helpers re-exported via `pysml/cuda/__init__.py`.【F:pysml/cuda/backend.py†L1-L84】【F:pysml/cuda/__init__.py†L1-L5】
- **What it does**: Uses CuPy arrays for data storage/computation, honors explicit device indices (e.g., `cuda:1`), and mirrors PyTorch’s CUDA utilities like `is_available()` and stream sync helpers.
- **Used in scripts**: `ExampleConfig(device="cuda:0")` in any example migrates weights + tensors onto this backend for GPU acceleration.【F:pysml/examples/parallel_utils.py†L11-L83】
- **Typical usage**:
  ```python
  if pysml.cuda.is_available():
      logits = model(inputs.cuda())
  ```
- **Equivalent APIs**: `torch.cuda` device helpers, TensorFlow’s `/GPU:0` placements.

### CUDA Toolkit Checklist
1. Install the matching CuPy wheel for your driver (`cupy-cuda11x` or `cupy-cuda12x`).
2. Export `CUDA_VISIBLE_DEVICES` if you want to mask GPUs from PySML.

## Intel XPU Backend (`pysml.xpu`)

- **Reference**: Thin adapter around dpnp/dpctl inside `pysml/xpu/__init__.py`.【F:pysml/xpu/__init__.py†L1-L5】
- **What it does**: Provides Intel GPU tensors with APIs parallel to the CUDA adapter so `.xpu()` calls behave just like `.cuda()`.
- **Used in scripts**: The custom DDP guide walks through rehearsing pipeline splits on XPU hardware, and `ExampleConfig` accepts `backend="xpu"` to target these devices.【F:pysml/docs/ddp.md†L1-L105】【F:pysml/examples/parallel_utils.py†L11-L83】
- **Typical usage**:
  ```python
  if pysml.xpu.is_available():
      batch = batch.to("xpu:0")
      loss = model(batch).mean()
  ```
- **Equivalent APIs**: PyTorch’s `torch.xpu` (via Intel Extension for PyTorch), TensorFlow with oneAPI plugins (`/XPU:0`).

## Backend Selection Logic

- **Reference**: `_backend(*tensors)` in `pysml/engine.py` selects CPU → XPU → CUDA priority unless you override via `.to(device)`.【F:pysml/engine.py†L1-L32】
- **What it does**: Keeps mixed-backend operations deterministic by running them on the “highest” accelerator present while falling back to CPU when needed.
- **Used in scripts**: All ops issued by the example models go through this selection logic, enabling multi-backend parity with zero code changes.【F:pysml/examples/transformer.py†L41-L189】
- **Equivalent APIs**: PyTorch’s device propagation rules, TensorFlow’s implicit device placement heuristics.
3. Double-check `nvidia-smi` reports the same driver/runtime combo you compiled CuPy against; version mismatches usually manifest as `CUDA_ERROR_INVALID_DEVICE`. Reinstall the correct wheel if you upgraded your driver.
4. When mixing accelerators, explicitly move tensors via `.cuda()`/`.xpu()` before invoking the custom DDP helpers so the simulated communicator knows which device metadata to record.

## Intel oneAPI / XPU Setup
1. Install `dpnp` and `dpctl` from Intel's PyPI mirror (`pip install -i https://software.repos.intel.com/python/pypi dpnp dpctl`).【F:pysml/xpu/__init__.py†L1-L5】
2. Source the oneAPI environment script so level-zero drivers are on `LD_LIBRARY_PATH`.
3. Verify `python -c "import dpnp; print(dpnp.__version__)"` before running PySML to catch missing runtime libraries early.
4. Configure the `pysml.ddp` communicator with explicit `"xpu:N"` entries (e.g., `Communicator(["xpu:0", "cpu:0"])`) so pipeline and data parallel wrappers can map stages to your preferred GPUs without guessing.

## Troubleshooting
- **`cupy.cuda.runtime.CUDARuntimeError: device not ready`** – confirm no other process is locking the GPU and rerun after `nvidia-smi --reset-gpu`. PySML's CUDA adapter exposes `pysml.cuda.synchronize()` to flush pending kernels when debugging timing issues.【F:pysml/cuda/backend.py†L120-L180】
- **`dpnp` complains about missing level-zero driver** – ensure the oneAPI `setvars.sh` script is sourced in non-interactive shells (e.g., add it to your systemd service unit) before importing PySML.
- **Pipelines hang when mixing CPU and accelerator tensors** – the dispatcher always prefers the highest priority backend. Explicitly move tensors via `tensor.to("cpu")` or `tensor.cuda()` to keep communication predictable.【F:pysml/examples/parallel_utils.py†L1-L80】
