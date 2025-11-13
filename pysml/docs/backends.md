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

## CUDA Toolkit Quick Checklist
1. Install the matching CuPy wheel for your driver (`cupy-cuda11x` or `cupy-cuda12x`). The adapter automatically reuses NumPy as a fallback, so the import never explodes in CI environments without GPUs.【F:pysml/cuda/backend.py†L1-L40】
2. Export `CUDA_VISIBLE_DEVICES` to restrict which cards PySML will see.
3. Double-check `nvidia-smi` reports the same driver/runtime combo you compiled CuPy against; version mismatches usually manifest as `CUDA_ERROR_INVALID_DEVICE`. Reinstall the correct wheel if you upgraded your driver.
4. Set `PYSML_DIST_DEVICE=cuda` when launching distributed jobs to force NCCL selection even if a CPU tensor is allocated first.【F:pysml/docs/distributed.md†L1-L54】

## Intel oneAPI / XPU Setup
1. Install `dpnp` and `dpctl` from Intel's PyPI mirror (`pip install -i https://software.repos.intel.com/python/pypi dpnp dpctl`).【F:pysml/xpu/__init__.py†L1-L5】
2. Source the oneAPI environment script so level-zero drivers are on `LD_LIBRARY_PATH`.
3. Verify `python -c "import dpnp; print(dpnp.__version__)"` before running PySML to catch missing runtime libraries early.
4. Pass `PYSML_DIST_DEVICE=xpu` to distributed launches so the communicator picks oneCCL automatically.【F:pysml/distributed/backends/oneccl.py†L1-L120】

## Troubleshooting
- **`cupy.cuda.runtime.CUDARuntimeError: device not ready`** – confirm no other process is locking the GPU and rerun after `nvidia-smi --reset-gpu`. PySML's CUDA adapter exposes `pysml.cuda.synchronize()` to flush pending kernels when debugging timing issues.【F:pysml/cuda/backend.py†L120-L180】
- **`dpnp` complains about missing level-zero driver** – ensure the oneAPI `setvars.sh` script is sourced in non-interactive shells (e.g., add it to your systemd service unit) before importing PySML.
- **Pipelines hang when mixing CPU and accelerator tensors** – the dispatcher always prefers the highest priority backend. Explicitly move tensors via `tensor.to("cpu")` or `tensor.cuda()` to keep communication predictable.【F:pysml/examples/parallel_utils.py†L1-L80】
