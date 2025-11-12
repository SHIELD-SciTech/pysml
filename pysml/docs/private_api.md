# Private & Experimental API Guide

These interfaces are not considered stable but are useful when extending PySML itself. Expect signatures and behavior to change between releases.

## Autograd Utilities
- **`Function` graph nodes** – Instances store weak references to tensor inputs and a `backward_fn` callable. Subclassing is unnecessary; instead, instantiate `Function` directly and provide metadata dicts for custom kernels.【F:pysml/autograd.py†L1-L32】
- **Gradient toggles** – The `no_grad` context manager and `set_grad_enabled()` helper manipulate a global `_grad_enabled` flag. Use them when writing inference-only utilities or performance-critical sections.【F:pysml/autograd.py†L34-L57】

## Engine Helpers
- **Backend arbitration** – `_backend(*tensors)` inspects candidate tensors and selects the highest-priority backend (CPU → XPU → CUDA). Set `ENSURE_BACKEND = True` during debugging to enforce mixed-backend safety checks.【F:pysml/engine.py†L1-L32】
- **Backward registries** – Backprop functions such as `backward_add`, `backward_multiply`, and friends take gradient outputs plus weak references to operands. They are invoked by `Function.apply_backward()` and must return `(tensor, grad)` tuples or `None` placeholders.【F:pysml/autograd.py†L59-L140】

## Tensor Internals
- **Memory lifecycle** – Tensors store backend handles, dtype info, and device strings; `_new_like` and `_resolve_backend` helpers handle conversions when constructing gradient views or migrating devices. These methods are implementation details and may change without notice.【F:pysml/tensor.py†L20-L118】
- **In-place grad accumulation** – When existing gradients are present, `backward()` uses backend-level addition to avoid creating temporary tensors, falling back to fresh tensor construction if necessary.【F:pysml/tensor.py†L58-L118】

## Memory Pool Hooks
- **`TensorBufferPool`** – Provides thread-safe buffer caching keyed by shape, dtype, backend, and device. Pools cap at 32 buffers per configuration to avoid unbounded growth.【F:pysml/memory_pool.py†L1-L48】
- **Global helpers** – `get_buffer_pool()`, `enable_buffer_pool()`, `disable_buffer_pool()`, `clear_buffer_pool()`, and `get_pool_stats()` operate on the singleton instance used by tensor allocation paths.【F:pysml/memory_pool.py†L50-L76】

## Future-Facing APIs
- **ONNX & Safetensors placeholders** – `export_onnx`, `save_safetensors`, and `load_safetensors` currently raise `NotImplementedError` but document the intended surface for future releases. Use standard pickle-based helpers until these land.【F:pysml/save_load.py†L110-L151】
