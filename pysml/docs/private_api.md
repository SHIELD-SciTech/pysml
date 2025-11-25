# Private & Experimental API Guide

These interfaces power PySML’s internals and evolve quickly. Use them when contributing new operators, backends, or tooling, and expect to update your code when the engine changes.

## Autograd Utilities

### `pysml.autograd.Function`
- **Reference**: Lightweight graph node defined at the top of `pysml/autograd.py`.【F:pysml/autograd.py†L1-L32】
- **What it does**: Captures tensor inputs, metadata, and a backward callable without requiring subclassing—custom ops simply instantiate `Function` directly.
- **Used in scripts**: Forward operators within `pysml/engine.py` create `Function` instances whenever gradients are enabled; extending the engine involves the same pattern.【F:pysml/engine.py†L33-L220】
- **Typical usage**:
  ```python
  from pysml.autograd import Function

  def relu_forward(x):
      out = backend.maximum(x, 0)
      return Function(inputs=(x,), backward_fn=backward_relu), out
  ```
- **Equivalent APIs**: PyTorch’s `torch.autograd.Function` subclasses, TensorFlow’s `tf.custom_gradient` decorator.

### Gradient toggles (`no_grad`, `set_grad_enabled`)
- **Reference**: Context managers and helpers in `pysml/autograd.py`.【F:pysml/autograd.py†L34-L57】
- **What it does**: Flip the global `_grad_enabled` flag to temporarily disable graph recording (useful for evaluation or kernel benchmarking).
- **Used in scripts**: Example evaluation loops wrap sections with `no_grad()` so inference is cheap even when the rest of the training script enables gradients.【F:pysml/examples/transformer.py†L18-L40】
- **Typical usage**:
  ```python
  from pysml.autograd import no_grad

  with no_grad():
      logits = model(dev_inputs)
  ```
- **Equivalent APIs**: `torch.no_grad`, `tf.device`/`tf.stop_gradient` (achieves similar effect for TensorFlow graphs).

## Engine Helpers

### `_backend(*tensors)`
- **Reference**: Dispatcher helper at the top of `pysml/engine.py`.【F:pysml/engine.py†L1-L32】
- **What it does**: Chooses CPU, CUDA, or XPU kernels based on the highest-priority tensor provided and enforces safety checks when `ENSURE_BACKEND` is enabled.
- **Used in scripts**: Invisible to end users but every operator (e.g., `pysml.matmul`) calls `_backend` internally. Understanding it is required when porting PySML to new devices.【F:pysml/engine.py†L33-L220】
- **Typical usage**:
  ```python
  backend = _backend(tensor_a, tensor_b)
  result = backend.matmul(tensor_a.data, tensor_b.data)
  ```
- **Equivalent APIs**: PyTorch’s dispatcher registry, TensorFlow’s device placement logic.

### Backward registries (`backward_add`, `backward_matmul`, ...)
- **Reference**: Gradient kernels located in `pysml/autograd.py`.【F:pysml/autograd.py†L59-L140】
- **What it does**: Translate upstream gradients plus saved tensors into gradients for each operand; invoked automatically through `Function.apply_backward()`.
- **Used in scripts**: Custom ops or research prototypes register bespoke backward callables to experiment with fused kernels before upstreaming them to the main engine.【F:pysml/engine.py†L121-L220】
- **Typical usage**:
  ```python
  def backward_square(grad_output, inputs, _):
      (x,) = inputs
      return grad_output * 2 * x
  ```
- **Equivalent APIs**: PyTorch’s `backward` static methods in `torch.autograd.Function`, TensorFlow’s gradient registrations via `tf.RegisterGradient`.

## Tensor Internals

### `_new_like` / `_resolve_backend`
- **Reference**: Helper methods on `Tensor` responsible for allocating gradient buffers or migrating data across devices.【F:pysml/tensor.py†L20-L174】
- **What it does**: Ensures that view tensors share metadata, that gradients reuse memory when possible, and that dtype/device combinations remain valid.
- **Used in scripts**: While not called directly, any new backend integration or custom tensor subclass must honor these helpers to remain ABI-compatible with existing operators.【F:pysml/tensor.py†L58-L174】
- **Typical usage**:
  ```python
  grad = tensor._new_like(shape=tensor.shape)
  backend = tensor._resolve_backend(target_device="xpu:0")
  ```
- **Equivalent APIs**: PyTorch’s `Tensor.new_zeros`/`to` implementations, TensorFlow’s tensor factory helpers when writing kernels in C++.

### In-place gradient accumulation
- **Reference**: Implemented within `Tensor.backward()` when combining multiple gradient contributions.【F:pysml/tensor.py†L58-L118】
- **What it does**: Adds new gradients to existing `.grad` tensors via backend-level addition to avoid temporary allocations—critical for large models.
- **Used in scripts**: Research experiments that accumulate gradients across micro-batches rely on this behavior when calling `loss.backward()` repeatedly before `optimizer.step()`.【F:pysml/examples/transformer.py†L68-L117】
- **Equivalent APIs**: PyTorch’s in-place `.grad` accumulation semantics, TensorFlow’s gradient accumulation patterns via `GradientTape`.

## Memory Pool Hooks

### `TensorBufferPool`
- **Reference**: Thread-safe allocator cache defined in `pysml/memory_pool.py`.【F:pysml/memory_pool.py†L1-L48】
- **What it does**: Reuses freed backend buffers keyed by shape/dtype/device to reduce fragmentation and allocation overhead.
- **Used in scripts**: Power users enable the pool when benchmarking throughput-heavy examples such as the diffusion U-Net to stabilize latency across iterations.【F:pysml/examples/diffusion.py†L146-L210】
- **Typical usage**:
  ```python
  from pysml.memory_pool import get_buffer_pool

  pool = get_buffer_pool()
  pool.enable()
  ```
- **Equivalent APIs**: PyTorch CUDA caching allocator (`torch.cuda.empty_cache`), TensorFlow’s BFC allocator knobs.

### Global helpers (`get_buffer_pool`, `enable_buffer_pool`, ...)
- **Reference**: Module-level functions located after the pool implementation.【F:pysml/memory_pool.py†L50-L76】
- **What it does**: Provide programmatic switches to toggle pooling, clear cached buffers, and report statistics.
- **Used in scripts**: The benchmarking helper prints pool stats before and after stress tests so you can visualize allocation reuse.【F:pysml/examples/benchmark.py†L120-L200】
- **Equivalent APIs**: PyTorch’s `torch.cuda.memory_stats`, TensorFlow’s `tf.config.experimental.get_memory_usage` utilities.

## Future-Facing APIs

### Serialization placeholders (`export_onnx`, `save_safetensors`, ...)
- **Reference**: Stub functions near the end of `pysml/save_load.py`.【F:pysml/save_load.py†L110-L150】
- **What it does**: Raise `NotImplementedError` today but document the intended hooks for ONNX and Safetensors support.
- **Used in scripts**: Not yet, but the DDP and persistence docs track these placeholders so contributors know where to add implementations.【F:pysml/docs/ddp.md†L1-L105】【F:pysml/docs/save_load.md†L1-L80】
- **Equivalent APIs**: `torch.onnx.export`, `safetensors.torch.save_file`, TensorFlow’s `tf.saved_model.save`.
