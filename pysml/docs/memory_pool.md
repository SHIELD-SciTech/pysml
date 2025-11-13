# Tensor Memory Pooling

High-throughput workloads (diffusion sampling, Transformer inference, etc.) benefit from reusing backend buffers instead of constantly reallocating them. The `pysml.memory_pool` module exposes opt-in helpers with the following structure.

## `TensorBufferPool`
- **Reference**: Class defined in `pysml/memory_pool.py`.【F:pysml/memory_pool.py†L1-L48】
- **What it does**: Caches backend buffers keyed by shape/dtype/device with a 32-buffer limit per key to prevent unbounded growth.
- **Used in scripts**: Benchmark helper toggles the pool when comparing latency between CPU, CUDA, and XPU runs of the diffusion example.【F:pysml/examples/benchmark.py†L80-L200】
- **Typical usage**:
  ```python
  from pysml.memory_pool import TensorBufferPool

  pool = TensorBufferPool()
  pool.enable()
  ```
- **Equivalent APIs**: PyTorch’s CUDA caching allocator, TensorFlow’s BFC allocator knobs.

## Module-level helpers

### `get_buffer_pool`
- **Reference**: Returns the singleton pool instance.【F:pysml/memory_pool.py†L46-L55】
- **What it does**: Provides global access so tensors and custom ops share the same cache.
- **Used in scripts**: Examples call `get_buffer_pool().enable()` once at startup to stabilize performance.【F:pysml/examples/diffusion.py†L146-L210】
- **Equivalent APIs**: `torch.cuda.memory_allocated` (for inspection) combined with `torch.cuda.empty_cache` when manually controlling caches.

### `enable_buffer_pool` / `disable_buffer_pool`
- **Reference**: Toggles caching in `pysml/memory_pool.py`.【F:pysml/memory_pool.py†L50-L60】
- **What it does**: Turns pooling on/off without recreating the singleton—useful for A/B tests.
- **Typical usage**:
  ```python
  from pysml.memory_pool import enable_buffer_pool, disable_buffer_pool

  enable_buffer_pool()
  ...  # benchmark
  disable_buffer_pool()
  ```
- **Equivalent APIs**: `torch.cuda.empty_cache` for forcing releases; TensorFlow’s experimental memory growth controls.

### `clear_buffer_pool` / `get_pool_stats`
- **Reference**: Maintenance helpers near the bottom of the module.【F:pysml/memory_pool.py†L34-L76】
- **What they do**: `clear_buffer_pool()` drops all cached buffers immediately, while `get_pool_stats()` reports how many keys/buffers remain for debugging fragmentation.
- **Used in scripts**: Benchmark utilities print stats before/after long runs to verify buffers are reused as expected.【F:pysml/examples/benchmark.py†L120-L200】
- **Equivalent APIs**: PyTorch’s `torch.cuda.memory_stats`, TensorFlow’s `tf.config.experimental.get_memory_usage`.

Pooling is optional—when disabled, tensors fall back to raw backend allocations exactly like vanilla NumPy/CuPy/dpnp code.
