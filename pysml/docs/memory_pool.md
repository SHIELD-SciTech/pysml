# Tensor Memory Pooling

The memory pool reduces allocation overhead by caching backend buffers keyed by tensor shape, dtype, backend, and device.

## Lifecycle
- `TensorBufferPool` maintains thread-safe lists of reusable buffers for each key and caps every pool at 32 entries to avoid uncontrolled growth.【F:pysml/memory_pool.py†L1-L44】
- Disabling the pool flushes all cached buffers; re-enabling starts fresh without residual allocations.【F:pysml/memory_pool.py†L12-L44】

## Global Controls
Use the helper functions exported at module scope:
- `get_buffer_pool()` – Access the singleton pool instance.【F:pysml/memory_pool.py†L46-L55】
- `enable_buffer_pool()` / `disable_buffer_pool()` – Toggle caching without restarting your script.【F:pysml/memory_pool.py†L46-L60】
- `clear_buffer_pool()` – Drop all cached buffers immediately.【F:pysml/memory_pool.py†L34-L60】
- `get_pool_stats()` – Inspect the number of tracked pools and total buffers for debugging fragmentation issues.【F:pysml/memory_pool.py†L36-L76】

Pooling is optional—allocation falls back to raw backend arrays when the cache is disabled.
