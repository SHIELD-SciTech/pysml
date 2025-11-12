# Autograd Internals

PySML uses a lightweight reverse-mode autodiff engine that records computation graphs lazily. Understanding its mechanics helps when adding new operators or debugging gradient issues.

## Graph Construction
- Every tensor resulting from an operation stores a `Function` instance in `_grad_fn`. The instance keeps weak references to input tensors and a backward callable that consumes the upstream gradient.【F:pysml/autograd.py†L1-L32】【F:pysml/tensor.py†L58-L118】
- The engine records dependency edges only when gradients are enabled. Wrap inference code with `pysml.autograd.no_grad()` or toggle `set_grad_enabled(False)` to bypass graph creation.【F:pysml/autograd.py†L34-L57】

## Backward Execution
- `Tensor.backward()` performs a reverse topological walk starting from the target tensor, accumulating gradients and invoking each node’s `apply_backward()` helper.【F:pysml/tensor.py†L58-L118】
- Backward kernels live in `pysml.autograd` (e.g., `backward_add`, `backward_multiply`, `backward_matmul`) and are registered by the forward operators inside `pysml.engine`. Each kernel must respect broadcasting semantics and return gradient tensors that match the original operand shapes.【F:pysml/autograd.py†L59-L140】【F:pysml/engine.py†L33-L120】

## Extending the Engine
1. **Write the forward op** in `pysml.engine`, using `_backend()` to choose a device and `Function(...)` to capture the backward kernel and metadata.
2. **Implement the backward kernel** in `pysml.autograd`, following existing patterns for broadcasting, scalar handling, and tensor reconstruction.
3. **Expose the op** from `pysml.__init__` if it should be part of the public API.

Following this recipe keeps gradients differentiable across CPU, CUDA, and XPU backends with minimal boilerplate.【F:pysml/engine.py†L1-L120】【F:pysml/__init__.py†L1-L63】
