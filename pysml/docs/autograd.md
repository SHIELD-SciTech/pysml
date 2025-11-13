# Autograd Internals

PySML records computation graphs lazily, mirroring PyTorch semantics. The following sections map each internal hook to its implementation, describe how it is used by the example scripts, provide quick snippets, and call out equivalent APIs in other frameworks.

## Graph Construction

### `_grad_fn` attachments
- **Reference**: Every tensor stores a `Function` instance in `_grad_fn` when the originating op had gradients enabled.【F:pysml/tensor.py†L58-L118】【F:pysml/autograd.py†L1-L32】
- **What it does**: Captures parents and metadata needed to replay the backward pass; each `Function` also tracks whether it should free buffers once the gradient finishes.
- **Used in scripts**: All training examples rely on this implicit recording when stacking layers—no explicit graph management is required to differentiate Transformer, RWKV, or diffusion workloads.【F:pysml/examples/transformer.py†L41-L189】【F:pysml/examples/rwkv.py†L90-L204】【F:pysml/examples/diffusion.py†L27-L141】
- **Typical usage**: Occurs automatically whenever you call PySML operators with tensors that have `requires_grad=True`.
- **Equivalent APIs**: PyTorch’s autograd graph nodes (`torch.autograd.Function`), TensorFlow’s tape-based graph captured by `tf.GradientTape`.

### Gradient toggles (`no_grad`, `set_grad_enabled`)
- **Reference**: Helpers in `pysml/autograd.py` that flip the global `_grad_enabled` flag.【F:pysml/autograd.py†L34-L57】
- **What it does**: Temporarily disables node creation, keeping inference loops lean even when the rest of the script trains models.
- **Used in scripts**: Evaluation sections inside the Transformer walkthrough wrap generation code with `no_grad()` to avoid extra memory pressure.【F:pysml/examples/transformer.py†L18-L40】
- **Typical usage**:
  ```python
  from pysml.autograd import no_grad

  with no_grad():
      logits = classifier(dev_inputs)
  ```
- **Equivalent APIs**: `torch.no_grad`, TensorFlow’s `tf.stop_gradient`/`tf.GradientTape(persistent=False)` usage.

## Backward Execution

### `Tensor.backward()`
- **Reference**: Method in `pysml/tensor.py` that launches reverse-mode autodiff.【F:pysml/tensor.py†L58-L118】
- **What it does**: Walks the graph in reverse topological order, accumulating gradients in-place and invoking each `Function.apply_backward()`.
- **Used in scripts**: Every training loop (`rwkv.py`, `diffusion.py`, `transformer.py`) invokes `loss.backward()` before optimizer updates, proving the method works across models/backends.【F:pysml/examples/rwkv.py†L205-L255】【F:pysml/examples/diffusion.py†L146-L210】
- **Typical usage**:
  ```python
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()
  ```
- **Equivalent APIs**: `torch.Tensor.backward`, TensorFlow’s `GradientTape.gradient` followed by manual variable updates.

### Backward kernels (`backward_add`, `backward_matmul`, ...)
- **Reference**: Located in `pysml/autograd.py` and called automatically via `Function.apply_backward()`.【F:pysml/autograd.py†L59-L140】
- **What it does**: Compute operand-specific gradients given upstream differentials, handling broadcasting and dtype conversions.
- **Used in scripts**: Not referenced directly, but any custom operation you add for RWKV/Transformer research will import these patterns to stay differentiable.【F:pysml/engine.py†L121-L220】
- **Equivalent APIs**: The backward static methods you implement in `torch.autograd.Function` subclasses; TensorFlow’s gradient registration callbacks.

## Extending the Engine

### Step-by-step recipe
1. **Reference**: Follow the pattern outlined in `pysml/engine.py` for forward ops and `pysml/autograd.py` for backward kernels.【F:pysml/engine.py†L33-L220】
2. **What it does**: Ensures your operator participates fully in PySML’s autodiff system across CPU, CUDA, and XPU.
3. **Used in scripts**: Future diffusion or RWKV kernels can slot in by copying this recipe and updating the example modules to call the new op.
4. **Typical usage**: Implement a forward function that selects a backend, computes outputs, and returns a `Function` capturing the backward callable.
5. **Equivalent APIs**: PyTorch custom C++/Python ops with paired autograd functions, TensorFlow custom ops with registered gradients.

## Comparing to Other Frameworks
- PyTorch similarity: Reverse-mode autodiff with eager graph construction mirrors PyTorch exactly, making porting straightforward.
- TensorFlow similarity: `tf.GradientTape` captures operations lazily like PySML; when persistent tapes are used, the workflow is nearly identical.
