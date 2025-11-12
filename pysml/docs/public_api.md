# Public API Reference

This guide summarizes the stable surface area of PySML that is intended for downstream projects. The focus is on tensor manipulation, operator dispatch, neural-network modules, and persistence helpers.

## Tensor Core
- **Construction & gradient control** – Instantiate tensors with optional dtype/device arguments and toggle gradient tracking via `.requires_grad_()` or the `requires_grad` constructor flag. The `backward()` implementation performs reverse-mode differentiation and accumulates gradients in-place.【F:pysml/tensor.py†L20-L118】
- **Device management** – Use `.to()`, `.cpu()`, `.cuda()`, and `.xpu()` to migrate data across backends while preserving dtype semantics. Casting can be forced with the `dtype` keyword or by passing another tensor as the device source.【F:pysml/tensor.py†L120-L174】
- **Interoperability helpers** – Methods such as `.numpy()`, `.item()`, slicing via `__getitem__`, and math dunder operators mirror PyTorch behavior, making porting model code straightforward.【F:pysml/tensor.py†L176-L420】

## Operator Dispatcher (`pysml.engine`)
- **Elementwise math** – Functions like `add`, `subtract`, `multiply`, and `divide` broadcast operands, respect scalar shortcuts, and attach matching backward closures when gradients are enabled.【F:pysml/engine.py†L33-L120】
- **Linear algebra & reductions** – `matmul`, `dot`, `sum`, `mean`, `max`, and `min` share the same backend resolution logic, ensuring results are produced on the highest-priority device among the inputs.【F:pysml/engine.py†L1-L32】【F:pysml/engine.py†L121-L220】
- **Modern DL ops** – Softmax, GELU, SiLU, normalization layers, dropout, embeddings, permutation, and masking utilities are all exported through the package initializer for immediate use.【F:pysml/__init__.py†L1-L63】

## Neural Network Stack (`pysml.nn`)
- **Module system** – `Module` implements parameter/buffer registration, recursive traversal, hooks, and call semantics, while `Parameter` wraps tensors that require gradients. Helper containers such as `Sequential`, `ModuleList`, and `ModuleDict` simplify model composition.【F:pysml/nn/module.py†L1-L140】
- **Layer catalog** – The package exports linear, convolutional, normalization, dropout, embedding, recurrent, attention, transformer, positional, and upsampling layers along with standard activation functions.【F:pysml/nn/__init__.py†L1-L142】
- **Optimizers & losses** – SGD, Adam, AdamW, RMSprop, and diverse loss functions (`CrossEntropyLoss`, `MSELoss`, etc.) provide end-to-end training support without additional dependencies.【F:pysml/nn/__init__.py†L143-L213】

## Persistence Utilities
- **State serialization** – `save`/`load` wrap `pickle` with optional device remapping; `save_state_dict` and `load_state_dict` bridge Module checkpoints; `save_checkpoint` bundles optimizer state and metadata for resumable training.【F:pysml/save_load.py†L1-L63】
- **Inspection helpers** – `get_model_size` calculates parameter counts and memory estimates, while `save_model_info` writes JSON metadata for reproducibility.【F:pysml/save_load.py†L64-L108】

## Documentation & Examples
Consult `pysml/examples` for runnable Transformer, RWKV, and diffusion demos, and explore the rest of this `docs/` directory for subsystem deep dives.
