# Saving & Loading Models

PySML’s persistence helpers mirror PyTorch/TensorFlow workflows while adding distributed-friendly metadata. Each entry below identifies the relevant reference, describes its purpose, points to scripts that use it, and shows the analogous APIs in other frameworks.

## Core Helpers

### `save(obj, path)` / `load(path)`
- **Reference**: Implemented at the top of `pysml/save_load.py`.【F:pysml/save_load.py†L1-L39】
- **What it does**: Pickle arbitrary Python objects (model state dicts, optimizer state, metadata) and optionally remap tensors to a specific device via `map_location`.
- **Used in scripts**: All examples describe saving their `state_dict()` payloads between runs to resume experiments quickly.【F:pysml/examples/transformer.py†L1-L189】【F:pysml/examples/rwkv.py†L1-L255】
- **Typical usage**:
  ```python
  from pysml import save, load

  save(model.state_dict(), "model.pkl")
  state = load("model.pkl", map_location="cuda:0")
  model.load_state_dict(state)
  ```
- **Equivalent APIs**: `torch.save` / `torch.load`, TensorFlow’s `tf.train.Checkpoint.save` / `.restore` (when serializing dictionaries).

### `save_state_dict` / `load_state_dict`
- **Reference**: Convenience wrappers following the base helpers.【F:pysml/save_load.py†L41-L55】
- **What it does**: Persist or load a module’s `state_dict()` with strictness control, keeping parity with PyTorch semantics.
- **Used in scripts**: Example configs mention these helpers when explaining how to checkpoint RWKV time-mix weights mid-training.【F:pysml/examples/rwkv.py†L205-L255】
- **Equivalent APIs**: `torch.nn.Module.state_dict` + `load_state_dict`, TensorFlow’s `model.get_weights` / `set_weights`.

## Training Checkpoints

### `save_checkpoint`
- **Reference**: Mid-file helper bundling model, optimizer, epoch, loss, and metadata.【F:pysml/save_load.py†L64-L110】
- **What it does**: Creates a single file containing both states plus user-defined scalars (e.g., epoch, validation loss) and optional `ParallelStrategy` info.
- **Used in scripts**: The distributed runtime guide (`ddp.md`) shows how to persist parallel strategy metadata between rehearsal runs so repeated launches can reuse weights without recomputing initial states.【F:pysml/docs/ddp.md†L105-L120】
- **Typical usage**:
  ```python
  pysml.save_checkpoint(
      model,
      optimizer,
      "checkpoint.pkl",
      epoch=epoch,
      loss=float(loss.item()),
      strategy=strategy,
  )
  ```
- **Equivalent APIs**: PyTorch’s pattern of saving `{"model": model.state_dict(), "optimizer": opt.state_dict()}`; TensorFlow’s `tf.train.Checkpoint` objects.

### `load_checkpoint`
- **Reference**: Companion loader below the saver.【F:pysml/save_load.py†L112-L130】
- **What it does**: Restores model/optimizer weights, applies `map_location` if requested, and returns metadata (including serialized `ParallelStrategy`) for launch scripts.
- **Used in scripts**: Example docstrings show reading metadata to resume training from the last epoch counter on any backend.【F:pysml/examples/diffusion.py†L146-L210】
- **Equivalent APIs**: PyTorch’s `torch.load` + manual assignment, TensorFlow’s `Checkpoint.restore` with `expect_partial()`.

## Introspection

### `get_model_size`
- **Reference**: Utility near the end of `pysml/save_load.py`.【F:pysml/save_load.py†L102-L123】
- **What it does**: Returns total/trainable parameter counts plus estimated memory usage.
- **Used in scripts**: Benchmark helper prints these stats to contextualize throughput results for Transformer/RWKV/Diffusion runs.【F:pysml/examples/benchmark.py†L1-L120】
- **Typical usage**:
  ```python
  stats = pysml.get_model_size(model)
  print(stats)
  ```
- **Equivalent APIs**: `sum(p.numel() for p in model.parameters())` in PyTorch, `model.count_params()` in TensorFlow/Keras.

### `save_model_info`
- **Reference**: Helper that serializes JSON manifests for experiment tracking.【F:pysml/save_load.py†L123-L150】
- **What it does**: Records class names, architecture descriptors, parameter counts, and optional metadata so you can catalog experiments.
- **Used in scripts**: Benchmark utilities call this function after profiling to snapshot model settings alongside throughput numbers.【F:pysml/examples/benchmark.py†L1-L200】
- **Typical usage**:
  ```python
  pysml.save_model_info(model, "model_info.json", extra={"throughput": samples_per_second})
  ```
- **Equivalent APIs**: PyTorch Lightning’s `Trainer.log_dir` summaries, TensorFlow’s `tf.summary` scalars combined with manual JSON dumps.
- `save_model_info(model, path)` writes JSON summaries including architecture string and class name for quick experiment cataloging.【F:pysml/save_load.py†L123-L134】

## Future Formats
Placeholders for ONNX export and Safetensors support document the intended API surface and currently raise `NotImplementedError`. Use the pickle-based functions until these targets are implemented.【F:pysml/save_load.py†L136-L151】
