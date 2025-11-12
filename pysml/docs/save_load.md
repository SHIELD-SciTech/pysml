# Saving & Loading Models

PySML’s persistence utilities live in `pysml.save_load` and focus on parity with familiar PyTorch workflows.

## Core Helpers
- `save(obj, path)` / `load(path)` – Serialize arbitrary Python objects (including tensors and state dicts) using `pickle`. Provide file paths or open file objects; optionally supply `map_location` during load to migrate tensors to a specific device.【F:pysml/save_load.py†L1-L39】
- `save_state_dict(model, path)` / `load_state_dict(model, path, strict=True)` – Round-trip `Module.state_dict()` payloads for checkpoints.【F:pysml/save_load.py†L41-L55】

## Training Checkpoints
- `save_checkpoint(model, optimizer, path, epoch=None, loss=None, **metadata)` packages model/optimizer state along with optional metrics, enabling resumable training loops.【F:pysml/save_load.py†L57-L83】
- `load_checkpoint(model, optimizer, path, map_location=None)` restores state and returns any auxiliary metadata saved alongside the weights.【F:pysml/save_load.py†L85-L100】

## Introspection
- `get_model_size(model)` computes total/trainable parameter counts and estimates memory consumption assuming 32-bit floats.【F:pysml/save_load.py†L102-L123】
- `save_model_info(model, path)` writes JSON summaries including architecture string and class name for quick experiment cataloging.【F:pysml/save_load.py†L123-L134】

## Future Formats
Placeholders for ONNX export and Safetensors support document the intended API surface and currently raise `NotImplementedError`. Use the pickle-based functions until these targets are implemented.【F:pysml/save_load.py†L136-L151】
