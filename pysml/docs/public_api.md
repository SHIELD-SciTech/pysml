# Public API Reference

The following catalog explains each stable PySML API in plain language so newcomers can connect the surface area to real scripts. Every entry includes a reference back to the implementation, the examples that rely on it, a runnable snippet, and pointers to the closest PyTorch and TensorFlow equivalents.

## Tensor Core

### `pysml.tensor.Tensor`
- **Reference**: Defined in `pysml/tensor.py`; underpins every array created inside PySML.【F:pysml/tensor.py†L20-L420】
- **What it does**: Holds multi-backend data, tracks gradient provenance through `_grad_fn`, and exposes NumPy/PyTorch-like conveniences (`.numpy()`, `.item()`, math dunders) for ergonomic model code.【F:pysml/tensor.py†L58-L420】
- **Used in scripts**: All example models create tensors directly; e.g., `pysml/examples/transformer.py` builds integer tensors for tokenized batches via `Tensor(np_array, requires_grad=False)`.【F:pysml/examples/transformer.py†L166-L187】
- **Typical usage**:
  ```python
  from pysml import Tensor

  embeddings = Tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True, device="cuda:0")
  loss = (embeddings ** 2).sum()
  loss.backward()
  ```
- **Equivalent APIs**: `torch.Tensor`, `tensorflow.Tensor` / `tf.Variable`.

### Device migration helpers (`Tensor.to/.cpu/.cuda/.xpu`)
- **Reference**: Convenience wrappers around `_resolve_backend` and backend adapters inside `pysml/tensor.py`.【F:pysml/tensor.py†L120-L174】
- **What it does**: Copies or views tensor storage on CPU, CUDA, or Intel XPU backends, ensuring dtype promotion is preserved and gradient history stays intact.
- **Used in scripts**: `ExampleConfig.apply()` migrates models and pipeline stages to user-chosen devices before wrapping parallel strategies.【F:pysml/examples/parallel_utils.py†L53-L83】
- **Typical usage**:
  ```python
  logits = model(inputs)
  probs = logits.softmax(axis=-1)
  probs_cpu = probs.cpu()
  ```
- **Equivalent APIs**: `torch.Tensor.to/device`, `tf.Tensor.numpy()`/`tf.identity` with specific device scopes.

### Gradient control (`Tensor.requires_grad_`, `Tensor.backward`)
- **Reference**: Methods in `pysml/tensor.py` that toggle `_requires_grad` and execute reverse-mode autodiff when `backward()` is invoked.【F:pysml/tensor.py†L58-L118】
- **What it does**: Records computational graphs during forward passes and propagates gradients to leaf tensors with optional gradient arguments (for non-scalars).
- **Used in scripts**: RWKV and diffusion demos call `loss.backward()` followed by optimizer steps to train toy models.【F:pysml/examples/rwkv.py†L205-L255】【F:pysml/examples/diffusion.py†L146-L210】
- **Typical usage**:
  ```python
  prediction = model(batch)
  criterion = pysml.nn.CrossEntropyLoss()
  loss = criterion(prediction, labels)
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()
  ```
- **Equivalent APIs**: `torch.Tensor.backward`, `tf.GradientTape` (context-managed equivalent for TensorFlow).

## Operator Dispatcher (`pysml.engine`)

### Elementwise math (`pysml.add`, `pysml.multiply`, ...)
- **Reference**: Implemented in `pysml/engine.py` and re-exported from `pysml/__init__.py`.【F:pysml/engine.py†L33-L120】【F:pysml/__init__.py†L1-L63】
- **What it does**: Performs broadcasting-aware arithmetic with automatic backend selection and registers matching backward kernels through `Function` nodes.
- **Used in scripts**: Activation functions and loss helpers across `pysml/examples` rely on these primitives when building residual connections, gating terms, or custom metrics.【F:pysml/examples/transformer.py†L68-L117】【F:pysml/examples/rwkv.py†L150-L204】
- **Typical usage**:
  ```python
  hidden = pysml.add(residual, feedforward)
  scaled = pysml.multiply(hidden, 0.5)
  ```
- **Equivalent APIs**: `torch.add`, `torch.mul`, `tf.math.add`, `tf.math.multiply`.

### Linear algebra & reductions (`pysml.matmul`, `pysml.sum`, ...)
- **Reference**: High-performance kernels defined in `pysml/engine.py` (matrix ops) and autograd closures in `pysml/autograd.py`.【F:pysml/engine.py†L121-L220】【F:pysml/autograd.py†L59-L140】
- **What it does**: Implements GEMM, dot products, pooling, and reduction operators that underpin attention heads, RWKV time mixing, and diffusion convolutions.
- **Used in scripts**: Transformer attention blocks and RWKV mixers both depend on `matmul` and `sum` via the `pysml.nn` layers they instantiate.【F:pysml/examples/transformer.py†L41-L117】【F:pysml/examples/rwkv.py†L118-L204】
- **Typical usage**:
  ```python
  context = pysml.matmul(query, key.transpose(0, 2, 1))
  pooled = pysml.sum(context, axis=-1)
  ```
- **Equivalent APIs**: `torch.matmul`, `torch.sum`, `tf.linalg.matmul`, `tf.reduce_sum`.

## Neural Network Stack (`pysml.nn`)

### `pysml.nn.Module` and `pysml.nn.Parameter`
- **Reference**: Core module container defined in `pysml/nn/module.py`.【F:pysml/nn/module.py†L1-L140】
- **What it does**: Handles parameter registration, recursive `.to()` migration, hooks, and state dict serialization—mirroring PyTorch semantics for effortless porting.
- **Used in scripts**: Every example defines subclasses such as `TinyTransformerClassifier`, `RWKVLanguageModel`, or `UNetDiffusion` that inherit from `Module`.【F:pysml/examples/transformer.py†L41-L163】【F:pysml/examples/rwkv.py†L90-L204】【F:pysml/examples/diffusion.py†L27-L141】
- **Typical usage**:
  ```python
  from pysml.nn import Module, Linear

  class MLP(Module):
      def __init__(self, d_model):
          super().__init__()
          self.fc1 = Linear(d_model, 4 * d_model)
          self.fc2 = Linear(4 * d_model, d_model)

      def forward(self, x):
          return self.fc2(self.fc1(x)).relu()
  ```
- **Equivalent APIs**: `torch.nn.Module`, `tf.keras.layers.Layer`.

### Layer catalog (linear, attention, diffusion-friendly blocks)
- **Reference**: Exported from `pysml/nn/__init__.py`, covering linear layers, convolutions, embeddings, normalization, attention, Transformer blocks, and diffusion-friendly utilities like `SinusoidalPositionalEncoding`.【F:pysml/nn/__init__.py†L1-L142】
- **What it does**: Supplies ready-made building blocks to compose Transformers, RWKV mixers, and convolutional U-Nets without reimplementing math primitives.
- **Used in scripts**: The Transformer demo wires encoder/decoder stacks, RWKV uses custom time-mix layers built atop linear + activation modules, and the diffusion walkthrough constructs residual conv blocks with LayerNorm and SiLU.【F:pysml/examples/transformer.py†L41-L163】【F:pysml/examples/rwkv.py†L90-L204】【F:pysml/examples/diffusion.py†L27-L141】
- **Typical usage**:
  ```python
  encoder_layer = pysml.nn.TransformerEncoderLayer(d_model=128, num_heads=8)
  model = pysml.nn.TransformerEncoder(encoder_layer, num_layers=6)
  ```
- **Equivalent APIs**: `torch.nn.Linear`, `torch.nn.LayerNorm`, `torch.nn.TransformerEncoderLayer`; TensorFlow’s `tf.keras.layers.Dense`, `tf.keras.layers.LayerNormalization`, `tf.keras.layers.MultiHeadAttention`.

### Optimizers & losses
- **Reference**: Implemented in `pysml/nn/optim/*.py` and re-exported via `pysml/nn/__init__.py`.【F:pysml/nn/__init__.py†L143-L213】
- **What it does**: Provides SGD, Adam, AdamW, RMSprop, and cross-entropy/MSE style loss functions used to train examples end-to-end.
- **Used in scripts**: Each demo instantiates an optimizer (often AdamW) and matching loss criterion before entering the training loop.【F:pysml/examples/transformer.py†L30-L117】【F:pysml/examples/rwkv.py†L205-L255】【F:pysml/examples/diffusion.py†L146-L210】
- **Typical usage**:
  ```python
  optimizer = pysml.nn.AdamW(model.parameters(), lr=1e-3)
  criterion = pysml.nn.CrossEntropyLoss()
  loss = criterion(logits, labels)
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()
  ```
- **Equivalent APIs**: `torch.optim.AdamW`, `torch.nn.CrossEntropyLoss`, `tf.keras.optimizers.Adam`, `tf.keras.losses.SparseCategoricalCrossentropy`.

## Distributed Runtime (`pysml.distributed`)

### `ParallelStrategy`
- **Reference**: Configuration helper in `pysml/distributed/strategy.py`.【F:pysml/distributed/strategy.py†L1-L170】
- **What it does**: Encapsulates data, pipeline, and tensor parallel degrees; validates topology; and wraps modules with tensor-parallel shards, pipeline partitions, and DDP replicas in a deterministic order.
- **Used in scripts**: `ExampleConfig.build_strategy()` constructs hybrid strategies so the Transformer, RWKV, and diffusion demos can be launched with varying degrees of parallelism on CPU, CUDA, or Intel XPU.【F:pysml/examples/parallel_utils.py†L11-L83】
- **Typical usage**:
  ```python
  from pysml.distributed import ParallelStrategy

  strategy = ParallelStrategy.hybrid(data=2, pipeline=2, tensor=2, schedule="1f1b")
  wrapped_model = strategy.apply(module, pipeline_stages=stages)
  ```
- **Equivalent APIs**: PyTorch’s `torch.distributed.fsdp` + `torch.distributed.pipeline.sync.PipelineModule` combos; TensorFlow’s `tf.distribute.MultiWorkerMirroredStrategy`/`ParameterServerStrategy` stacks.

### `DistributedDataParallel`
- **Reference**: Implementation in `pysml/distributed/ddp.py`.【F:pysml/distributed/ddp.py†L1-L200】
- **What it does**: Wraps `Module` instances to synchronize gradients via all-reduce, bucket optimizer state, and expose rank/world metadata.
- **Used in scripts**: Advanced launcher templates reference DDP for scaling the provided examples once multi-process backends are configured; the distributed doc walks through rehearsal steps on Intel XPU hardware.【F:pysml/docs/distributed.md†L1-L200】
- **Typical usage**:
  ```python
  from pysml.distributed import DistributedDataParallel as DDP

  ddp_model = DDP(model)
  output = ddp_model(batch)
  ddp_model.backward_and_sync(output, target)
  ```
- **Equivalent APIs**: `torch.nn.parallel.DistributedDataParallel`, `tf.distribute.ReplicaContext.all_reduce` within mirrored strategies.

## Persistence Utilities (`pysml.save_load`)

### `save` / `load`
- **Reference**: Serializer helpers in `pysml/save_load.py`.【F:pysml/save_load.py†L1-L55】
- **What it does**: Pickle arbitrary Python objects (modules, tensors, metadata) with optional `map_location` support to retarget tensors during load.
- **Used in scripts**: Examples demonstrate saving model checkpoints between epochs to resume experiments across CPU, CUDA, and XPU hosts (see docstrings in each example module).【F:pysml/examples/transformer.py†L1-L189】【F:pysml/examples/rwkv.py†L1-L220】【F:pysml/examples/diffusion.py†L1-L170】
- **Typical usage**:
  ```python
  from pysml import save, load

  save(model.state_dict(), "transformer.pkl")
  state = load("transformer.pkl", map_location="cuda:0")
  model.load_state_dict(state)
  ```
- **Equivalent APIs**: `torch.save` / `torch.load`, `tf.train.Checkpoint.save` / `Checkpoint.restore`.

### `save_checkpoint` / `load_checkpoint`
- **Reference**: Extended persistence helpers at the bottom of `pysml/save_load.py`.【F:pysml/save_load.py†L64-L130】
- **What it does**: Capture model + optimizer state, epoch counters, loss metrics, and optional `ParallelStrategy` metadata so distributed runs can resume safely.
- **Used in scripts**: Distributed walkthroughs show how to stash RWKV/Transformer states between pipeline rehearsal runs to avoid recomputing random inits.【F:pysml/docs/distributed.md†L120-L200】
- **Typical usage**:
  ```python
  metadata = {"epoch": epoch, "loss": float(loss.item())}
  pysml.save_checkpoint(model, optimizer, "checkpoint.pkl", **metadata)
  state = pysml.load_checkpoint(model, optimizer, "checkpoint.pkl")
  ```
- **Equivalent APIs**: PyTorch’s `torch.save` with `state_dict` dictionaries, TensorFlow’s `tf.train.Checkpoint` combined with `tf.train.CheckpointManager`.

### Model inspection (`get_model_size`, `save_model_info`)
- **Reference**: Utility functions at the end of `pysml/save_load.py`.【F:pysml/save_load.py†L102-L150】
- **What it does**: Count trainable parameters, estimate memory footprints, and persist JSON manifests that describe architectures for reproducibility.
- **Used in scripts**: Benchmark helpers print parameter counts for each example configuration before running timing loops so you can compare models apples-to-apples.【F:pysml/examples/benchmark.py†L1-L200】
- **Typical usage**:
  ```python
  stats = pysml.get_model_size(model)
  print(f"Parameters: {stats['trainable']} trainable")
  pysml.save_model_info(model, "model_info.json", extra=stats)
  ```
- **Equivalent APIs**: `torchinfo.summary` / manual `sum(p.numel())`, TensorFlow’s `model.count_params()` or `tf.keras.utils.plot_model` for metadata dumps.

## Documentation & Examples
- **Reference**: Explore `pysml/docs/` (deep dives) and `pysml/examples/` (Transformer, RWKV, diffusion) to see every API in context.【F:pysml/docs/README.md†L1-L18】【F:pysml/examples/transformer.py†L1-L200】
- **What it does**: Provides narrative explanations plus runnable code so even first-time users can ramp up quickly.
- **Used in scripts**: Start with `pysml/examples/parallel_utils.py` to configure devices/parallelism, then run the model-specific modules to observe the APIs described above in action.【F:pysml/examples/parallel_utils.py†L1-L83】
