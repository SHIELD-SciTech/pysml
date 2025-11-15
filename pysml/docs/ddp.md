# Custom Distributed Runtime

PySML’s new distributed stack lives entirely inside `pysml/ddp/` and avoids dependencies on `torch.distributed`, oneCCL, or TensorFlow. The modules described below mirror the most frequently requested features from the old implementation—collectives, partition planning, and lightweight wrappers—while remaining portable across CPU, CUDA, and Intel XPU backends.

## Communication Primitives (`pysml/ddp/communication/primitives.py`)
- **Reference**: `Communicator`, `send`, `recv`, `all_reduce`, and friends are defined here.【F:pysml/ddp/communication/primitives.py†L1-L210】
- **What it does**: Provides single-process mailboxes that simulate device-to-device messaging. CUDA↔CUDA and XPU↔XPU transfers reuse `Tensor.to()` to mimic peer-to-peer copies, while cross-backend hops route through CPU memory to keep semantics deterministic.
- **Typical usage**:
  ```python
  from pysml.ddp.communication import primitives as comms
  comm = comms.Communicator(["cuda:0", "xpu:0", "cpu:0"])
  comm.send(tensor, "xpu:0", src="cuda:0")
  replica = comm.recv("xpu:0", src="cuda:0")
  reduced = comm.all_reduce(replica, op="sum", tag="loss")
  ```
- **Equivalent APIs**: `torch.distributed` collectives or TensorFlow’s `tf.distribute.ReplicaContext.all_reduce`, but implemented purely with PySML tensors.

## Partition Planning (`pysml/ddp/config/partition_config.py`)
- **Reference**: `DeviceSpec`, `PartitionConfig`, and `ParallelStrategy` live here.【F:pysml/ddp/config/partition_config.py†L1-L190】
- **What it does**: Detects device strings (e.g., `"cuda:0"`, `"xpu:0"`), estimates relative memory/compute headroom, and emits layer counts per stage. `ParallelStrategy` captures user intent (data/pipeline/tensor degrees) and serializes into checkpoints so launches stay reproducible.
- **Typical usage**:
  ```python
  from pysml.ddp.config.partition_config import build_partition_config
  plan = build_partition_config(num_layers=12, devices=["xpu:0", "cuda:0", "cpu:0"], chunks=4)
  print(plan.partitions)  # -> [5, 4, 3]
  ```
- **Equivalent APIs**: PyTorch’s `torch.distributed.pipeline.sync.PipelineModule` partition planners or TensorFlow GPipe layouts, but simplified for single-process rehearsal.

## Parallel Wrappers (`pysml/ddp/parallel/`)
- **Reference**: `PipelineParallel` and `DataParallel` orchestrators.【F:pysml/ddp/parallel/pipeline_parallel.py†L1-L70】【F:pysml/ddp/parallel/data_parallel.py†L1-L100】
- **What they do**:
  - `PipelineParallel` wraps a module constructor, slices sequential layers into stages according to a `PartitionConfig`, and wires them through `pysml.nn.pipeline.PipelineModule` with configurable micro-batches.
  - `DataParallel` deep-copies a module across devices, splits batch tensors along the 0th dimension, runs each replica, and concatenates the outputs on the lead device.
- **API example**:
  ```python
  from pysml import ddp
  Model = ddp.PipelineParallel(MySequentialModel, devices=["xpu:0", "cuda:0"], chunks=4)
  model = Model(config)
  replicated = ddp.DataParallel(lambda: model, devices=["cuda:0", "cuda:1"])
  data_parallel_model = replicated()
  ```
- **Equivalent APIs**: PyTorch’s `torch.distributed.pipeline.sync` and `DistributedDataParallel`, but implemented via PySML tensors and NumPy fallbacks so they work on systems without NCCL/Gloo.

Together these components form a cohesive workflow: use `ParallelStrategy` to describe the target topology, inspect the resulting `PartitionConfig`, rely on `PipelineParallel`/`DataParallel` to wrap modules, and coordinate gradients or metrics via the communication primitives. The new stack is intentionally lightweight so that extending it (adding new collective verbs, refining partition heuristics) requires editing only a handful of pure-Python files.
