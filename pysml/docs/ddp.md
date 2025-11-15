# Custom Distributed Runtime

PySML ships a pure-Python distributed runtime under ``pysml/ddp`` so model
scripts can rehearse data/pipeline splits across CPU, CUDA, and Intel XPU
without relying on ``torch.distributed`` or vendor-specific launchers. This
document walks through the major building blocks, how they map to the
example scripts, and which persistence helpers keep distributed metadata
replayable.

## Quickstart Checklist
1. **Select devices** – Provide explicit device strings (``"xpu:0"``,
   ``"cuda:0"``, ``"cpu"``) for every participant. The helpers normalize CPU
   strings so ``"cpu"`` is always valid even when lists contain GPU entries.
2. **Register a communicator** – ``ddp.register_global_communicator(devices)``
   seeds a mailbox-based communicator for send/recv/all-reduce calls.
3. **Plan partitions** – ``PartitionConfig`` / ``ParallelStrategy`` describe
   how many layers live on each stage and how many micro-batches flow through
   the pipeline.
4. **Wrap your module** – Use ``PipelineParallel`` and ``DataParallel``
   decorators to split sequential stacks or replicate whole models.
5. **Train / evaluate** – Run the resulting module just like the original.
   The new Transformer and RWKV DDP examples demonstrate the full workflow on
   Intel XPU systems.【F:pysml/examples/ddp_transformer_xpu.py†L1-L120】【F:pysml/examples/ddp_rwkv_xpu.py†L1-L104】

## Communication Primitives (`pysml/ddp/communication/primitives.py`)
- **Reference**: ``Communicator`` plus ``send``, ``recv``, ``all_reduce``,
  ``broadcast``, and ``gather`` live here.【F:pysml/ddp/communication/primitives.py†L1-L210】
- **What it does**: Implements a single-process mailbox where each device has
  an inbox. Tensors hop between devices via ``Tensor.to`` so CUDA↔XPU hops
  automatically stage through CPU memory, while reductions use NumPy to keep
  semantics deterministic.
- **Typical usage**:
  ```python
  from pysml.ddp.communication import primitives as comms

  comm = comms.Communicator(["xpu:0", "xpu:1", "cpu"])
  comm.send(tensor, dst="xpu:1", src="xpu:0")
  replica = comm.recv("xpu:1", src="xpu:0")
  reduced = comm.all_reduce(replica, op="sum", tag="loss")
  ```
- **Equivalent APIs**: ``torch.distributed`` collectives or TensorFlow’s
  ``tf.distribute`` replica contexts, but implemented purely with PySML
  tensors so they run anywhere NumPy is available.

## Partition Planning (`pysml/ddp/config/partition_config.py`)
- **Reference**: ``DeviceSpec``, ``PartitionConfig``, and ``ParallelStrategy``
  are implemented in this module.【F:pysml/ddp/config/partition_config.py†L1-L170】
- **What it does**: Normalizes device strings (``"cpu"`` vs ``"cuda:0"``),
  estimates per-device memory/compute budgets, validates world sizes, and
  emits layer counts per stage for ``PipelineParallel``.
- **Typical usage**:
  ```python
  from pysml.ddp.config.partition_config import build_partition_config

  plan = build_partition_config(num_layers=12, devices=["xpu:0", "cuda:0"], chunks=4)
  print(plan.partitions)  # -> [6, 6]
  ```
- **Equivalent APIs**: PyTorch’s pipeline plans or TensorFlow GPipe layouts,
  but represented as plain dataclasses so configs can be serialized alongside
  checkpoints.

## Parallel Wrappers (`pysml/ddp/parallel/`)
- **Reference**: ``PipelineParallel`` and ``DataParallel`` live under
  ``pysml/ddp/parallel``.【F:pysml/ddp/parallel/pipeline_parallel.py†L1-L70】【F:pysml/ddp/parallel/data_parallel.py†L1-L100】
- **What they do**:
  - ``PipelineParallel`` inspects ``Sequential``/``Module`` children, slices
    them into stages based on the resolved ``PartitionConfig``, moves each
    stage to the specified device, and returns a ``PipelineModule`` with the
    requested micro-batch schedule.
  - ``DataParallel`` deep-copies a module constructor across devices, splits
    batch dimension zero across replicas, and concatenates the results on the
    lead device. Gradients flow independently per replica, making the helper
    ideal for deterministic inference or supervised rehearsal loops.
- **Equivalent APIs**: ``torch.distributed.pipeline.sync.PipelineModule`` and
  ``torch.nn.parallel.DistributedDataParallel`` without NCCL/Gloo bindings.

## Example: Transformer on Intel XPU
- **Reference**: ``pysml/examples/ddp_transformer_xpu.py`` exposes
  ``train_classifier_with_pipeline`` and ``data_parallel_logits`` helpers.
  【F:pysml/examples/ddp_transformer_xpu.py†L1-L120】
- **Workflow**:
  1. Resolve devices via ``_preferred_devices`` (prefers XPU, falls back to
     CUDA/CPU).
  2. Register a communicator and wrap ``TinyTransformerClassifier`` with
     ``PipelineParallel`` so encoder layers live on separate XPUs.
  3. Optimise with ``AdamW`` + ``CrossEntropyLoss`` and inspect loss history.
  4. Optionally wrap the pipeline in ``DataParallel`` to batch inference
     requests across replicas.
- **CLI snippet**:
  ```bash
  python -m pysml.examples.ddp_transformer_xpu
  ```

## Example: RWKV on Intel XPU
- **Reference**: ``pysml/examples/ddp_rwkv_xpu.py`` mirrors the transformer
  example but pipelines RWKV embedding/block/head stages.【F:pysml/examples/ddp_rwkv_xpu.py†L1-L104】
- **Workflow**:
  1. Build a ``Sequential`` stack from ``build_rwkv_stages`` so each stage is
     explicit.
  2. Feed the constructor to ``PipelineParallel`` with per-stage device names
     plus micro-batch counts.
  3. Optimise with ``Adam`` and reuse ``DataParallel`` for batched inference.

## Checkpointing & Resume
- **Reference**: ``save_checkpoint`` / ``load_checkpoint`` store model,
  optimizer, and ``ParallelStrategy`` metadata in one file so DDP rehearsals
  can resume on a different host.【F:pysml/save_load.py†L64-L130】
- **Workflow**:
  ```python
  from pysml import save_checkpoint, load_checkpoint
  from pysml.ddp import ParallelStrategy

  strategy = ParallelStrategy.pipeline(stages=2, chunks=4)
  save_checkpoint(model, optimizer, "rwkv-ddp.pkl", strategy=strategy, epoch=epoch)
  state = load_checkpoint(model, optimizer, "rwkv-ddp.pkl")
  restored = ParallelStrategy.from_dict(state["distributed_state"]["strategy"])
  ```
- **Tip**: Attach the ``strategy`` metadata emitted by your launcher so reruns
  can recreate the same partition plan without recomputing heuristics.
