# Distributed Training

PySML ships with a lightweight distributed runtime that mirrors the ergonomics
of popular frameworks while keeping optional dependencies truly optional.
Collective communication is exposed via the :mod:`pysml.distributed`
package which automatically selects an appropriate backend (Gloo for CPU,
NCCL for CUDA, oneCCL for Intel XPU) and gracefully falls back to local
execution when the required libraries are missing.

## Key APIs and Concepts

### `init_process_group` / `shutdown`
- **Reference**: Process bootstrap helpers inside `pysml/distributed/__init__.py`.【F:pysml/distributed/__init__.py†L1-L61】
- **What it does**: Creates or tears down process groups using environment variables (`WORLD_SIZE`, `RANK`, etc.) plus optional overrides like `PYSML_INIT_METHOD`.
- **Used in scripts**: Launcher snippets in every example refer to these helpers when demonstrating how to rehearse Transformer/RWKV splits on multi-device setups.【F:pysml/examples/transformer.py†L1-L200】
- **Typical usage**:
  ```python
  from pysml.distributed import init_process_group, shutdown

  init_process_group(backend="nccl")
  ...  # training loop
  shutdown()
  ```
- **Equivalent APIs**: `torch.distributed.init_process_group`, `tf.distribute.Server` initialization for multi-worker jobs.

### Collective helpers (`all_reduce`, `broadcast`, ...)
- **Reference**: Thin wrappers located in `pysml/distributed/collectives.py` and re-exported from the package init.【F:pysml/distributed/__init__.py†L1-L61】
- **What it does**: Provide PyTorch-style functions that operate on PySML tensors, automatically selecting NCCL/Gloo/oneCCL based on device.
- **Used in scripts**: Loss synchronization examples in this document demonstrate calling `all_reduce` inside multi-GPU loops; DDP wrappers rely on the same primitives under the hood.【F:pysml/docs/distributed.md†L1-L120】
- **Typical usage**:
  ```python
  from pysml.distributed import all_reduce

  loss = criterion(output, target)
  all_reduce(loss)
  loss /= world_size
  ```
- **Equivalent APIs**: `torch.distributed.all_reduce`, TensorFlow collective ops exposed via `tf.distribute.ReplicaContext.all_reduce`.

### `ParallelStrategy`
- **Reference**: High-level orchestrator in `pysml/distributed/strategy.py`.【F:pysml/distributed/strategy.py†L1-L170】
- **What it does**: Encodes data/pipeline/tensor degrees, validates them against the current world size, and wraps modules in the correct order.
- **Used in scripts**: `ExampleConfig.build_strategy()` feeds directly into `ParallelStrategy.apply()` so each example can toggle between pure data parallelism and hybrid schemes.【F:pysml/examples/parallel_utils.py†L11-L83】
- **Typical usage**:
  ```python
  strategy = ParallelStrategy.hybrid(data=2, pipeline=2, tensor=2, schedule="gpipe")
  model = strategy.apply(module, pipeline_stages=stages)
  ```
- **Equivalent APIs**: PyTorch combinations of DDP + pipeline modules + tensor-parallel sharding; TensorFlow’s `tf.distribute.MultiWorkerMirroredStrategy` combined with manual model partitioning.

### `PipelineModule`
- **Reference**: Prototype pipeline scheduler inside `pysml/nn/pipeline.py`.【F:pysml/nn/pipeline.py†L1-L220】
- **What it does**: Splits sequential modules into stages, simulates GPipe/1F1B schedules, and records instrumentation; currently executes all stages in-process while stubbing out rank hand-offs.
- **Used in scripts**: Transformer and RWKV examples create four-stage lists and feed them to `ParallelStrategy.pipeline(...)` to rehearse stage boundaries even though execution remains local.【F:pysml/examples/transformer.py†L119-L200】【F:pysml/examples/rwkv.py†L220-L280】
- **Typical usage**:
  ```python
  stages = build_transformer_stages(vocab_size=256)
  strategy = ParallelStrategy.pipeline(stages=len(stages), schedule="1f1b", chunks=4)
  model = strategy.apply(pipeline_stages=stages)
  ```
- **Equivalent APIs**: `torch.distributed.pipeline.sync.PipelineModule`, TensorFlow’s GPipe implementations (e.g., Mesh TensorFlow), albeit PySML’s version is currently single-process only.

### `DistributedMonitor`
- **Reference**: Metrics helper in `pysml/distributed/monitor.py`.【F:pysml/distributed/monitor.py†L1-L160】
- **What it does**: Aggregates throughput/loss metrics via all-reduce and prints rank-aware logs without scattering conditional statements through training loops.
- **Used in scripts**: Showcased later in this document’s monitoring section, which mirrors how you might track RWKV or diffusion experiments on clusters.【F:pysml/docs/distributed.md†L80-L140】
- **Typical usage**:
  ```python
  monitor = DistributedMonitor(total_steps=100)
  monitor.update(batch_size=len(batch), loss=float(loss))
  monitor.flush()
  ```
- **Equivalent APIs**: PyTorch Lightning/TorchMetrics progress trackers, TensorFlow’s `tf.keras.callbacks.History` combined with `tf.summary` logging.

### Rank-aware checkpointing (`save_rank_checkpoint`, `load_rank_checkpoint`)
- **Reference**: Helpers in `pysml/distributed/checkpointing.py`.【F:pysml/distributed/checkpointing.py†L1-L243】
- **What it does**: Saves one shard per rank plus a manifest describing topology, allowing elastic restarts and tensor-parallel-aware persistence.
- **Used in scripts**: Distributed doc’s checkpointing section shows how to pair these helpers with pipeline rehearsals so RWKV or diffusion jobs can resume after crashes.【F:pysml/docs/distributed.md†L140-L200】
- **Typical usage**:
  ```python
  from pysml.distributed.checkpointing import save_rank_checkpoint

  save_rank_checkpoint(model, optimizer, "./ckpt_dir")
  ```
- **Equivalent APIs**: PyTorch’s `torch.distributed.checkpoint` utilities, TensorFlow’s parameter-server-style `tf.train.Checkpoint` sharding.

## Environment Driven Initialisation

The runtime inspects common launch environment variables when the first tensor
queries its communicator. When variables such as ``WORLD_SIZE`` or ``RANK`` are
present, the process group is initialised automatically using the backend that
best matches the visible hardware:

```bash
# Single node example using torchrun style environment variables
export WORLD_SIZE=4
export RANK=0
export LOCAL_RANK=0
python train.py
```

If the environment indicates a single process (``WORLD_SIZE=1`` or unset) the
runtime configures an in-process ``dummy`` backend that keeps the code path
identical without incurring any network overhead. Advanced users can force a
specific backend via ``PYSML_DIST_BACKEND`` or steer the heuristic with
``PYSML_DIST_DEVICE``.

## Single Node, Multi-GPU

1. Launch one worker per GPU. ``torchrun`` and ``mpirun`` both work; PySML only
   requires the standard ``RANK``/``WORLD_SIZE`` variables to be set.
2. Ensure CUDA devices are visible (``CUDA_VISIBLE_DEVICES``) and leave
   ``PYSML_DIST_BACKEND`` unset so the runtime automatically selects NCCL.
3. Inside your training script call collective helpers directly on PySML
   tensors:

```python
from pysml.distributed import all_reduce
...
loss = loss_tensor.communicator.all_reduce(loss_tensor)
```

All tensors created on CUDA devices route to the NCCL communicator. CPU tensors
continue to use the Gloo fallback which is useful for host side coordination.

## Multi-Node Clusters

For multi-node deployments supply an explicit initialisation method
(e.g. ``tcp://host:port``) or use your cluster launcher of choice. Set
``PYSML_INIT_METHOD`` to the rendezvous endpoint to avoid duplicating
configuration in every script. Example using two nodes with four GPUs each:

```bash
# On the head node
export WORLD_SIZE=8
export RANK=0
export MASTER_ADDR=10.0.0.5
export MASTER_PORT=29500
export PYSML_INIT_METHOD=tcp://10.0.0.5:29500
python train.py
```

Subsequent ranks should update ``RANK`` (and ``LOCAL_RANK`` if desired) while
sharing the same rendezvous URL. The process group bootstrap reuses the
configuration and selects NCCL/oneCCL/Gloo depending on the device reported by
individual tensors.

## Manual Control

Developers who prefer explicit control can bypass the automatic initialisation
and call :func:`pysml.distributed.init_process_group` manually. The returned
communicator object exposes ``rank`` and ``world_size`` properties alongside
standard collective operations. Calling :func:`pysml.distributed.shutdown`
resets the state so the process can join a new job safely.

Refer to :mod:`pysml.distributed.collectives` for thin wrappers around
``all_reduce``, ``broadcast``, ``all_gather`` and ``reduce_scatter`` that follow
PySML's tensor semantics.

### Fault Tolerance & Timeouts

Collectives now honour the ``PYSML_COLLECTIVE_TIMEOUT`` and
``PYSML_COLLECTIVE_RETRIES`` environment variables. Wrappers such as
``pysml.distributed.all_reduce`` guard each operation with a configurable
deadline and retry loop. When a timeout or failure occurs a
``CollectiveOperationError`` is raised and subsequent collectives abort early to
prevent additional deadlocks. This behaviour provides clear diagnostics for
stalled ranks and keeps the remaining workers from hanging indefinitely.

### Monitoring & Metrics Hooks

Use :class:`pysml.distributed.DistributedMonitor` to collect throughput and loss
metrics without sprinkling conditional ``if rank == 0`` checks through the
training loop. The monitor supports context-managed batch timing, periodic
reports, and callback hooks to integrate progress bars or experiment trackers.
All metrics are aggregated with ``all_reduce`` so every log line represents the
global view across ranks.

```python
from pysml.distributed import DistributedMonitor

monitor = DistributedMonitor(total_steps=len(train_loader), log_every=20)
for batch, target in train_loader:
    loss = step(batch, target)
    monitor.update(batch_size=len(batch), loss=float(loss))
monitor.flush()  # final summary
```

### Distributed Checkpointing

:mod:`pysml.distributed.checkpointing` exposes ``save_rank_checkpoint`` and
``load_rank_checkpoint`` helpers that persist one shard per rank along with a
``manifest.json`` describing the layout. The manifest enables elastic restarts
– when the world size changes the loader maps new ranks onto the saved shards
round-robin. Combine the helper with ``tensor_parallel_shard_state_dict`` to
persist only the tensor-parallel slice owned by each worker.

### Debugging Multi-Process Runs

The :mod:`pysml.distributed.debugging` module adds two frequently requested
tools:

* ``launch_rank_repl`` opens an interactive console whose namespace already
  contains ``rank`` and ``world_size`` so you can introspect local tensors
  without leaving the running job.
* ``register_gradient_anomaly_detector`` instruments a module with post-backward
  hooks that flag NaNs, infinities, or exploding gradients. You can supply a
  custom callback (for logging or paging) or rely on the default exception.

These hooks are rank-local which makes them safe to enable even on large jobs –
each worker only inspects its own gradients and reports anomalies with its rank
ID for quick triage.

## Composite Parallel Strategies

PySML exposes a :class:`~pysml.distributed.ParallelStrategy` helper to describe
how a model should be sharded across data, pipeline, and tensor parallel
dimensions. The object keeps the configuration in a single place, validates it
against the detected ``WORLD_SIZE``, and provides convenience constructors. The
following listing mirrors the layout that PyTorch users are familiar with but
adds explicit commentary about how the three axes interact:

```python
from pysml.distributed import ParallelStrategy

# Single convenience axis
dp = ParallelStrategy.data(degree=4)
tp = ParallelStrategy.tensor(degree=2, mode="2d", dims=(2, 1))
pp = ParallelStrategy.pipeline(stages=3, schedule="1f1b", chunks=4)

# Mixed configuration with checkpointed pipeline activations. The constructor
# always validates the request against WORLD_SIZE so you immediately discover
# whether the current launch can host the desired hybrid layout.
hybrid = ParallelStrategy.hybrid(
    data=2,
    pipeline=4,
    tensor=2,
    schedule="1f1b",
    chunks=2,
    activation_checkpoint=True,
)
```

Calling :meth:`ParallelStrategy.apply` performs the wrapping order automatically
(tensor parallel initialisation, followed by pipeline partitioning, then DDP
replicas):

```python
model = TinyTransformerClassifier()
strategy = ParallelStrategy.hybrid(data=2, pipeline=4, tensor=2)
model = strategy.apply(model, pipeline_stages=custom_stages)
```

### Distributed Pipeline Readiness (CPU, CUDA, and Intel XPU)

`PipelineModule` currently runs every stage within a single process and uses
collectives only to ship tensors to the next stage placeholder. There is no
rank-to-rank rendezvous yet – the helper simply calls
``dist_collectives.gather(..., dst=0)`` or ``send(..., dst=0)`` on each stage
output while ignoring the optional ``stage_ranks`` argument. In practice this
means pipeline parallel execution behaves like a highly instrumented sequential
pass regardless of whether the tensors live on CPU, CUDA, or Intel XPU devices.
The implementation is intentionally structured so distributed hand-off can be
added later, but the necessary per-stage rank management is still missing.

Consequently PySML is **not yet ready** for cross-rank pipeline parallelism on
Intel XPUs (or any other backend). RWKV, Transformer, and Diffusion examples can
be wrapped in `PipelineModule`, but each stage continues to execute within the
local interpreter. Keep this limitation in mind when evaluating performance or
memory scaling claims – today’s pipeline support provides scheduling and
profiling scaffolding rather than genuine distributed execution.

### Example: Prototyping RWKV Pipeline Splits on XPU

Even without cross-rank communication you can rehearse stage boundaries on XPU
hardware and gather profiling data. The example modules in ``pysml.examples``
now document end-to-end snippets for CUDA, CPU, and XPU. A typical RWKV rehearsal
looks like this:

```python
from pysml.examples import rwkv
from pysml.examples.parallel_utils import ExampleConfig

cfg = ExampleConfig(
    backend="xpu",
    pipeline_parallel=4,
    pipeline_schedule="1f1b",
    pipeline_chunks=4,
    activation_checkpoint=True,
)

# Prints latency/memory metrics for each stage so you can tune partitions.
rwkv.demonstrate_pipeline_segments(vocab_size=256, config=cfg)
```

The same pattern applies to the Transformer and Diffusion demonstrations – the
`ExampleConfig` exposes the backend and pipeline knobs in one place, while each
example exports ``train_example`` and ``demonstrate_pipeline_*`` helpers with
rich docstrings describing the workflow.

When ``pipeline_parallel`` exceeds ``1`` you must provide ``pipeline_stages`` –
a list of :class:`~pysml.nn.Module` objects that represent each stage. PySML does
not yet infer cross-rank partitions automatically. Tensor-parallel helpers are
activated transparently via :func:`pysml.distributed.tensor_parallel.init_tensor_parallel`.

```mermaid
flowchart LR
    subgraph Replica0[Data Replica 0]
        P0S0(Stage 0 / TP Shard 0)
        P0S1(Stage 1 / TP Shard 1)
        P0S0 --> P0S1
    end
    subgraph Replica1[Data Replica 1]
        P1S0(Stage 0 / TP Shard 0)
        P1S1(Stage 1 / TP Shard 1)
        P1S0 --> P1S1
    end
    P0S1 == gradient sync ==> P1S1
```

### Validation & Failure Modes

``ParallelStrategy.validate()`` inspects the current process group and raises
clear exceptions when a request cannot be fulfilled:

| Symptom | Explanation | Fix |
| --- | --- | --- |
| ``tensor_parallel=4`` with ``WORLD_SIZE=2`` | Not enough ranks to host the requested tensor shards. | Launch more processes or reduce the tensor degree. |
| ``pipeline_parallel>1`` without ``pipeline_stages`` | PySML cannot infer stage boundaries. | Pass an explicit list of stages to ``apply()``. |
| ``data_parallel>1`` but ``WORLD_SIZE=1`` | DDP needs multiple processes. | Use ``torchrun``/``mpirun`` or lower the data parallel degree. |

If the runtime runs on a single process the validation relaxes the relationship
between pipeline degree and world size so you can prototype pipelined models on
one device before scaling out.

## Scaling & Resource Utilisation Best Practices

* Prefer smaller ``log_every`` values in :class:`DistributedMonitor` when
  experimenting; bump the interval for production runs to reduce log volume.
* Set ``PYSML_COLLECTIVE_TIMEOUT`` to a value slightly higher than your longest
  expected batch duration to automatically flag deadlocks.
* When checkpointing, write shards to a shared filesystem and keep
  ``manifest.json`` under version control – it documents the parallel strategy
  and world size that produced the checkpoint.
* Use ``register_gradient_anomaly_detector`` together with small batches before
  scaling up so numerical issues are caught early.
