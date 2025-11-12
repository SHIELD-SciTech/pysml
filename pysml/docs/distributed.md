# Distributed Training

PySML ships with a lightweight distributed runtime that mirrors the ergonomics
of popular frameworks while keeping optional dependencies truly optional.
Collective communication is exposed via the :mod:`pysml.distributed`
package which automatically selects an appropriate backend (Gloo for CPU,
NCCL for CUDA, oneCCL for Intel XPU) and gracefully falls back to local
execution when the required libraries are missing.

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
