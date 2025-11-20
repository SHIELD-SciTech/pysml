# PySML Documentation

This directory is your onboarding ramp into PySML. Each guide mixes narrative context, explicit API references, runnable snippets, and cross-links to the example scripts (`pysml/examples`) so a first-time reader can gain deep familiarity without digging through source files blindly.

## Contents at a Glance

| Document | Purpose | Example tie-ins |
| --- | --- | --- |
| [Public API Reference](public_api.md) | Enumerates the stable user-facing surface with references, usage snippets, and PyTorch/TensorFlow equivalents. | Transformer, RWKV, and diffusion demos all highlight these APIs in their docstrings.【F:pysml/docs/public_api.md†L1-L154】 |
| [Private & Experimental API Guide](private_api.md) | Explains the internals (autograd, dispatcher, memory pool) needed when extending PySML itself. | Useful when modifying operators used by the example suite or building custom backends.【F:pysml/docs/private_api.md†L1-L147】 |
| [Autograd Internals](autograd.md) | Describes how computation graphs are recorded and executed, with guidance for adding new ops and verifying reductions. | Mirrors the steps taken in `pysml/examples/transformer.py` when composing differentiable modules and in the reduction sanity-check demo.【F:pysml/docs/autograd.md†L1-L40】【F:pysml/examples/autograd_reduction.py†L1-L46】 |
| [Backend Abstraction](backends.md) | Lists CPU/CUDA/XPU adapter capabilities and troubleshooting tips. | Use alongside `ExampleConfig` to select the right device before launching demos.【F:pysml/docs/backends.md†L1-L40】 |
| [Tensor Memory Pooling](memory_pool.md) | Documents the buffer cache used during high-throughput workloads. | Enables deterministic benchmarking for the diffusion example.【F:pysml/docs/memory_pool.md†L1-L40】 |
| [Saving & Loading Models](save_load.md) | Shows how to serialize checkpoints, gather stats, and plan for distributed recovery. | Shared across all demos that need resumable training.【F:pysml/docs/save_load.md†L1-L60】 |
| [Custom Distributed Runtime](ddp.md) | Explains the new communication primitives, partition planner, and pipeline/data parallel wrappers. | Required reading before wrapping demos with the DDP helpers introduced in `pysml/ddp/`.【F:pysml/docs/ddp.md†L1-L105】 |

## Distributed Training At a Glance

1. Pick the devices you want to target (`"cuda:0"`, `"xpu:0"`, `"cpu"`).
2. Register the communicator via `ddp.register_global_communicator(devices)`.
3. Create a `ParallelStrategy` (or pass `devices` directly) and wrap your module with `ddp.PipelineParallel` and/or `ddp.DataParallel`.
4. Launch one of the rehearsal scripts such as `python -m pysml.examples.ddp_transformer_xpu` to validate the plan before integrating it into your own training code.

The [DDP guide](ddp.md) contains complete walkthroughs along with RWKV and Transformer snippets so you can mirror the same structure in custom projects.

Each document now includes “reference / what it does / scripts / usage / equivalents” callouts per API entry to remove guesswork for new users. Contributions are welcome—please mirror the structure when documenting additional modules.
