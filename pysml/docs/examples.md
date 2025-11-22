# Example Gallery

PySML ships a curated set of runnable scripts that mirror the core research areas the framework targets. Every script accepts the same device strings (`"cpu"`, `"cuda"`, `"xpu"`) and will gracefully skip accelerator-only code paths when the corresponding backend is unavailable.

| Domain | Script | What it shows |
| --- | --- | --- |
| Autograd sanity checks | `pysml/examples/autograd_reduction.py` | Reduction correctness and gradient propagation across axes. |
| Transformers | `pysml/examples/transformer_[cpu|cuda|xpu].py` | Encoder/decoder stacks, classification heads, and distributed rehearsal flows. |
| RWKV | `pysml/examples/rwkv_[cpu|cuda|xpu].py` | Streaming-friendly recurrent cells with optional DDP wrappers. |
| Diffusion (image + text prompts) | `pysml/examples/diffusion_[cpu|cuda|xpu].py` | Conditional/unconditional sampling loops and sampler controls. |
| Upscaling / super-resolution | `pysml/examples/upscaling.py` | Bilinear resize with a shallow convolutional refiner that auto-selects CPU/CUDA/XPU. |
| Parallel runtime helpers | `pysml/examples/parallel_utils.py` | Shared configuration objects used by the distributed demos. |

To try any example:

```bash
python -m pysml.examples.transformer_cpu          # CPU Transformer run
python -m pysml.examples.upscaling                # Device auto-detect
python -m pysml.examples.diffusion_cuda --steps 4 # Requires CUDA
```

When adding new examples, mirror this table and ensure the script accepts a `--device` flag or relies on the shared `ExampleConfig` helpers so users can test on their available hardware.
