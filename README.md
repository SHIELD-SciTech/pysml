# PySML

PySML (Python Strategic Hardware-Independent Learning) is a research-grade deep learning framework that mirrors the ergonomics of PyTorch while targeting CPU, NVIDIA CUDA, and Intel XPU from a single codebase. The library emphasizes explicit control over tensors, modules, and device placement so experimenters can prototype new architectures without juggling backend-specific forks.

## Project Highlights
- **Unified tensor core** – `pysml.tensor.Tensor` centralizes device-aware storage, gradient tracking, and dtype conversions while exposing familiar helpers such as `.to()`, `.cuda()`, and `.backward()`.
- **Composable autograd** – A minimal `Function` graph records backward closures for every op registered in `pysml.engine`, enabling custom differentiable primitives without boilerplate.
- **Backend dispatch** – CPU (NumPy), CUDA (CuPy), and Intel XPU (dpnp/dpctl) backends share a common operator surface and can be toggled at runtime.
- **PyTorch-inspired modules** – `pysml.nn` ships parameters, buffers, optimizers, attention layers, and Transformer building blocks ready for research-scale experiments.
- **Custom distributed runtime** – `pysml.ddp` provides pipeline/data parallel wrappers and pure-Python communication primitives so you can rehearse heterogeneous launches without `torch.distributed`.
- **First-party utilities** – Checkpointing, model-size reporting, memory pooling, and computation-graph visualization streamline day-to-day experimentation.

> Personal note: The rewrite that unified attention, normalization, and optimizer utilities finally made it possible to port my Transformer notebooks between laptop CPU runs and datacenter GPUs without changing a line of model code.

## Repository Map
```
pysml/
├── tensor.py          # Tensor implementation and autograd integration
├── engine.py          # Backend-aware operator dispatch
├── autograd.py        # Function graph and backward kernels
├── nn/                # Module base class, layers, optimizers, attention blocks
├── save_load.py       # Checkpointing and model-size helpers
├── memory_pool.py     # Device/dtype aware buffer pooling
├── utils.py           # Graph tracing helpers
└── docs/              # Detailed API and subsystem documentation
```

## Installation
### Python Requirements
- Python 3.8+
- NumPy 1.20 or newer

### CPU-Only Setup
```bash
pip install numpy
```

### Optional GPU Backends
- **NVIDIA CUDA** – `pip install cupy-cuda11x` or `cupy-cuda12x` (matching your driver toolkit)
- **Intel XPU** – `pip install dpnp dpctl` (consider Intel's `-i https://software.repos.intel.com/python/pypi` mirror for faster wheels)

### From Source
```bash
git clone https://github.com/SHIELD-SciTech/pysml.git
cd pysml
python -m pip install -e .
```

After installation you can verify backend availability:
```bash
python -c "import pysml; print('PySML', pysml.__version__); print('CUDA:', pysml.cuda.is_available()); print('XPU:', pysml.xpu.is_available())"
```

> Personal note: When Intel's drivers complain, I check that oneAPI is on `PATH` **and** that Secure Boot isn't blocking kernel modules—solves 90% of setup hiccups.

## QuickStart
### 1. Create and manipulate tensors
```python
import pysml

a = pysml.Tensor([[1, 2], [3, 4]])
b = pysml.Tensor([[4, 3], [2, 1]])
print(pysml.add(a, b))
```

### 2. Run a gradient pass
```python
import pysml

x = pysml.Tensor([[1.0, 2.0]], requires_grad=True)
w = pysml.Tensor([[3.0], [4.0]], requires_grad=True)
logits = pysml.matmul(x, w)
loss = pysml.mean(logits)
loss.backward()
print('x.grad:', x.grad)
print('w.grad:', w.grad)
```

### 3. Train a tiny model
```python
import pysml
from pysml.nn import Linear, Module
from pysml.nn.optim import AdamW

class Classifier(Module):
    def __init__(self):
        super().__init__()
        self.fc = Linear(8, 2)

    def forward(self, inputs):
        return self.fc(inputs)

model = Classifier()
optimizer = AdamW(model.parameters(), lr=3e-4)
inputs = pysml.Tensor([[0.5] * 8])
targets = pysml.Tensor([[0.0, 1.0]])

for _ in range(10):
    logits = model(inputs)
    loss = pysml.mean((logits - targets) ** 2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 4. Scale out with the custom DDP runtime
```python
from pysml import ddp
from pysml.examples.parallel_utils import ExampleConfig
from pysml.examples.transformer import TinyTransformerClassifier

devices = ["xpu:0", "xpu:1"]  # Works with CUDA or CPU strings as well
ddp.register_global_communicator(devices)

PipelineModel = ddp.PipelineParallel(
    TinyTransformerClassifier,
    devices=devices,
    chunks=4,
)
model = PipelineModel(ExampleConfig())
```
Wrap any sequential module with `PipelineParallel` and optionally nest it inside
`ddp.DataParallel` for replica-style batching. See the dedicated DDP guide for
training/inference loops plus RWKV and Transformer launch scripts.

## Learn More
- Browse the new [`pysml/docs`](pysml/docs/README.md) directory for subsystem guides, API references, and integration tips.
- Read the [Custom Distributed Runtime guide](pysml/docs/ddp.md) for communicator setup, partition planning, and the `PipelineParallel` / `DataParallel` workflow.
- Explore `pysml/examples/` for ready-to-run Transformer, RWKV, and diffusion demos.
- Reuse the XPU-focused [Transformer](pysml/examples/ddp_transformer_xpu.py) and [RWKV](pysml/examples/ddp_rwkv_xpu.py) distributed rehearsal scripts as launch templates for your own models.
- Experiment with extending `pysml.engine` and `pysml.autograd` to register custom operations alongside built-ins.

> Personal note: I keep a scratchpad that registers experimental ops under `pysml.engine`—once the backward works, porting it into the main dispatcher takes only a few lines.

## Contributing
Issues and pull requests are welcome! Please describe the backend(s) you tested, include reproduction scripts when filing bugs, and update the docs if you introduce new public APIs.

Built with ❤️ by S.H.I.E.L.D.
Advancing AI Research Through Hardware-Agnostic Innovation
PySML v0.5.3
