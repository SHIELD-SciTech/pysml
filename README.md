# PySML — Python SHIELD Machine Learning Framework

PySML is S.H.I.E.L.D. AI’s research-grade deep learning framework for fast, transparent model development. It mirrors PyTorch ergonomics while giving explicit control over tensors, devices, and distributed execution across CPU, NVIDIA CUDA, and Intel XPU from a single codebase.

> Internal-first: PySML is maintained for S.H.I.E.L.D. AI projects. External use is allowed only with prior permission.

## Why PySML
- One stack, three backends: NumPy (CPU), CuPy (CUDA), and dpnp/dpctl (XPU) with runtime dispatch.
- Minimal, transparent autograd: each op records its backward closure; easy to add custom differentiable primitives.
- PyTorch-like modules: familiar `Module`/`Parameter`, layers, losses, optimizers, and LR schedulers.
- Distributed without torch.distributed: pure-Python data, pipeline, and hybrid parallel for heterogeneous clusters.
- RWKV-ready: reference RWKV implementations (including an 80M chatbot) for text-generation research.
- Checkpointing helpers: save/load tensors, state_dicts, and distributed strategy metadata.

## Installation
Requirements: Python 3.8+, NumPy 1.20+

CPU only:
```bash
pip install numpy
```

CUDA:
```bash
pip install cupy-cuda11x  # or cupy-cuda12x
```

Intel XPU:
```bash
pip install dpnp dpctl
```

From source:
```bash
git clone https://github.com/SHIELD-SciTech/pysml.git
cd pysml
pip install -e .
python -c "import pysml; print('PySML', pysml.__version__)"
```

## Quick Start
```python
import pysml
from pysml import Tensor, nn, optim
import pysml.dtype as dtype

# Tensors + autograd
x = Tensor([[1., 2.]], dtype=dtype.fp32(), requires_grad=True)
w = Tensor([[3.], [4.]], dtype=dtype.fp32(), requires_grad=True)
y = pysml.matmul(x, w)
loss = pysml.mean(y)
loss.backward()
print("x.grad:", x.grad)

# Tiny MLP
class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hid)
        self.fc2 = nn.Linear(d_hid, d_out)
    def forward(self, x):
        return self.fc2(pysml.relu(self.fc1(x)))

model = MLP(10, 64, 2)
opt = optim.AdamW(model.parameters(), lr=1e-3)
```

Move to GPU/XPU:
```python
x = Tensor([[1., 2.]], dtype=dtype.fp32(), device="cuda")
x = x.to("cuda:0")      # or "xpu:0"
x = x.cuda()             # convenience
```

## Core APIs
- **Tensor** (`pysml.Tensor`): device-aware storage with dtype helpers (`fp32`, `fp16`, `bf16`), cloning, in-place ops, `.to()/.cuda()/.xpu()`, hooks, and backward graph traversal.
- **Ops** (`pysml.engine`): matmul, add/subtract/multiply/divide, exp/log/sqrt/sin/cos, relu/gelu/silu/sigmoid/tanh, softmax/log_softmax, layer_norm/rms_norm/batch_norm/group_norm, dropout, embedding, reshape/transpose/concatenate/stack/split, reductions (`sum/mean/max/min`), and more.
- **Modules & layers** (`pysml.nn`): Linear/Bilinear, Conv1d/2d/Transpose, Embedding, MultiheadAttention/ScaledDotProductAttention, norms (LayerNorm, RMSNorm, BatchNorm1d/2d, GroupNorm), activations (ReLU, GELU, SiLU, Sigmoid, Tanh, Softmax, LogSoftmax, LeakyReLU, Mish), pooling (Max/Avg/Adaptive), Dropout/Dropout2d, containers (Sequential, ModuleList, ModuleDict), losses (CrossEntropy, MSE, L1, BCE/BCEWithLogits, NLL, KLDiv, SmoothL1).
- **Optim & schedulers** (`pysml.optim`): SGD, Adam, AdamW; schedulers StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, LinearLR, ConstantLR, OneCycleLR.
- **Distributed** (`pysml.ddp`): DDP (data parallel), PipelineParallel with micro-batching and schedules, HybridParallel (data + pipeline), communication backend setup (`init_process_group`, `get_rank`, `barrier`), launch helpers, and samplers.
- **Serialization** (`pysml.save_load`): save/load tensors or checkpoints; `save_checkpoint`, `load_state_dict`, optional parallel strategy metadata.

## Distributed Example
```python
from pysml.ddp import init_process_group, DDP, get_rank, barrier
from pysml import nn

init_process_group(rank=rank, world_size=world_size, device_type="cuda", device_id=rank)
model = nn.Linear(128, 64).cuda()
model = DDP(model)

# forward/backward as usual; grads sync in backward
...  # training loop
barrier()
```

## RWKV Examples
```python
from pysml.examples.rwkv import RWKV, create_rwkv_small
m = create_rwkv_small(vocab_size=50257, device="cuda")
print("Params:", m.num_parameters())
```
80M chatbot:
```bash
cd pysml/examples
python train_chatbot.py
```

## Project Layout
```
pysml/
  __init__.py        # Public API surface
  tensor.py          # Tensor class + grad helpers
  engine.py          # Ops with autograd wiring
  autograd.py        # Backward functions/Function graph
  dtype.py           # fp32/fp16/bf16 dtypes
  cpu/ cuda/ xpu/    # Backends
  nn/                # Modules, layers, activations, loss
  optim/             # Optimizers & LR schedulers
  ddp/               # Data/pipeline/hybrid parallel
  examples/          # RWKV + training scripts
  save_load.py       # Checkpoint utilities
```

## Status and Use
- Supported internally by S.H.I.E.L.D. AI ML infrastructure.
- External use requires prior permission; coordinate with S.H.I.E.L.D. AI before redistribution or integration.
- Expect API surface to evolve as research needs change.

## License
MIT License. Built by S.H.I.E.L.D. AI for internal research; external use is granted only with explicit approval.

Built with ❤️ by S.H.I.E.L.D.
