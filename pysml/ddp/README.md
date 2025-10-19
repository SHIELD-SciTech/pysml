# PySML Distributed Data Parallel (DDP)

Production-quality distributed training for PySML models across multiple XPU/CUDA devices.

## Overview

The `pysml.ddp` module provides:

- **Data Parallel**: Replicate models across devices, split batches for speed
- **Pipeline Parallel**: Split model layers across devices for larger models
- **Device Management**: Automatic device detection and allocation
- **Strategy Selection**: Automatic recommendation of optimal distributed strategy
- **Utilities**: Gradient synchronization, batch splitting, memory estimation

## Quick Start

```python
import pysml
import pysml.nn as nn
from pysml.ddp import DataParallelModel, PipelineTransformer, DeviceManager

# Detect available devices
dm = DeviceManager()
devices = dm.get_devices('xpu', count=2)

# Option 1: Data Parallel (for speed)
model = nn.Classifier.from_preset('DEEP')
dp_model = DataParallelModel(model, devices)

# Train with data parallel
loss = dp_model.forward_and_backward(X_batch, y_batch)
optimizer.step()

# Option 2: Pipeline Parallel (for model size)
large_model = PipelineTransformer(
    vocab_size=50000,
    d_model=768,
    num_layers=24,  # Split across devices
    num_heads=12,
    d_ff=3072,
    max_seq_len=1024,
    devices=devices
)
```

## Module Structure

```
pysml/ddp/
├── __init__.py              # Package initialization
├── data_parallel.py         # Data parallel implementation
├── pipeline_parallel.py     # Pipeline parallel implementation  
├── device_manager.py        # Device detection and management
├── strategies.py            # Strategy selection and comparison
├── utils.py                 # Utility functions
└── README.md               # This file
```

## Components

### DataParallelModel

Replicates model on each device and splits batches across devices.

```python
from pysml.ddp import DataParallelModel

model = nn.TransformerLM.from_preset('SMALL')
dp_model = DataParallelModel(model, devices=['xpu:0', 'xpu:1'])

# Training loop
for epoch in range(epochs):
    loss = dp_model.forward_and_backward(X, y)
    optimizer.step()
```

**Use when**: Model fits on single device, want to speed up training with large batches.

**Benefits**:
- Near-linear speedup with number of devices
- Simple API, minimal code changes
- Works with any model architecture

### PipelineTransformer

Splits transformer layers across devices to train models larger than single-device memory.

```python
from pysml.ddp import PipelineTransformer

# Create model split across 4 devices
model = PipelineTransformer(
    vocab_size=50000,
    d_model=1024,
    num_layers=48,  # 12 layers per device
    num_heads=16,
    d_ff=4096,
    max_seq_len=2048,
    devices=['xpu:0', 'xpu:1', 'xpu:2', 'xpu:3']
)

# Show memory distribution
model.print_memory_breakdown()
```

**Use when**: Model too large for single device memory.

**Benefits**:
- Train models 4x larger with 4 devices
- Each device holds ~1/N of parameters
- Enables very large model training

### DeviceManager

Central manager for device detection and allocation.

```python
from pysml.ddp import DeviceManager

dm = DeviceManager()

# Get device info
print(f"XPU devices: {dm.num_xpus}")
print(f"CUDA devices: {dm.num_cudas}")

# Allocate devices
data_parallel_devices = dm.allocate_for_data_parallel('xpu', num_replicas=4)
pipeline_devices = dm.allocate_for_pipeline_parallel('xpu', num_stages=8)

# Print summary
dm.print_summary()
```

### Strategy Selection

Automatically recommend optimal distributed strategy based on model and resources.

```python
from pysml.ddp import StrategySelector, print_strategy_comparison

# Compare all strategies
print_strategy_comparison()

# Analyze specific model
selector = StrategySelector(model, devices, device_memory_gb=16.0)
strategy = selector.recommend(batch_size=64)
print(f"Recommended: {strategy}")

# Detailed analysis
selector.print_analysis()
```

### Utilities

Helper functions for distributed training.

```python
from pysml.ddp.utils import (
    split_batch,
    average_gradients,
    compute_gradient_norm,
    clip_gradients,
    print_memory_estimate
)

# Split batch across devices
splits = split_batch(X, num_splits=4)

# Average gradients (simulates AllReduce)
average_gradients(model.parameters(), num_devices=4)

# Gradient clipping
clip_gradients(model.parameters(), max_norm=1.0)

# Memory estimation
print_memory_estimate(model, dtype_bytes=4)
```

## Distributed Strategies Comparison

### Data Parallel

**Description**: Replicate model on each device, split batch across devices

**When to use**:
- Model fits on single device
- Large batch sizes (>= num_devices * 32)
- Want to speed up training

**Pros**:
- Near-linear speedup
- Simple to implement
- Works with any model

**Cons**:
- Model must fit on each device
- Gradient sync overhead
- Doesn't help with model size

### Pipeline Parallel

**Description**: Split model layers across devices

**When to use**:
- Model too large for single device
- Have 2+ devices available
- Sequential layer structure

**Pros**:
- Train models larger than single device
- Each device holds ~1/N parameters
- Reduces memory per device

**Cons**:
- Pipeline bubbles (sequential execution)
- More complex implementation
- Requires careful layer distribution

### Hybrid Parallel

**Description**: Combine data parallel + pipeline parallel

**When to use**:
- Very large models (GPT-3 scale)
- Many devices available (8+)
- Production training

**Pros**:
- Maximum scalability
- Combines benefits of both strategies
- Used in production (GPT-3, PaLM)

**Cons**:
- Most complex to implement
- Requires many devices
- Difficult to tune

## Examples

### Example 1: Data Parallel Training

```python
from pysml.ddp import DataParallelModel, DeviceManager
import pysml.nn as nn

# Setup
dm = DeviceManager()
devices = dm.get_devices('xpu', count=2)

# Create model
model = nn.Classifier.from_preset('DEEP')
dp_model = DataParallelModel(model, devices)

# Train
optimizer = nn.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    loss = dp_model.forward_and_backward(X_train, y_train)
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
```

### Example 2: Pipeline Parallel Transformer

```python
from pysml.ddp import PipelineTransformer

# Create large model split across devices
model = PipelineTransformer(
    vocab_size=50000,
    d_model=768,
    num_layers=24,
    num_heads=12,
    d_ff=3072,
    max_seq_len=1024,
    devices=['xpu:0', 'xpu:1', 'xpu:2', 'xpu:3']
)

# Show distribution
model.print_memory_breakdown()

# Train
optimizer = nn.AdamW(model.parameters(), lr=0.0001)

for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
```

### Example 3: Automatic Strategy Selection

```python
from pysml.ddp import StrategySelector, DeviceManager

dm = DeviceManager()
devices = dm.get_devices('auto')

# Analyze model
selector = StrategySelector(model, devices)
strategy = selector.recommend(batch_size=64)

if strategy == DistributedStrategy.DATA_PARALLEL:
    dp_model = DataParallelModel(model, devices)
elif strategy == DistributedStrategy.PIPELINE_PARALLEL:
    # Convert to pipeline model
    pass
```

## Best Practices

### 1. Choose the Right Strategy

- **Small model, large batch** → Data Parallel
- **Large model** → Pipeline Parallel
- **Huge model, fast training** → Hybrid

### 2. Gradient Accumulation

```python
# Correct gradient accumulation
model.zero_grad()  # Once at start

for device_split in splits:
    loss = forward_backward(device_split)  # Accumulates

average_gradients(model.parameters(), num_devices)  # Average
optimizer.step()
```

### 3. Memory Management

```python
# Check memory before training
from pysml.ddp.utils import print_memory_estimate

print_memory_estimate(model, dtype_bytes=4)  # FP32
print_memory_estimate(model, dtype_bytes=2)  # FP16

# Use FP16 to reduce memory by 50%
```

### 4. Gradient Clipping

```python
from pysml.ddp.utils import clip_gradients

# After backward, before step
loss.backward()
clip_gradients(model.parameters(), max_norm=1.0)
optimizer.step()
```

## Performance Tips

1. **Batch Size**: Use batch_size >= num_devices * 32 for data parallel
2. **Device Balance**: Distribute layers evenly in pipeline parallel
3. **Gradient Accumulation**: For very large batches, accumulate over multiple steps
4. **Mixed Precision**: Use FP16 to reduce memory and increase speed
5. **Profiling**: Monitor device utilization to identify bottlenecks

## Production Deployment

For production use, consider these frameworks:

- **PyTorch DDP**: `torch.nn.parallel.DistributedDataParallel`
- **DeepSpeed**: Pipeline + ZeRO optimization
- **Megatron-LM**: Tensor + pipeline parallelism  
- **FSDP**: Fully Sharded Data Parallel

PySML's DDP module provides educational implementations that demonstrate the core concepts used in these production frameworks.

## Troubleshooting

### Common Issues

**Issue**: Gradient size mismatch
```
Solution: Ensure matmul backward properly reduces batch dimensions
```

**Issue**: Out of memory
```
Solution: 
1. Use pipeline parallel to split model
2. Reduce batch size
3. Use gradient accumulation
4. Enable FP16 training
```

**Issue**: Slow training with data parallel
```
Solution:
1. Increase batch size
2. Check device utilization
3. Reduce gradient sync overhead
```

## References

- [PyTorch Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [DeepSpeed](https://www.deepspeed.ai/)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Efficient Large-Scale Language Model Training](https://arxiv.org/abs/2104.04473)

## License

Part of PySML - see main LICENSE file.