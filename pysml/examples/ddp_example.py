"""
PySML Distributed Data Parallel (DDP) Example
Using the pysml.ddp module for clean, production-style distributed training
"""

import sys, os
sys.path.append(os.getcwd())
import pysml
import pysml.nn as nn
import numpy as np
import time

# Import DDP components
from pysml.ddp import (
    DataParallelModel,
    PipelineTransformer,
    DeviceManager,
    print_device_info,
    DistributedStrategy,
    print_strategy_comparison,
    print_memory_estimate,
)


print("=" * 80)
print("PySML Multi-Device Distributed Training Example")
print("Using pysml.ddp module")
print("=" * 80)


# ===== Setup: Device Detection =====
print("\n" + "=" * 80)
print("Device Detection")
print("=" * 80)

dm = DeviceManager()
xpu_devices, cuda_devices = print_device_info()


# ===== Strategy Overview =====
print("\n\n")
print_strategy_comparison()


# ===== Example 1: Data Parallel Training =====
def example_data_parallel():
    """
    Data Parallel: Replicate model, split batch
    Purpose: Speed up training with large batches
    """
    
    print("\n" + "=" * 80)
    print("Example 1: Data Parallel Training")
    print("Purpose: Speed up training (NOT for larger models)")
    print("=" * 80)
    
    # Get devices
    devices = dm.get_devices('auto', count=2)
    
    # Create standard model
    print("\nCreating standard classifier...")
    model = nn.Classifier.from_preset('SIMPLE_MLP')
    print(f"Model parameters: {sum(p.size for p in model.parameters()):,}")
    
    # Wrap in data parallel
    dp_model = DataParallelModel(model, devices)
    
    # Generate training data
    print("\nGenerating training data...")
    np.random.seed(42)
    n_samples = 256
    X_train = np.random.randn(n_samples, 784).astype(np.float32) * 0.5
    y_train_labels = np.random.randint(0, 10, n_samples)
    y_train = np.zeros((n_samples, 10), dtype=np.float32)
    for i in range(n_samples):
        y_train[i, y_train_labels[i]] = 1.0
    
    print(f"Total batch size: {n_samples}")
    print(f"Per-device batch size: {n_samples // len(devices)}")
    
    # Setup optimizer
    optimizer = nn.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("\nTraining with Data Parallel...")
    epochs = 10
    start_time = time.time()
    
    for epoch in range(epochs):
        # Data parallel forward and backward
        loss = dp_model.forward_and_backward(X_train, y_train)
        
        # Optimizer step
        optimizer.step()
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}, Loss: {loss:.6f}")
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f} seconds")
    print(f"Average time per epoch: {elapsed/epochs:.3f} seconds")


# ===== Example 2: Pipeline Parallel Training =====
def example_pipeline_parallel():
    """
    Pipeline Parallel: Split model layers across devices
    Purpose: Train models LARGER than single-device memory
    """
    
    print("\n" + "=" * 80)
    print("Example 2: Pipeline Parallel Training")
    print("Purpose: Scale model size beyond single device memory")
    print("=" * 80)
    
    # Get devices for pipeline
    devices = dm.get_devices('auto', count=2)
    
    # Create LARGE pipeline-parallel transformer
    print("\nCreating LARGE pipeline-parallel transformer...")
    vocab_size = 1000
    model = PipelineTransformer(
        vocab_size=vocab_size,
        d_model=256,
        num_layers=8,
        num_heads=8,
        d_ff=1024,
        max_seq_len=512,
        devices=devices,
        dropout=0.1
    )
    
    # Show memory breakdown
    model.print_memory_breakdown()
    
    # Generate training data
    print("\nGenerating training data...")
    np.random.seed(123)
    batch_size = 32
    seq_len = 64
    
    X_train = np.random.randint(0, vocab_size, (batch_size, seq_len))
    y_train_labels = np.random.randint(0, vocab_size, (batch_size, seq_len))
    y_train = np.zeros((batch_size, seq_len, vocab_size), dtype=np.float32)
    for i in range(batch_size):
        for j in range(seq_len):
            y_train[i, j, y_train_labels[i, j]] = 1.0
    
    print(f"Training data: {X_train.shape}")
    
    # Setup optimizer
    optimizer = nn.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Training loop
    print("\nTraining with Pipeline Parallelism...")
    epochs = 5
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        X_tensor = pysml.Tensor(X_train, requires_grad=False)
        y_tensor = pysml.Tensor(y_train, requires_grad=False)
        
        # Forward pass (data flows through pipeline)
        output = model(X_tensor)
        
        # Compute loss
        loss = pysml.mse_loss(output, y_tensor)
        
        # Backward pass (gradients flow backward through pipeline)
        loss.backward()
        
        # Update all parameters across all devices
        optimizer.step()
        
        print(f"Epoch {epoch+1:3d}/{epochs}, Loss: {loss.item():.6f}")
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f} seconds")
    print(f"Average time per epoch: {elapsed/epochs:.3f} seconds")


# ===== Example 3: Strategy Selection =====
def example_strategy_selection():
    """
    Demonstrate automatic strategy selection
    """
    
    print("\n" + "=" * 80)
    print("Example 3: Automatic Strategy Selection")
    print("=" * 80)
    
    from pysml.ddp.strategies import StrategySelector
    
    # Create different sized models
    models = {
        'Small': nn.Classifier.from_preset('SIMPLE_MLP'),
        'Medium': nn.Classifier.from_preset('DEEP'),
        'Large': nn.Classifier.from_preset('WIDE'),
    }
    
    devices = dm.get_devices('auto', count=min(dm.num_xpus or 1, 4))
    
    print(f"\nAnalyzing models with {len(devices)} devices available:\n")
    
    for name, model in models.items():
        print(f"\n{name} Model:")
        print(f"  Parameters: {sum(p.size for p in model.parameters()):,}")
        
        selector = StrategySelector(model, devices, device_memory_gb=16.0)
        strategy = selector.recommend(batch_size=64)
        
        if strategy:
            print(f"  Recommended strategy: {strategy}")
        else:
            print(f"  Recommended strategy: Single device (no distribution needed)")


# ===== Example 4: Memory Analysis =====
def example_memory_analysis():
    """
    Analyze memory requirements for different models
    """
    
    print("\n" + "=" * 80)
    print("Example 4: Memory Analysis")
    print("=" * 80)
    
    # Create a transformer model
    print("\nAnalyzing Transformer model:")
    vocab_size = 10000
    model = nn.TransformerLM.from_preset('SMALL', vocab_size=vocab_size)
    
    print_memory_estimate(model, dtype_bytes=4)
    
    print("\nWith FP16 (half precision):")
    print_memory_estimate(model, dtype_bytes=2)


# ===== Performance Comparison =====
def compare_single_vs_distributed():
    """Compare single device vs distributed approaches"""
    
    print("\n" + "=" * 80)
    print("Example 5: Performance Comparison")
    print("=" * 80)
    
    devices = dm.get_devices('auto', count=2)
    
    # Setup
    model_single = nn.Classifier.from_preset('SIMPLE_MLP')
    model_dp = nn.Classifier.from_preset('SIMPLE_MLP')
    
    # Data
    np.random.seed(789)
    n_samples = 256
    X = np.random.randn(n_samples, 784).astype(np.float32)
    y_labels = np.random.randint(0, 10, n_samples)
    y = np.zeros((n_samples, 10), dtype=np.float32)
    for i in range(n_samples):
        y[i, y_labels[i]] = 1.0
    
    # Single device
    print("\n1. Single Device Training:")
    optimizer_single = nn.Adam(model_single.parameters(), lr=0.001)
    
    start = time.time()
    for epoch in range(10):
        optimizer_single.zero_grad()
        output = model_single(pysml.Tensor(X, requires_grad=False))
        loss = pysml.mse_loss(output, pysml.Tensor(y, requires_grad=False))
        loss.backward()
        optimizer_single.step()
    single_time = time.time() - start
    print(f"   Time: {single_time:.3f} seconds")
    
    # Data parallel
    print("\n2. Data Parallel Training:")
    dp_wrapper = DataParallelModel(model_dp, devices)
    optimizer_dp = nn.Adam(model_dp.parameters(), lr=0.001)
    
    start = time.time()
    for epoch in range(10):
        loss = dp_wrapper.forward_and_backward(X, y)
        optimizer_dp.step()
    dp_time = time.time() - start
    print(f"   Time: {dp_time:.3f} seconds")
    
    # Results
    print("\n" + "=" * 80)
    print("Results:")
    print(f"  Single Device: {single_time:.3f}s")
    print(f"  Data Parallel: {dp_time:.3f}s")
    speedup = single_time / dp_time
    print(f"  Speedup:       {speedup:.2f}x")
    print("=" * 80)
    
    print("\nNote: In simulation, speedup may be <1x due to overhead.")
    print("In production with real multi-GPU/XPU:")
    print("  • Data parallel: Near-linear speedup")
    print("  • Pipeline parallel: Enables models beyond single-device memory")


# ===== Main Execution =====
if __name__ == "__main__":
    print("\n\n")
    
    # Run all examples
    try:
        example_data_parallel()
    except Exception as e:
        print(f"\nError in Example 1: {e}")
    
    print("\n\n")
    
    try:
        example_pipeline_parallel()
    except Exception as e:
        print(f"\nError in Example 2: {e}")
    
    print("\n\n")
    
    try:
        example_strategy_selection()
    except Exception as e:
        print(f"\nError in Example 3: {e}")
    
    print("\n\n")
    
    try:
        example_memory_analysis()
    except Exception as e:
        print(f"\nError in Example 4: {e}")
    
    print("\n\n")
    
    try:
        compare_single_vs_distributed()
    except Exception as e:
        print(f"\nError in Example 5: {e}")
    
    # Final Summary
    print("\n" + "=" * 80)
    print("Summary - Distributed Training with pysml.ddp")
    print("=" * 80)
    
    print("\n✓ Module Structure:")
    print("  • pysml.ddp.DataParallelModel - Data parallel training")
    print("  • pysml.ddp.PipelineTransformer - Pipeline parallel transformers")
    print("  • pysml.ddp.DeviceManager - Device detection and allocation")
    print("  • pysml.ddp.DistributedStrategy - Strategy enumeration")
    print("  • pysml.ddp.utils - Utility functions")
    
    print("\n✓ Key Concepts:")
    print("  • Data Parallel: Replicate model, split batches → Speed")
    print("  • Pipeline Parallel: Split layers across devices → Model Size")
    print("  • Hybrid: Combine both → Maximum Scale")
    
    print("\n✓ When to Use What:")
    print("  • Model fits on 1 device + large batches → Data Parallel")
    print("  • Model too large for 1 device → Pipeline Parallel")
    print("  • Huge model + fast training → Hybrid Parallel")
    
    print("\n✓ Production Frameworks:")
    print("  • PyTorch DDP: torch.nn.parallel.DistributedDataParallel")
    print("  • DeepSpeed: Pipeline + ZeRO optimization")
    print("  • Megatron-LM: Tensor + Pipeline parallelism")
    print("  • FSDP: Fully Sharded Data Parallel")
    
    print("\n" + "=" * 80)
    print("Examples completed successfully!")
    print("=" * 80)

