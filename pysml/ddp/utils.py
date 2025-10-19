"""
Utilities for Distributed Training

Helper functions for gradient synchronization, batch splitting, etc.
"""

import numpy as np
from typing import List, Tuple, Union
from ..tensor import Tensor


def split_batch(data: Union[np.ndarray, Tensor], 
                num_splits: int) -> List[Union[np.ndarray, Tensor]]:
    """
    Split batch across multiple devices
    
    Args:
        data: Input data to split
        num_splits: Number of splits (typically number of devices)
    
    Returns:
        List of data splits
    
    Example:
        >>> X = np.random.randn(128, 784)
        >>> splits = split_batch(X, num_splits=4)
        >>> len(splits)
        4
        >>> splits[0].shape
        (32, 784)
    """
    if isinstance(data, Tensor):
        data_array = data.data
        is_tensor = True
        backend = data.backend
        device = data.device
    else:
        data_array = data
        is_tensor = False
    
    batch_size = data_array.shape[0]
    split_size = batch_size // num_splits
    
    splits = []
    for i in range(num_splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < num_splits - 1 else batch_size
        
        split = data_array[start_idx:end_idx]
        
        if is_tensor:
            split = Tensor(split, backend=backend, device=device)
        
        splits.append(split)
    
    return splits


def merge_outputs(outputs: List[Union[np.ndarray, Tensor]], 
                  axis: int = 0) -> Union[np.ndarray, Tensor]:
    """
    Merge outputs from multiple devices
    
    Args:
        outputs: List of outputs to merge
        axis: Axis along which to concatenate
    
    Returns:
        Merged output
    
    Example:
        >>> outputs = [np.random.randn(32, 10) for _ in range(4)]
        >>> merged = merge_outputs(outputs)
        >>> merged.shape
        (128, 10)
    """
    if isinstance(outputs[0], Tensor):
        arrays = [o.data for o in outputs]
        backend = outputs[0].backend
        device = outputs[0].device
        
        if hasattr(backend, 'concatenate'):
            merged = backend.concatenate(arrays, axis=axis)
        else:
            merged = np.concatenate(arrays, axis=axis)
        
        return Tensor(merged, backend=backend, device=device)
    else:
        return np.concatenate(outputs, axis=axis)


def average_gradients(parameters: List[Tensor], num_devices: int):
    """
    Average gradients across devices (simulates AllReduce)
    
    This is what happens in production DDP:
    - Each device computes gradients on its data split
    - Gradients are synchronized across devices via AllReduce
    - Each device gets the averaged gradient
    
    Args:
        parameters: List of parameters with gradients
        num_devices: Number of devices to average over
    
    Example:
        >>> # After backward pass on all device splits
        >>> average_gradients(model.parameters(), num_devices=4)
    """
    for param in parameters:
        if param.grad is not None:
            param.grad.data = param.grad.data / num_devices


def merge_gradients(gradients_list: List[List[Tensor]]) -> List[Tensor]:
    """
    Merge gradients from multiple devices
    
    Takes gradients from different devices and averages them.
    
    Args:
        gradients_list: List of gradient lists, one per device
    
    Returns:
        Averaged gradients
    
    Example:
        >>> # Collect gradients from each device
        >>> device1_grads = [p.grad for p in model.parameters()]
        >>> device2_grads = [p.grad for p in model.parameters()]
        >>> merged = merge_gradients([device1_grads, device2_grads])
    """
    num_devices = len(gradients_list)
    num_params = len(gradients_list[0])
    
    merged = []
    
    for param_idx in range(num_params):
        # Collect this parameter's gradients from all devices
        param_grads = [gradients_list[dev_idx][param_idx] 
                      for dev_idx in range(num_devices)]
        
        # Average them
        if param_grads[0] is not None:
            avg_grad = param_grads[0]
            for grad in param_grads[1:]:
                avg_grad.data = avg_grad.data + grad.data
            avg_grad.data = avg_grad.data / num_devices
            merged.append(avg_grad)
        else:
            merged.append(None)
    
    return merged


def compute_gradient_norm(parameters: List[Tensor]) -> float:
    """
    Compute the L2 norm of gradients
    
    Useful for gradient clipping and monitoring training.
    
    Args:
        parameters: List of parameters
    
    Returns:
        Gradient norm
    
    Example:
        >>> norm = compute_gradient_norm(model.parameters())
        >>> print(f"Gradient norm: {norm:.4f}")
    """
    total_norm = 0.0
    
    for param in parameters:
        if param.grad is not None:
            grad_data = param.grad.data
            if hasattr(grad_data, 'asnumpy'):
                grad_data = grad_data.asnumpy()
            
            param_norm = np.linalg.norm(grad_data)
            total_norm += param_norm ** 2
    
    return np.sqrt(total_norm)


def clip_gradients(parameters: List[Tensor], max_norm: float):
    """
    Clip gradients by global norm
    
    Prevents exploding gradients in training.
    
    Args:
        parameters: List of parameters
        max_norm: Maximum allowed gradient norm
    
    Example:
        >>> # After backward pass
        >>> clip_gradients(model.parameters(), max_norm=1.0)
        >>> optimizer.step()
    """
    total_norm = compute_gradient_norm(parameters)
    
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        
        for param in parameters:
            if param.grad is not None:
                param.grad.data = param.grad.data * clip_coef


def synchronize_parameters(source_params: List[Tensor], 
                          target_params: List[Tensor]):
    """
    Copy parameters from source to target
    
    Used for model replication in data parallel training.
    
    Args:
        source_params: Source parameters to copy from
        target_params: Target parameters to copy to
    
    Example:
        >>> # Replicate model parameters to another device
        >>> synchronize_parameters(model1.parameters(), model2.parameters())
    """
    for src, tgt in zip(source_params, target_params):
        tgt.data = src.data.copy()


def count_parameters(model) -> Tuple[int, int]:
    """
    Count total and trainable parameters
    
    Args:
        model: Model to analyze
    
    Returns:
        (total_params, trainable_params)
    
    Example:
        >>> total, trainable = count_parameters(model)
        >>> print(f"Total: {total:,}, Trainable: {trainable:,}")
    """
    params = model.parameters()
    
    total = sum(p.size for p in params)
    trainable = sum(p.size for p in params if p.requires_grad)
    
    return total, trainable


def estimate_model_memory(model, dtype_bytes: int = 4) -> dict:
    """
    Estimate memory requirements for model
    
    Args:
        model: Model to analyze
        dtype_bytes: Bytes per parameter (4 for float32, 2 for float16)
    
    Returns:
        Dictionary with memory estimates in GB
    
    Example:
        >>> mem = estimate_model_memory(model)
        >>> print(f"Parameters: {mem['parameters_gb']:.2f} GB")
        >>> print(f"Gradients: {mem['gradients_gb']:.2f} GB")
        >>> print(f"Total: {mem['total_gb']:.2f} GB")
    """
    total_params, _ = count_parameters(model)
    
    # Parameter memory
    param_memory = total_params * dtype_bytes
    
    # Gradient memory (same as parameters)
    grad_memory = param_memory
    
    # Optimizer state (e.g., Adam uses 2x params for momentum and velocity)
    optimizer_memory = param_memory * 2
    
    # Convert to GB
    gb = 1024 ** 3
    
    return {
        'parameters_gb': param_memory / gb,
        'gradients_gb': grad_memory / gb,
        'optimizer_gb': optimizer_memory / gb,
        'total_gb': (param_memory + grad_memory + optimizer_memory) / gb
    }


def print_memory_estimate(model, dtype_bytes: int = 4):
    """
    Print detailed memory estimate
    
    Args:
        model: Model to analyze
        dtype_bytes: Bytes per parameter
    
    Example:
        >>> print_memory_estimate(model)
        ================================================================================
        Memory Estimate
        ================================================================================
        Parameters:       2.15 GB
        Gradients:        2.15 GB
        Optimizer State:  4.30 GB
        ----------------------------------------
        Total:            8.60 GB
        ================================================================================
    """
    mem = estimate_model_memory(model, dtype_bytes)
    total_params, trainable_params = count_parameters(model)
    
    print(f"\n{'='*80}")
    print("Memory Estimate")
    print(f"{'='*80}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"\nMemory breakdown (assuming {dtype_bytes}-byte precision):")
    print(f"  Parameters:       {mem['parameters_gb']:.2f} GB")
    print(f"  Gradients:        {mem['gradients_gb']:.2f} GB")
    print(f"  Optimizer State:  {mem['optimizer_gb']:.2f} GB")
    print(f"  {'-'*40}")
    print(f"  Total:            {mem['total_gb']:.2f} GB")
    print(f"{'='*80}")
    print(f"\nNote: This excludes activation memory, which depends on batch size")


def calculate_optimal_batch_split(total_batch_size: int, 
                                  num_devices: int,
                                  min_batch_per_device: int = 1) -> Tuple[int, int]:
    """
    Calculate optimal batch split across devices
    
    Args:
        total_batch_size: Desired total batch size
        num_devices: Number of devices
        min_batch_per_device: Minimum batch size per device
    
    Returns:
        (batch_per_device, effective_total_batch)
    
    Example:
        >>> batch_per_dev, total = calculate_optimal_batch_split(100, 8)
        >>> print(f"Each device: {batch_per_dev}, Total: {total}")
        Each device: 12, Total: 96
    """
    batch_per_device = max(total_batch_size // num_devices, min_batch_per_device)
    effective_total = batch_per_device * num_devices
    
    if effective_total != total_batch_size:
        print(f"Warning: Adjusted batch size from {total_batch_size} to {effective_total}")
    
    return batch_per_device, effective_total