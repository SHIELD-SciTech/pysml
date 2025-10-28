"""
DDP Utilities — optimized
- Zero-copy splitting (views)
- In-place grad ops on backend; no host round-trips
"""

from typing import List, Tuple, Union
import numpy as np
from ..tensor import Tensor

ArrayLike = Union[np.ndarray, Tensor]


def split_batch(data: ArrayLike, num_splits: int) -> List[ArrayLike]:
    """
    Split batch into num_splits contiguous views (no copies).
    """
    if isinstance(data, Tensor):
        arr = data.data
        bsz = arr.shape[0]
        step = max(1, bsz // num_splits)
        out: List[Tensor] = []
        for i in range(num_splits):
            s = i * step
            e = bsz if i == num_splits - 1 else (i + 1) * step
            if s >= e:
                continue
            # view/slice only
            out.append(Tensor(arr[s:e], backend=data.backend, device=data.device, requires_grad=data.requires_grad))
        return out
    else:
        bsz = data.shape[0]
        step = max(1, bsz // num_splits)
        splits = []
        for i in range(num_splits):
            s = i * step
            e = bsz if i == num_splits - 1 else (i + 1) * step
            if s >= e:
                continue
            splits.append(data[s:e])  # numpy view
        return splits


def merge_outputs(outputs: List[ArrayLike], axis: int = 0) -> ArrayLike:
    """
    Concatenate outputs along axis with backend-aware path.
    """
    if isinstance(outputs[0], Tensor):
        backend = outputs[0].backend
        device = outputs[0].device
        arrays = [o.data for o in outputs]
        if hasattr(backend, "concatenate"):
            merged = backend.concatenate(arrays, axis=axis)
        else:
            merged = np.concatenate(arrays, axis=axis)
        return Tensor(merged, backend=backend, device=device)
    return np.concatenate(outputs, axis=axis)


def average_gradients(parameters: List[Tensor], num_devices: int):
    """
    In-place AllReduce average (single process simulation).
    """
    if num_devices <= 1:
        return
    scale = 1.0 / float(num_devices)
    for p in parameters:
        if p.grad is not None:
            p.grad.data *= scale


def merge_gradients(gradients_list: List[List[Tensor]]) -> List[Tensor]:
    """
    Average a list of gradient lists (one per device). Uses in-place adds.
    """
    num_devices = len(gradients_list)
    if num_devices == 0:
        return []
    num_params = len(gradients_list[0])
    merged: List[Tensor] = [None] * num_params  # type: ignore
    for i in range(num_params):
        grads = [gl[i] for gl in gradients_list]
        g0 = grads[0]
        if g0 is None:
            merged[i] = None  # type: ignore
            continue
        acc = g0
        for g in grads[1:]:
            if g is not None:
                acc.grad.data += g.grad.data  # type: ignore
        acc.grad.data /= num_devices  # type: ignore
        merged[i] = acc
    return merged


def compute_gradient_norm(parameters: List[Tensor]) -> float:
    """
    L2 norm using backend ops (avoids host copies).
    """
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        b = p.backend
        g = p.grad.data
        # sum(g*g) -> float
        total += float(b.sum(g * g))
    return float(total) ** 0.5


def clip_gradients(parameters: List[Tensor], max_norm: float):
    """
    Clip by global norm; in-place scale on backend.
    """
    total = compute_gradient_norm(parameters)
    if total <= max_norm:
        return
    coef = max_norm / (total + 1e-6)
    for p in parameters:
        if p.grad is not None:
            p.grad.data *= coef


def synchronize_parameters(source_params: List[Tensor], target_params: List[Tensor]):
    """
    Copy parameters from source to target (in-place, device-local).
    """
    for src, tgt in zip(source_params, target_params):
        # keep device; copy value without reallocation if shapes match
        tgt.data[...] = src.data


def count_parameters(model) -> Tuple[int, int]:
    params = list(model.parameters())
    total = sum(p.size for p in params)
    trainable = sum(p.size for p in params if p.requires_grad)
    return total, trainable


def estimate_model_memory(model, dtype_bytes: int = 4) -> dict:
    total_params, _ = count_parameters(model)
    param_memory = total_params * dtype_bytes
    grad_memory = param_memory
    optimizer_memory = param_memory * 2  # e.g., Adam m/v
    gb = 1024 ** 3
    return {
        "parameters_gb": param_memory / gb,
        "gradients_gb": grad_memory / gb,
        "optimizer_gb": optimizer_memory / gb,
        "total_gb": (param_memory + grad_memory + optimizer_memory) / gb,
    }


def print_memory_estimate(model, dtype_bytes: int = 4):
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
    print(f"\nNote: excludes activation memory (depends on batch size)")

def calculate_optimal_batch_split(total_batch_size: int, num_devices: int, min_batch_per_device: int = 1) -> Tuple[int, int]:
    bpd = max(total_batch_size // max(1, num_devices), min_batch_per_device)
    effective = bpd * max(1, num_devices)
    if effective != total_batch_size:
        print(f"Warning: Adjusted batch size from {total_batch_size} to {effective}")
    return bpd, effective
