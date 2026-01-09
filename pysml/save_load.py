import json
import pickle
import warnings
from importlib import import_module, util
from typing import Any, Dict, Optional

import numpy as np

def _load_parallel_strategy():

    spec = util.find_spec("pysml.ddp.config.partition_config")
    if spec is None:
        return None
    module = import_module("pysml.ddp.config.partition_config")
    return getattr(module, "ParallelStrategy", None)

ParallelStrategy = _load_parallel_strategy()


def _to_cpu_numpy(obj):
    """Recursively convert tensors/arrays to numpy for serialization."""
    from .tensor import Tensor

    if isinstance(obj, Tensor):
        # Convert tensor data to numpy
        data = obj.data
        if hasattr(data, 'asnumpy'):
            # dpnp/dpctl array - convert to numpy
            return data.asnumpy()
        elif hasattr(obj._backend, 'asnumpy'):
            # Use backend's asnumpy if available
            return obj._backend.asnumpy(data)
        else:
            return np.asarray(data)
    elif isinstance(obj, dict):
        return {k: _to_cpu_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu_numpy(item) for item in obj)
    elif hasattr(obj, 'asnumpy'):
        # dpnp/dpctl array directly
        return obj.asnumpy()
    else:
        # Try numpy conversion as last resort for unknown array types
        try:
            if hasattr(obj, '__array__'):
                return np.asarray(obj)
        except Exception:
            pass
        return obj


def save(obj, f, pickle_protocol=2):
    # Convert any XPU/GPU tensors to numpy before pickling
    obj = _to_cpu_numpy(obj)
    if isinstance(f, str):
        with open(f, "wb") as file:
            pickle.dump(obj, file, protocol=pickle_protocol)
    else:
        pickle.dump(obj, f, protocol=pickle_protocol)

def load(f, map_location=None):
    if isinstance(f, str):
        with open(f, "rb") as file:
            obj = pickle.load(file)
    else:
        obj = pickle.load(f)

    if map_location is not None:
        obj = _map_location(obj, map_location)

    return obj

def save_state_dict(model, filepath):
    state_dict = model.state_dict()
    save(state_dict, filepath)

def load_state_dict(model, filepath, strict=True, map_location=None):
    state_dict = load(filepath, map_location=map_location)
    model.load_state_dict(state_dict, strict=strict)

def _serialize_strategy(strategy):
    if strategy is None:
        return None
    if ParallelStrategy is not None and isinstance(strategy, ParallelStrategy):
        return strategy.to_dict()
    if hasattr(strategy, "to_dict"):
        return strategy.to_dict()
    raise TypeError("strategy must be a ParallelStrategy or expose to_dict()")

def _deserialize_strategy(payload):
    if payload is None or ParallelStrategy is None:
        return payload
    return ParallelStrategy.from_dict(payload)

def save_checkpoint(
    model,
    optimizer,
    filepath,
    epoch=None,
    loss=None,
    *,
    strategy=None,
    distributed_state: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "loss": loss,
    }

    if strategy is not None:
        checkpoint["parallel_strategy"] = _serialize_strategy(strategy)
    if distributed_state is not None:
        checkpoint["distributed_state"] = distributed_state

    checkpoint.update(kwargs)

    save(checkpoint, filepath)

def load_checkpoint(model, optimizer, filepath, map_location=None, strict=False):
    checkpoint = load(filepath, map_location=map_location)

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    metadata = {
        k: v
        for k, v in checkpoint.items()
        if k not in ["model_state_dict", "optimizer_state_dict"]
    }
    if "parallel_strategy" in metadata:
        metadata["parallel_strategy"] = _deserialize_strategy(
            metadata["parallel_strategy"]
        )

    return metadata

def get_model_size(model):
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        param_count = 1
        for dim in param.shape:
            param_count *= dim

        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count

    memory_bytes = total_params * 4
    memory_mb = memory_bytes / (1024 * 1024)
    memory_gb = memory_mb / 1024

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": total_params - trainable_params,
        "memory_bytes": memory_bytes,
        "memory_mb": memory_mb,
        "memory_gb": memory_gb,
    }

def save_model_info(model, filepath):
    info = get_model_size(model)
    info["architecture"] = str(model)
    info["model_class"] = model.__class__.__name__

    with open(filepath, "w") as f:
        json.dump(info, f, indent=2, default=str)

def _map_location(obj, device):
    from .tensor import Tensor

    if isinstance(obj, Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _map_location(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_map_location(item, device) for item in obj)
    return obj

def _to_numpy(value):

    if hasattr(value, "numpy"):
        try:
            return value.numpy()
        except Exception:
            pass

    if hasattr(value, "data"):
        backend = getattr(value, "_backend", None)
        if backend is not None and hasattr(backend, "asnumpy"):
            try:
                return backend.asnumpy(value.data)
            except Exception:
                pass
        try:
            return np.asarray(value.data)
        except Exception:
            pass

    try:
        return np.asarray(value)
    except Exception:
        return None

torch_save = save
torch_load = load

def export_onnx(model, dummy_input, filepath, opset_version=12, **kwargs):
    raise NotImplementedError(
        "ONNX export is not yet implemented. "
        "This feature is planned for PySML v0.5.0. "
        "For now, please use standard save/load functions."
    )

def save_safetensors(model, filepath):
    warnings.warn(
        "Safetensors support is currently experimental/missing. "
        "Falling back to standard pickle checkpointing."
    )
    return save_state_dict(model, filepath)

def load_safetensors(filepath):
    warnings.warn(
        "Safetensors support is currently experimental/missing. "
        "Falling back to standard pickle loading."
    )
    return load(filepath)

__all__ = [
    "save",
    "load",
    "save_state_dict",
    "load_state_dict",
    "save_checkpoint",
    "load_checkpoint",
    "get_model_size",
    "save_model_info",
    "torch_save",
    "torch_load",
    "export_onnx",
    "save_safetensors",
    "load_safetensors",
]
