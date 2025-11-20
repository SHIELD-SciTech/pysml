import json
import pickle
from importlib import import_module, util
from typing import Any, Dict, Optional


def _load_parallel_strategy():
    """Load ``ParallelStrategy`` without raising on optional dependency absence."""

    spec = util.find_spec("pysml.ddp.config.partition_config")
    if spec is None:
        return None

    module = import_module("pysml.ddp.config.partition_config")
    return getattr(module, "ParallelStrategy", None)


ParallelStrategy = _load_parallel_strategy()


def save(obj, f, pickle_protocol=2):
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
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }

    if strategy is not None:
        checkpoint["parallel_strategy"] = _serialize_strategy(strategy)
    if distributed_state is not None:
        checkpoint["distributed_state"] = distributed_state

    checkpoint.update(kwargs)

    save(checkpoint, filepath)


def load_checkpoint(model, optimizer, filepath, map_location=None):
    checkpoint = load(filepath, map_location=map_location)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
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


torch_save = save
torch_load = load


def export_onnx(model, dummy_input, filepath, opset_version=12, **kwargs):
    raise NotImplementedError(
        "ONNX export is not yet implemented. "
        "This feature is planned for PySML v0.5.0. "
        "For now, please use standard save/load functions."
    )


def save_safetensors(model, filepath):
    raise NotImplementedError(
        "Safetensors format is not yet implemented. "
        "This feature is planned for PySML v0.5.0. "
        "For now, please use standard save/load functions."
    )


def load_safetensors(filepath):
    raise NotImplementedError(
        "Safetensors format is not yet implemented. "
        "This feature is planned for PySML v0.5.0. "
        "For now, please use standard load function."
    )


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
