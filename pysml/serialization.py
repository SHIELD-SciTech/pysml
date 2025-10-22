"""
PySML Model Serialization
Save and load models, optimizers, and training checkpoints
"""

import pickle
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import pysml


def _serialize_array(arr):
    """
    Convert backend array to serializable NumPy array
    
    Args:
        arr: Array from any backend (NumPy, CuPy, DPNP)
    
    Returns:
        NumPy array
    """
    if arr is None:
        return None
    
    if hasattr(arr, 'asnumpy'):
        # CuPy or DPNP array
        return arr.asnumpy()
    elif hasattr(arr, 'get'):
        # CuPy array (alternative method)
        return arr.get()
    elif isinstance(arr, np.ndarray):
        # Already NumPy
        return arr
    else:
        # Convert to NumPy
        return np.array(arr)


def _deserialize_array(arr, backend_name='cpu'):
    """
    Convert NumPy array back to backend-specific array
    
    Args:
        arr: NumPy array
        backend_name: Target backend ('cpu', 'cuda', 'xpu')
    
    Returns:
        Backend-specific array
    """
    if arr is None:
        return None
    
    if backend_name == 'cpu':
        return arr
    elif backend_name == 'cuda':
        try:
            import cupy as cp
            return cp.array(arr)
        except ImportError:
            print("Warning: CuPy not available, using CPU array")
            return arr
    elif backend_name == 'xpu':
        try:
            import dpnp
            return dpnp.array(arr)
        except ImportError:
            print("Warning: DPNP not available, using CPU array")
            return arr
    else:
        return arr


def save_model(model, filepath: str):
    """
    Save model weights only
    
    Args:
        model: Model to save
        filepath: Path to save file (.pysml extension recommended)
    
    Example:
        >>> model = nn.Linear(10, 5)
        >>> save_model(model, "model_weights.pysml")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    state_dict = model.state_dict()
    
    # Convert all arrays to NumPy for serialization
    serialized_state = {}
    for key, value in state_dict.items():
        if isinstance(value, dict):
            # Nested state dict (for child modules)
            serialized_state[key] = _serialize_state_dict(value)
        else:
            # Parameter or tensor
            if hasattr(value, 'data'):
                serialized_state[key] = _serialize_array(value.data)
            else:
                serialized_state[key] = _serialize_array(value)
    
    with open(filepath, 'wb') as f:
        pickle.dump(serialized_state, f)
    
    print(f"Model saved to {filepath}")


def _serialize_state_dict(state_dict: Dict) -> Dict:
    """Recursively serialize a state dict"""
    serialized = {}
    for key, value in state_dict.items():
        if isinstance(value, dict):
            serialized[key] = _serialize_state_dict(value)
        else:
            if hasattr(value, 'data'):
                serialized[key] = _serialize_array(value.data)
            else:
                serialized[key] = _serialize_array(value)
    return serialized


def _deserialize_state_dict(state_dict: Dict, backend_name: str = 'cpu') -> Dict:
    """Recursively deserialize a state dict"""
    deserialized = {}
    for key, value in state_dict.items():
        if isinstance(value, dict):
            deserialized[key] = _deserialize_state_dict(value, backend_name)
        else:
            deserialized[key] = _deserialize_array(value, backend_name)
    return deserialized


def load_model(model, filepath: str, device: Optional[str] = None):
    """
    Load model weights
    
    Args:
        model: Model to load weights into
        filepath: Path to saved model file
        device: Optional device to load to ('cpu', 'cuda:0', 'xpu:0')
    
    Example:
        >>> model = nn.Linear(10, 5)
        >>> load_model(model, "model_weights.pysml")
        >>> # Or load to specific device
        >>> load_model(model, "model_weights.pysml", device='cuda:0')
    """
    filepath = Path(filepath)
    
    with open(filepath, 'rb') as f:
        state_dict = pickle.load(f)
    
    # Determine target backend
    if device is None:
        backend_name = 'cpu'
    else:
        backend_name = device.split(':')[0]
    
    # Deserialize arrays to target backend
    deserialized_state = _deserialize_state_dict(state_dict, backend_name)
    
    model.load_state_dict(deserialized_state)
    
    print(f"Model loaded from {filepath}")


def save_checkpoint(model, optimizer, filepath: str, epoch: Optional[int] = None,
                   loss: Optional[float] = None, **kwargs):
    """
    Save complete training checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        filepath: Path to save checkpoint
        epoch: Current epoch number
        loss: Current loss value
        **kwargs: Additional metadata (e.g., best_accuracy, learning_rate)
    
    Example:
        >>> save_checkpoint(
        ...     model, optimizer, "checkpoint.pth",
        ...     epoch=50, loss=0.234,
        ...     best_accuracy=0.95, learning_rate=0.001
        ... )
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': _serialize_state_dict(model.state_dict()),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    
    # Add any additional metadata
    checkpoint.update(kwargs)
    
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"Checkpoint saved to {filepath}")
    if epoch is not None:
        print(f"  Epoch: {epoch}")
    if loss is not None:
        print(f"  Loss: {loss:.6f}")


def load_checkpoint(model, optimizer, filepath: str, device: Optional[str] = None) -> Dict[str, Any]:
    """
    Load training checkpoint
    
    Args:
        model: Model to load into
        optimizer: Optimizer to load into (pass None to skip optimizer loading)
        filepath: Path to checkpoint
        device: Optional device to load to
    
    Returns:
        Dictionary containing checkpoint metadata (epoch, loss, etc.)
    
    Example:
        >>> info = load_checkpoint(model, optimizer, "checkpoint.pth")
        >>> start_epoch = info['epoch'] + 1
        >>> print(f"Resuming from epoch {start_epoch}")
    """
    filepath = Path(filepath)
    
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Determine target backend
    if device is None:
        backend_name = 'cpu'
    else:
        backend_name = device.split(':')[0]
    
    # Load model state
    model_state = _deserialize_state_dict(checkpoint['model_state_dict'], backend_name)
    model.load_state_dict(model_state)
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    if 'epoch' in checkpoint and checkpoint['epoch'] is not None:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint and checkpoint['loss'] is not None:
        print(f"  Loss: {checkpoint['loss']:.6f}")
    
    # Return metadata
    metadata = {k: v for k, v in checkpoint.items() 
                if k not in ['model_state_dict', 'optimizer_state_dict']}
    
    return metadata


def get_model_size(model) -> Dict[str, Any]:
    """
    Get model size information
    
    Args:
        model: Model to analyze
    
    Returns:
        Dictionary with parameter counts and memory estimates
    
    Example:
        >>> info = get_model_size(model)
        >>> print(f"Parameters: {info['total_params']:,}")
        >>> print(f"Memory: {info['memory_mb']:.2f} MB")
    """
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        param_count = param.size if hasattr(param, 'size') else np.prod(param.shape)
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
    
    # Estimate memory (4 bytes per float32 parameter)
    memory_bytes = total_params * 4
    memory_mb = memory_bytes / (1024 ** 2)
    memory_gb = memory_bytes / (1024 ** 3)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params,
        'memory_bytes': memory_bytes,
        'memory_mb': memory_mb,
        'memory_gb': memory_gb,
    }


def save_config(config: Dict[str, Any], filepath: str):
    """
    Save model configuration as JSON
    
    Args:
        config: Configuration dictionary
        filepath: Path to save config (.json extension recommended)
    
    Example:
        >>> config = {
        ...     'model_type': 'transformer',
        ...     'num_layers': 12,
        ...     'd_model': 768,
        ... }
        >>> save_config(config, "model_config.json")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Config saved to {filepath}")


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load model configuration from JSON
    
    Args:
        filepath: Path to config file
    
    Returns:
        Configuration dictionary
    
    Example:
        >>> config = load_config("model_config.json")
        >>> model = create_model(**config)
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    print(f"Config loaded from {filepath}")
    return config


def save_training_state(model, optimizer, scheduler, filepath: str,
                       epoch: int, loss: float, **kwargs):
    """
    Save complete training state including scheduler
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Learning rate scheduler to save
        filepath: Path to save state
        epoch: Current epoch
        loss: Current loss
        **kwargs: Additional metadata
    
    Example:
        >>> save_training_state(
        ...     model, optimizer, scheduler, "training_state.pth",
        ...     epoch=100, loss=0.123, best_val_acc=0.96
        ... )
    """
    checkpoint = {
        'model_state_dict': _serialize_state_dict(model.state_dict()),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
        'epoch': epoch,
        'loss': loss,
    }
    
    checkpoint.update(kwargs)
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"Training state saved to {filepath}")


def load_training_state(model, optimizer, scheduler, filepath: str,
                       device: Optional[str] = None) -> Dict[str, Any]:
    """
    Load complete training state including scheduler
    
    Args:
        model: Model to load into
        optimizer: Optimizer to load into
        scheduler: Scheduler to load into
        filepath: Path to saved state
        device: Optional device to load to
    
    Returns:
        Dictionary with metadata
    
    Example:
        >>> info = load_training_state(model, optimizer, scheduler, "training_state.pth")
        >>> start_epoch = info['epoch'] + 1
    """
    filepath = Path(filepath)
    
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Determine backend
    if device is None:
        backend_name = 'cpu'
    else:
        backend_name = device.split(':')[0]
    
    # Load states
    model_state = _deserialize_state_dict(checkpoint['model_state_dict'], backend_name)
    model.load_state_dict(model_state)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        if hasattr(scheduler, 'load_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Training state loaded from {filepath}")
    
    # Return metadata
    metadata = {k: v for k, v in checkpoint.items() 
                if k not in ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict']}
    
    return metadata


# Convenience aliases matching PyTorch naming
save_state_dict = save_model
load_state_dict = load_model