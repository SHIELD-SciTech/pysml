import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
import warnings

warnings.filterwarnings("ignore")


class SerializationError(Exception):
    # Exception raised for serialization/deserialization errors.
    pass


def save_state_dict(model, filepath: Union[str, Path], compress: bool = True):
    filepath = Path(filepath)
    
    # Get all parameters from the model
    state_dict = {}
    params = model.parameters()
    
    # Create parameter mapping with paths
    param_map = _build_parameter_map(model)
    
    for name, param in param_map.items():
        # Convert to CPU and numpy for serialization
        if hasattr(param.data, 'get'):  # CuPy array
            data = param.data.get()
        elif hasattr(param.data, 'asnumpy'):  # DPNP array
            data = param.data.asnumpy()
        else:  # NumPy array
            data = param.data
        
        state_dict[name] = {
            'data': data,
            'shape': param.shape,
            'dtype': str(param.dtype),
            'requires_grad': param.requires_grad
        }
    
    # Save to file
    try:
        with open(filepath, 'wb') as f:
            if compress:
                import gzip
                with gzip.open(filepath, 'wb') as gz_f:
                    pickle.dump(state_dict, gz_f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                pickle.dump(state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"[PySML] State dict saved to {filepath} ({len(state_dict)} parameters)")
    except Exception as e:
        raise SerializationError(f"Failed to save state dict: {e}")


def load_state_dict(model, filepath: Union[str, Path], strict: bool = True):
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"State dict file not found: {filepath}")
    
    # Load state dict
    try:
        try:
            import gzip
            with gzip.open(filepath, 'rb') as gz_f:
                state_dict = pickle.load(gz_f)
        except (gzip.BadGzipFile, OSError):
            # Fall back to uncompressed
            with open(filepath, 'rb') as f:
                state_dict = pickle.load(f)
    except Exception as e:
        raise SerializationError(f"Failed to load state dict: {e}")
    
    # Get model's parameter map
    param_map = _build_parameter_map(model)
    
    # Check for missing/unexpected keys
    model_keys = set(param_map.keys())
    loaded_keys = set(state_dict.keys())
    
    missing_keys = model_keys - loaded_keys
    unexpected_keys = loaded_keys - model_keys
    
    if strict and (missing_keys or unexpected_keys):
        error_msg = []
        if missing_keys:
            error_msg.append(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            error_msg.append(f"Unexpected keys: {unexpected_keys}")
        raise SerializationError("\n".join(error_msg))
    elif not strict:
        if missing_keys:
            warnings.warn(f"Missing keys in state dict: {missing_keys}")
        if unexpected_keys:
            warnings.warn(f"Unexpected keys in state dict: {unexpected_keys}")
    
    # Load parameters
    from pysml.tensor import Tensor
    loaded_count = 0
    
    for name, param in param_map.items():
        if name in state_dict:
            saved_param = state_dict[name]
            
            # Create tensor from saved data
            new_data = saved_param['data']
            
            # Verify shapes match
            if tuple(saved_param['shape']) != tuple(param.shape):
                if strict:
                    raise SerializationError(
                        f"Shape mismatch for {name}: "
                        f"expected {param.shape}, got {saved_param['shape']}"
                    )
                else:
                    warnings.warn(f"Skipping {name} due to shape mismatch")
                    continue
            
            # Update parameter data (keep it on the same device)
            if hasattr(param.data, '__cuda_array_interface__'):
                # CUDA tensor
                import cupy as cp
                param.data = cp.array(new_data)
            elif hasattr(param.data, '__sycl_usm_array_interface__'):
                # XPU tensor
                import dpnp
                param.data = dpnp.array(new_data, sycl_queue=param.data.sycl_queue)
            else:
                # CPU tensor
                param.data = np.array(new_data)
            
            loaded_count += 1
    
    print(f"[PySML] Loaded {loaded_count}/{len(state_dict)} parameters from {filepath}")


def save_checkpoint(
    model,
    optimizer,
    filepath: Union[str, Path],
    epoch: Optional[int] = None,
    loss: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    compress: bool = True
):
    filepath = Path(filepath)
    
    checkpoint = {
        'model_state_dict': _extract_state_dict(model),
        'optimizer_state_dict': _extract_optimizer_state(optimizer),
        'epoch': epoch,
        'loss': loss,
        'metadata': metadata or {}
    }
    
    try:
        if compress:
            import gzip
            with gzip.open(filepath, 'wb') as gz_f:
                pickle.dump(checkpoint, gz_f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"[PySML] Checkpoint saved to {filepath}")
        if epoch is not None:
            print(f"  Epoch: {epoch}, Loss: {loss:.4f}" if loss else f"  Epoch: {epoch}")
    except Exception as e:
        raise SerializationError(f"Failed to save checkpoint: {e}")


def load_checkpoint(
    model,
    optimizer,
    filepath: Union[str, Path],
    strict: bool = True
) -> Dict[str, Any]:
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    try:
        # Try compressed format first
        try:
            import gzip
            with gzip.open(filepath, 'rb') as gz_f:
                checkpoint = pickle.load(gz_f)
        except (gzip.BadGzipFile, OSError):
            with open(filepath, 'rb') as f:
                checkpoint = pickle.load(f)
    except Exception as e:
        raise SerializationError(f"Failed to load checkpoint: {e}")
    
    # Load model state
    _load_state_dict_from_data(model, checkpoint['model_state_dict'], strict)
    
    # Load optimizer state
    _load_optimizer_state(optimizer, checkpoint['optimizer_state_dict'])
    
    print(f"[PySML] Checkpoint loaded from {filepath}")
    if checkpoint.get('epoch') is not None:
        epoch_str = f"  Epoch: {checkpoint['epoch']}"
        if checkpoint.get('loss') is not None:
            epoch_str += f", Loss: {checkpoint['loss']:.4f}"
        print(epoch_str)
    
    return {
        'epoch': checkpoint.get('epoch'),
        'loss': checkpoint.get('loss'),
        'metadata': checkpoint.get('metadata', {})
    }


def save_model(
    model,
    filepath: Union[str, Path],
    compress: bool = True,
    include_metadata: bool = True
):
    filepath = Path(filepath)
    
    model_data = {
        'model': model,
        'class_name': model.__class__.__name__,
        'module_name': model.__class__.__module__,
    }
    
    if include_metadata:
        model_data['metadata'] = {
            'parameters': len(list(model.parameters())),
            'training': model.training
        }
    
    try:
        if compress:
            import gzip
            with gzip.open(filepath, 'wb') as gz_f:
                pickle.dump(model_data, gz_f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"[PySML] Full model saved to {filepath}")
    except Exception as e:
        raise SerializationError(f"Failed to save model: {e}")


def load_model(filepath: Union[str, Path]):
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    warnings.warn(
        "Loading pickled models can execute arbitrary code. "
        "Only load models from trusted sources.",
        UserWarning
    )
    
    try:
        # Try compressed format first
        try:
            import gzip
            with gzip.open(filepath, 'rb') as gz_f:
                model_data = pickle.load(gz_f)
        except (gzip.BadGzipFile, OSError):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
    except Exception as e:
        raise SerializationError(f"Failed to load model: {e}")
    
    model = model_data['model']
    print(f"[PySML] Full model loaded from {filepath}")
    
    if 'metadata' in model_data:
        metadata = model_data['metadata']
        print(f"  Parameters: {metadata.get('parameters', 'Unknown')}")
    
    return model


# Helper funcs ;)

def _build_parameter_map(model, prefix: str = '') -> Dict[str, Any]:
    from pysml.nn.module import Module
    from pysml.tensor import Tensor
    
    param_map = {}
    
    for attr_name in dir(model):
        if attr_name.startswith('_'):
            continue
        
        try:
            attr = getattr(model, attr_name)
        except AttributeError:
            continue
        
        # Full path name for this attribute
        full_name = f"{prefix}.{attr_name}" if prefix else attr_name
        
        if isinstance(attr, Tensor) and attr.requires_grad:
            param_map[full_name] = attr
        elif isinstance(attr, Module):
            # Recursively get parameters from submodules
            sub_params = _build_parameter_map(attr, full_name)
            param_map.update(sub_params)
        elif isinstance(attr, (list, tuple)):
            # Handle lists of modules (like in TransformerEncoder)
            for i, item in enumerate(attr):
                if isinstance(item, Module):
                    sub_params = _build_parameter_map(item, f"{full_name}.{i}")
                    param_map.update(sub_params)
    
    return param_map


def _extract_state_dict(model) -> Dict[str, Any]:
    param_map = _build_parameter_map(model)
    state_dict = {}
    
    for name, param in param_map.items():
        if hasattr(param.data, 'get'):
            data = param.data.get()
        elif hasattr(param.data, 'asnumpy'):
            data = param.data.asnumpy()
        else:
            data = param.data
        
        state_dict[name] = {
            'data': data,
            'shape': param.shape,
            'dtype': str(param.dtype),
            'requires_grad': param.requires_grad
        }
    
    return state_dict


def _load_state_dict_from_data(model, state_dict: Dict[str, Any], strict: bool = True):
    param_map = _build_parameter_map(model)
    
    for name, param in param_map.items():
        if name in state_dict:
            saved_param = state_dict[name]
            
            if tuple(saved_param['shape']) != tuple(param.shape):
                if strict:
                    raise SerializationError(f"Shape mismatch for {name}")
                continue
            
            new_data = saved_param['data']
            
            if hasattr(param.data, '__cuda_array_interface__'):
                import cupy as cp
                param.data = cp.array(new_data)
            elif hasattr(param.data, '__sycl_usm_array_interface__'):
                import dpnp
                param.data = dpnp.array(new_data, sycl_queue=param.data.sycl_queue)
            else:
                param.data = np.array(new_data)


def _extract_optimizer_state(optimizer) -> Dict[str, Any]:
    opt_state = {
        'class_name': optimizer.__class__.__name__,
        'lr': optimizer.lr,
    }
    
    # Save optimizer-specific state
    if hasattr(optimizer, 't'):  # Adam/AdamW
        opt_state['t'] = optimizer.t
    
    if hasattr(optimizer, 'm'):  # Adam/AdamW
        opt_state['m'] = [m.copy() if hasattr(m, 'copy') else m for m in optimizer.m]
        opt_state['v'] = [v.copy() if hasattr(v, 'copy') else v for v in optimizer.v]
    
    if hasattr(optimizer, 'velocity'):  # SGD w. momentum
        if optimizer.velocity:
            opt_state['velocity'] = [v.copy() if hasattr(v, 'copy') else v for v in optimizer.velocity]
    
    return opt_state


def _load_optimizer_state(optimizer, opt_state: Dict[str, Any]):
    """Load optimizer state."""
    if opt_state['class_name'] != optimizer.__class__.__name__:
        warnings.warn(
            f"Optimizer class mismatch: saved {opt_state['class_name']}, "
            f"current {optimizer.__class__.__name__}"
        )
    
    optimizer.lr = opt_state['lr']
    
    if 't' in opt_state:
        optimizer.t = opt_state['t']
    
    if 'm' in opt_state:
        optimizer.m = opt_state['m']
        optimizer.v = opt_state['v']
    
    if 'velocity' in opt_state:
        optimizer.velocity = opt_state['velocity']


def get_model_size(model) -> Dict[str, Any]:
    params = model.parameters()
    
    total_params = 0
    trainable_params = 0
    total_bytes = 0
    
    for param in params:
        num_params = np.prod(param.shape)
        total_params += num_params
        
        if param.requires_grad:
            trainable_params += num_params
        
        # Calculate memory usage
        dtype_size = param.data.itemsize
        total_bytes += num_params * dtype_size
    
    return {
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'non_trainable_params': int(total_params - trainable_params),
        'memory_mb': total_bytes / (1024 * 1024),
        'memory_gb': total_bytes / (1024 * 1024 * 1024)
    }

