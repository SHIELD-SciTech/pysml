import pickle
import json
import os
from collections import OrderedDict
import numpy as np


def save(obj, f, pickle_protocol=2):
	if isinstance(f, str):
		with open(f, 'wb') as file:
			pickle.dump(obj, file, protocol=pickle_protocol)
	else:
		pickle.dump(obj, f, protocol=pickle_protocol)


def load(f, map_location=None):
	if isinstance(f, str):
		with open(f, 'rb') as file:
			obj = pickle.load(file)
	else:
		obj = pickle.load(f)
	
	# Map to device if specified
	if map_location is not None:
		obj = _map_location(obj, map_location)
	
	return obj


def save_state_dict(model, filepath):
	state_dict = model.state_dict()
	save(state_dict, filepath)


def load_state_dict(model, filepath, strict=True, map_location=None):
	state_dict = load(filepath, map_location=map_location)
	model.load_state_dict(state_dict, strict=strict)


def save_checkpoint(model, optimizer, filepath, epoch=None, loss=None, **kwargs):
	checkpoint = {
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'epoch': epoch,
		'loss': loss,
	}
	
	# Add any additional metadata
	checkpoint.update(kwargs)
	
	save(checkpoint, filepath)


def load_checkpoint(model, optimizer, filepath, map_location=None):
	checkpoint = load(filepath, map_location=map_location)
	
	model.load_state_dict(checkpoint['model_state_dict'])
	
	if optimizer is not None and 'optimizer_state_dict' in checkpoint:
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	
	# Return metadata
	metadata = {k: v for k, v in checkpoint.items() 
				if k not in ['model_state_dict', 'optimizer_state_dict']}
	
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
	
	# Estimate memory (assuming float32 = 4 bytes)
	memory_bytes = total_params * 4
	memory_mb = memory_bytes / (1024 * 1024)
	memory_gb = memory_mb / 1024
	
	return {
		'total_params': total_params,
		'trainable_params': trainable_params,
		'non_trainable_params': total_params - trainable_params,
		'memory_bytes': memory_bytes,
		'memory_mb': memory_mb,
		'memory_gb': memory_gb
	}


def save_model_info(model, filepath):
	info = get_model_size(model)
	info['architecture'] = str(model)
	info['model_class'] = model.__class__.__name__
	
	with open(filepath, 'w') as f:
		json.dump(info, f, indent=2, default=str)


def _map_location(obj, device):
	from ..tensor import Tensor
	
	if isinstance(obj, Tensor):
		return obj.to(device)
	elif isinstance(obj, dict):
		return {k: _map_location(v, device) for k, v in obj.items()}
	elif isinstance(obj, (list, tuple)):
		return type(obj)(_map_location(item, device) for item in obj)
	else:
		return obj


# PyTorch-compatible aliases
torch_save = save
torch_load = load


# ONNX Export (placeholder for future implementation)
def export_onnx(model, dummy_input, filepath, opset_version=12, **kwargs):
	raise NotImplementedError(
		"ONNX export is not yet implemented. "
		"This feature is planned for PySML v0.5.0. "
		"For now, please use standard save/load functions."
	)


# Safetensors support (placeholder for future implementation)
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
	'save',
	'load',
	'save_state_dict',
	'load_state_dict',
	'save_checkpoint',
	'load_checkpoint',
	'get_model_size',
	'save_model_info',
	'torch_save',
	'torch_load',
	'export_onnx',
	'save_safetensors',
	'load_safetensors',
]