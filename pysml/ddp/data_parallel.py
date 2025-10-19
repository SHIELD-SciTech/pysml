"""
Data Parallel Training

Replicate model across devices and split batches for faster training.
"""

import pysml
import numpy as np
from typing import List, Optional
from ..nn import Module


class DataParallelModel:
    """
    Data Parallel wrapper for distributed training
    
    Replicates the model on each device and splits the batch across devices.
    Gradients are synchronized via AllReduce (simulated by averaging).
    
    Purpose: Speed up training with large batches (NOT for larger models)
    
    Args:
        model: Model to distribute
        devices: List of device strings
        
    Example:
        >>> model = nn.TransformerLM.from_preset('SMALL')
        >>> dp_model = DataParallelModel(model, ['xpu:0', 'xpu:1'])
        >>> loss = dp_model.forward_and_backward(X, y)
        >>> optimizer.step()
    """
    
    def __init__(self, model: Module, devices: List[str]):
        """
        Initialize data parallel model
        
        Args:
            model: Base model to replicate (should already be on primary device)
            devices: List of devices for replicas
        """
        self.base_model = model
        self.devices = devices
        self.num_devices = len(devices)
        self.primary_device = devices[0]
        
        print(f"\n{'='*80}")
        print(f"Data Parallel Configuration")
        print(f"{'='*80}")
        print(f"Devices: {devices}")
        print(f"Primary device: {self.primary_device}")
        print(f"Model will process data splits on primary device")
        print(f"Batch will be split: batch_size // {self.num_devices}")
        print(f"Gradients will be synchronized via AllReduce")
        print(f"{'='*80}")
    
    def forward_and_backward(self, X, y):
        """
        Forward pass with data parallelism and correct gradient accumulation
        
        This is the CORRECT way to do data parallel training:
        1. Zero gradients ONCE before all device splits
        2. Forward + backward on each device's data split (accumulates gradients)
        3. Average gradients after all splits processed (simulates AllReduce)
        
        Args:
            X: Input data (numpy array or tensor)
            y: Target data (numpy array or tensor)
        
        Returns:
            Average loss across all devices
        """
        batch_size = X.shape[0]
        split_size = batch_size // self.num_devices
        
        # CRITICAL: Zero gradients ONCE at the start
        self.base_model.zero_grad()
        
        losses = []
        
        # Process on primary device only (in true DDP, would be on all devices in parallel)
        # For simplicity, we run sequentially on primary device with different data splits
        for i, device in enumerate([self.primary_device] * self.num_devices):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < self.num_devices - 1 else batch_size
            
            # Split data for this device
            X_split = X[start_idx:end_idx]
            y_split = y[start_idx:end_idx]
            
            # Create tensors and move to device
            X_split_tensor = pysml.Tensor(X_split, requires_grad=False)
            y_split_tensor = pysml.Tensor(y_split, requires_grad=False)
            
            # Move to device
            X_split_tensor = pysml.to_device(X_split_tensor, device)
            y_split_tensor = pysml.to_device(y_split_tensor, device)
            
            # Forward pass on this device's data split
            output = self.base_model(X_split_tensor)
            
            # Handle shape mismatches (e.g., for transformers)
            if output.shape != y_split_tensor.shape:
                if output.size == y_split_tensor.size:
                    y_split_data = y_split_tensor.data
                    if hasattr(y_split_data, 'reshape'):
                        y_split_tensor = pysml.Tensor(
                            y_split_data.reshape(output.shape),
                            requires_grad=False
                        )
                    else:
                        y_split_tensor = pysml.Tensor(
                            np.array(y_split_data).reshape(output.shape),
                            requires_grad=False
                        )
                    # Move reshaped tensor to device
                    y_split_tensor = pysml.to_device(y_split_tensor, device)
            
            # Compute loss
            loss = pysml.mse_loss(output, y_split_tensor)
            
            # CRITICAL: Backward pass accumulates gradients
            # Don't call zero_grad() here!
            loss.backward()
            
            losses.append(loss.item())
        
        # Average gradients (simulates AllReduce in production DDP)
        for param in self.base_model.parameters():
            if param.grad is not None:
                param.grad.data = param.grad.data / self.num_devices
        
        # Return average loss
        return sum(losses) / len(losses)
    
    def __call__(self, X, y):
        """Allow calling as function"""
        return self.forward_and_backward(X, y)


class DistributedDataParallel:
    """
    Production-style DDP wrapper (more PyTorch-like API)
    
    This is an alternative interface that's closer to PyTorch's DDP.
    
    Example:
        >>> model = nn.TransformerLM.from_preset('SMALL')
        >>> ddp_model = DistributedDataParallel(model, device_ids=['xpu:0', 'xpu:1'])
        >>> 
        >>> for epoch in range(epochs):
        >>>     for X, y in dataloader:
        >>>         optimizer.zero_grad()
        >>>         output = ddp_model(X)
        >>>         loss = criterion(output, y)
        >>>         loss.backward()
        >>>         optimizer.step()
    """
    
    def __init__(self, model: Module, device_ids: List[str], 
                 output_device: Optional[str] = None):
        """
        Initialize DDP wrapper
        
        Args:
            model: Model to wrap
            device_ids: List of device IDs
            output_device: Device for output (defaults to first device)
        """
        self.module = model
        self.device_ids = device_ids
        self.output_device = output_device or device_ids[0]
        self.num_devices = len(device_ids)
        
        # In production, this would register hooks for gradient synchronization
        print(f"Initialized DDP with {self.num_devices} devices: {device_ids}")
    
    def forward(self, *inputs, **kwargs):
        """
        Forward pass (in production, this would handle device placement)
        
        For now, just delegates to the underlying model.
        In production DDP:
        - Input is replicated to all devices
        - Forward happens on all devices in parallel
        - Outputs are gathered to output_device
        """
        return self.module(*inputs, **kwargs)
    
    def __call__(self, *inputs, **kwargs):
        """Make callable"""
        return self.forward(*inputs, **kwargs)
    
    def parameters(self):
        """Get model parameters"""
        return self.module.parameters()
    
    def train(self, mode: bool = True):
        """Set training mode"""
        self.module.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode"""
        self.module.eval()
        return self
    
    def zero_grad(self):
        """Zero gradients"""
        self.module.zero_grad()