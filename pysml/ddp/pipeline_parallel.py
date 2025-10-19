"""
Pipeline Parallel Training

Split model layers across devices to train models larger than single-device memory.
"""

import pysml
import pysml.nn as nn
import pysml.nn.functional as F
from pysml.nn.models import TransformerBlock, LayerNorm
import numpy as np
from typing import List, Optional, Union


class PipelineModule:
    """
    Base class for pipeline-parallel modules
    
    Provides utilities for splitting a model across devices.
    """
    
    def __init__(self, devices: List[str]):
        """
        Initialize pipeline module
        
        Args:
            devices: List of devices for pipeline stages
        """
        self.devices = devices
        self.num_devices = len(devices)
    
    def distribute_layers(self, num_layers: int) -> List[tuple]:
        """
        Calculate layer distribution across devices
        
        Args:
            num_layers: Total number of layers
        
        Returns:
            List of (start_layer, end_layer, device) tuples
        """
        layers_per_device = num_layers // self.num_devices
        remainder = num_layers % self.num_devices
        
        distribution = []
        current_layer = 0
        
        for device_idx, device in enumerate(self.devices):
            # Devices with lower indices get the remainder
            num_layers_on_device = layers_per_device + (1 if device_idx < remainder else 0)
            end_layer = current_layer + num_layers_on_device
            
            distribution.append((current_layer, end_layer, device))
            current_layer = end_layer
        
        return distribution
    
    def print_distribution(self, layer_names: Optional[List[str]] = None):
        """Print how layers are distributed across devices"""
        print(f"\n{'='*80}")
        print("Pipeline Parallel Layer Distribution")
        print(f"{'='*80}")
        
        if layer_names:
            for layer_idx, layer_name in enumerate(layer_names):
                device = self.get_device_for_layer(layer_idx)
                print(f"  Layer {layer_idx:2d} ({layer_name:20s}) -> {device}")
        
        print(f"{'='*80}")
    
    def get_device_for_layer(self, layer_idx: int) -> str:
        """Get device assignment for a specific layer"""
        # Implement in subclass
        raise NotImplementedError


class PipelineTransformer(nn.Module):
    """
    Pipeline-parallel Transformer that splits layers across devices
    
    This is KEY to scaling model size beyond single-device memory.
    
    Purpose: Train models larger than single device can hold
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        max_seq_len: Maximum sequence length
        devices: List of devices for pipeline stages
        dropout: Dropout probability
    
    Example:
        >>> # Create a large model split across 2 XPUs
        >>> model = PipelineTransformer(
        ...     vocab_size=10000,
        ...     d_model=512,
        ...     num_layers=24,  # Split 24 layers across devices
        ...     num_heads=8,
        ...     d_ff=2048,
        ...     max_seq_len=1024,
        ...     devices=['xpu:0', 'xpu:1']
        ... )
    """
    
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, 
                 num_heads: int, d_ff: int, max_seq_len: int, 
                 devices: List[str], dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout_p = dropout
        self.devices = devices
        self.num_devices = len(devices)
        
        print(f"\n{'='*80}")
        print(f"Initializing Pipeline-Parallel TransformerLM")
        print(f"{'='*80}")
        print(f"  vocab_size={vocab_size}, d_model={d_model}")
        print(f"  num_layers={num_layers}, num_heads={num_heads}")
        print(f"  d_ff={d_ff}, max_seq_len={max_seq_len}")
        print(f"  Devices: {devices}")
        
        # Calculate layers per device for pipeline parallelism
        layers_per_device = num_layers // self.num_devices
        remainder = num_layers % self.num_devices
        
        print(f"\nPipeline Parallelism Strategy:")
        print(f"  Layers per device: ~{layers_per_device}")
        
        # Embeddings on first device
        self.token_embedding = pysml.randn(vocab_size, d_model, requires_grad=True) * 0.02
        self.pos_embedding = pysml.randn(max_seq_len, d_model, requires_grad=True) * 0.02
        self.embedding_device = devices[0]
        print(f"  Embeddings -> {self.embedding_device}")
        
        # Distribute transformer blocks across devices
        self.blocks = nn.ModuleList()
        self.block_devices = []
        
        current_device_idx = 0
        layers_on_current = 0
        layers_for_current = layers_per_device + (1 if current_device_idx < remainder else 0)
        
        for layer_idx in range(num_layers):
            block = TransformerBlock(d_model, num_heads, d_ff, dropout)
            self.blocks.append(block)
            
            device = devices[current_device_idx]
            self.block_devices.append(device)
            
            print(f"  Layer {layer_idx:2d} -> {device}")
            
            layers_on_current += 1
            if layers_on_current >= layers_for_current:
                current_device_idx += 1
                if current_device_idx < self.num_devices:
                    layers_on_current = 0
                    layers_for_current = layers_per_device + (1 if current_device_idx < remainder else 0)
        
        # Final layers on last device
        self.ln_f = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.output_device = devices[-1]
        print(f"  Output layers -> {self.output_device}")
        
        total_params = sum(p.size for p in self.parameters())
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Parameters per device: ~{total_params // self.num_devices:,}")
        print(f"{'='*80}")
    
    def forward(self, x):
        """
        Forward pass with pipeline parallelism
        
        Data flows sequentially through devices with actual device transfers.
        
        Args:
            x: Token indices of shape (batch_size, seq_len)
        
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        
        # Embeddings (device 0)
        # Move input to embedding device
        if x.device != self.embedding_device:
            x = pysml.to_device(x, self.embedding_device)
        
        token_emb = F.embedding(x, self.token_embedding)
        pos_emb = self.pos_embedding[:seq_len]
        pos_emb_expanded = pysml.reshape(pos_emb, (1, seq_len, self.d_model))
        x = token_emb + pos_emb_expanded
        
        if self.training and self.dropout_p > 0:
            x = pysml.dropout(x, p=self.dropout_p, training=True)
        
        # Pass through transformer blocks with device transfers
        current_device = self.embedding_device
        for block, target_device in zip(self.blocks, self.block_devices):
            # Transfer to target device if needed
            if current_device != target_device:
                x = pysml.to_device(x, target_device)
                current_device = target_device
            
            x = block(x)
        
        # Final layers (last device)
        if current_device != self.output_device:
            x = pysml.to_device(x, self.output_device)
        
        x = self.ln_f(x)
        
        x_flat = pysml.reshape(x, (batch_size * seq_len, self.d_model))
        logits = self.lm_head(x_flat)
        logits = pysml.reshape(logits, (batch_size, seq_len, self.vocab_size))
        
        return logits
    
    def get_memory_breakdown(self) -> dict:
        """
        Calculate memory usage per device
        
        Returns:
            Dictionary mapping device to parameter count
        """
        memory_map = {device: 0 for device in self.devices}
        
        # Embeddings
        memory_map[self.embedding_device] += self.token_embedding.size
        memory_map[self.embedding_device] += self.pos_embedding.size
        
        # Transformer blocks
        for block, device in zip(self.blocks, self.block_devices):
            block_params = sum(p.size for p in block.parameters())
            memory_map[device] += block_params
        
        # Output layers
        memory_map[self.output_device] += sum(p.size for p in self.ln_f.parameters())
        memory_map[self.output_device] += sum(p.size for p in self.lm_head.parameters())
        
        return memory_map
    
    def print_memory_breakdown(self):
        """Print memory usage per device"""
        memory_map = self.get_memory_breakdown()
        total = sum(memory_map.values())
        
        print(f"\n{'='*80}")
        print("Memory Distribution")
        print(f"{'='*80}")
        for device, params in memory_map.items():
            percentage = (params / total) * 100
            print(f"  {device}: {params:,} parameters ({percentage:.1f}%)")
        print(f"  Total: {total:,} parameters")
        print(f"{'='*80}")


def create_pipeline_model(model_type: str = 'transformer', 
                          preset: str = 'SMALL',
                          devices: Optional[List[str]] = None,
                          **kwargs) -> PipelineModule:
    """
    Factory function to create pipeline-parallel models
    
    Args:
        model_type: Type of model ('transformer', etc.)
        preset: Model preset name
        devices: List of devices (auto-detected if None)
        **kwargs: Additional model arguments
    
    Returns:
        Pipeline-parallel model
    
    Example:
        >>> model = create_pipeline_model(
        ...     model_type='transformer',
        ...     preset='LARGE',
        ...     devices=['xpu:0', 'xpu:1', 'xpu:2']
        ... )
    """
    from .device_manager import DeviceManager
    
    if devices is None:
        dm = DeviceManager()
        devices = dm.get_devices('auto', count=2)
    
    if model_type == 'transformer':
        # Get preset config
        from pysml.nn.models import TransformerConfig
        config = getattr(TransformerConfig, preset).copy()
        config.update(kwargs)
        
        return PipelineTransformer(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            max_seq_len=config['max_seq_len'],
            devices=devices,
            dropout=config.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")