"""
PySML Distributed Data Parallel (DDP) Module

This package provides distributed training capabilities for PySML:
- Data Parallel: Replicate model across devices, split batches
- Pipeline Parallel: Split model layers across devices
- Hybrid Parallel: Combine both strategies
- Utilities for multi-device training

Usage:
    from pysml.ddp import DataParallelModel, PipelineTransformer
    from pysml.ddp import get_available_devices, DeviceManager
"""

from .data_parallel import DataParallelModel, DistributedDataParallel
from .pipeline_parallel import PipelineTransformer, PipelineModule, create_pipeline_model
from .device_manager import DeviceManager, get_available_devices, print_device_info
from .strategies import (
    DistributedStrategy, 
    StrategySelector, 
    get_strategy_description,
    print_strategy_comparison,
    select_strategy
)
from .utils import (
    split_batch, 
    merge_outputs,
    merge_gradients, 
    average_gradients,
    compute_gradient_norm,
    clip_gradients,
    synchronize_parameters,
    count_parameters,
    estimate_model_memory,
    print_memory_estimate,
    calculate_optimal_batch_split
)

__all__ = [
    # Data Parallel
    'DataParallelModel',
    'DistributedDataParallel',
    
    # Pipeline Parallel
    'PipelineTransformer',
    'PipelineModule',
    'create_pipeline_model',
    
    # Device Management
    'DeviceManager',
    'get_available_devices',
    'print_device_info',
    
    # Strategies
    'DistributedStrategy',
    'StrategySelector',
    'get_strategy_description',
    'print_strategy_comparison',
    'select_strategy',
    
    # Utilities
    'split_batch',
    'merge_outputs',
    'merge_gradients',
    'average_gradients',
    'compute_gradient_norm',
    'clip_gradients',
    'synchronize_parameters',
    'count_parameters',
    'estimate_model_memory',
    'print_memory_estimate',
    'calculate_optimal_batch_split',
]

__version__ = '0.1.0'