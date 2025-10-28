"""
PySML Distributed Data Parallel (DDP)
"""

from .data_parallel import DataParallelModel, DistributedDataParallel
from .pipeline_parallel import PipelineTransformer, PipelineModule, create_pipeline_model
from .device_manager import DeviceManager, get_available_devices, print_device_info
from .strategies import (
    DistributedStrategy,
    StrategySelector,
    get_strategy_description,
    print_strategy_comparison,
    select_strategy,
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
    calculate_optimal_batch_split,
)

__all__ = [
    "DataParallelModel",
    "DistributedDataParallel",
    "PipelineTransformer",
    "PipelineModule",
    "create_pipeline_model",
    "DeviceManager",
    "get_available_devices",
    "print_device_info",
    "DistributedStrategy",
    "StrategySelector",
    "get_strategy_description",
    "print_strategy_comparison",
    "select_strategy",
    "split_batch",
    "merge_outputs",
    "merge_gradients",
    "average_gradients",
    "compute_gradient_norm",
    "clip_gradients",
    "synchronize_parameters",
    "count_parameters",
    "estimate_model_memory",
    "print_memory_estimate",
    "calculate_optimal_batch_split",
]
__version__ = "0.1.1"
