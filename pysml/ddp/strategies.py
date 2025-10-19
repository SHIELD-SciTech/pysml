"""
Distributed Training Strategies

High-level strategies for different distributed training scenarios.
"""

from enum import Enum
from typing import List, Optional
from ..nn import Module


class DistributedStrategy(Enum):
    """
    Enumeration of distributed training strategies
    
    - DATA_PARALLEL: Replicate model, split batches (for speed)
    - PIPELINE_PARALLEL: Split model layers (for model size)
    - TENSOR_PARALLEL: Split individual layers/tensors (for very large layers)
    - HYBRID: Combination of strategies
    """
    DATA_PARALLEL = "data_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    TENSOR_PARALLEL = "tensor_parallel"
    HYBRID = "hybrid"
    
    def __str__(self):
        return self.value


class StrategySelector:
    """
    Helper to select appropriate distributed strategy
    
    Analyzes model size, available devices, and training goals to
    recommend the best distributed training strategy.
    
    Example:
        >>> selector = StrategySelector(model, devices=['xpu:0', 'xpu:1'])
        >>> strategy = selector.recommend()
        >>> print(f"Recommended: {strategy}")
    """
    
    def __init__(self, model: Module, devices: List[str],
                 device_memory_gb: float = 16.0):
        """
        Initialize strategy selector
        
        Args:
            model: Model to distribute
            devices: Available devices
            device_memory_gb: Memory per device in GB
        """
        self.model = model
        self.devices = devices
        self.device_memory_gb = device_memory_gb
        self.num_devices = len(devices)
        
        # Calculate model size
        self.model_params = sum(p.size for p in model.parameters())
        # Rough estimate: 4 bytes per float32 parameter
        self.model_size_gb = (self.model_params * 4) / (1024**3)
    
    def fits_on_single_device(self) -> bool:
        """Check if model fits on a single device"""
        # Reserve 50% of memory for activations, optimizer states, etc.
        available_memory = self.device_memory_gb * 0.5
        return self.model_size_gb < available_memory
    
    def recommend(self, batch_size: int = 32, 
                  prefer_speed: bool = True) -> DistributedStrategy:
        """
        Recommend best distributed strategy
        
        Args:
            batch_size: Training batch size
            prefer_speed: Prefer speed over memory efficiency
        
        Returns:
            Recommended strategy
        """
        single_device_fit = self.fits_on_single_device()
        
        # If model doesn't fit on single device, must use pipeline/tensor parallel
        if not single_device_fit:
            if self.num_devices < 2:
                raise RuntimeError(
                    f"Model size ({self.model_size_gb:.2f} GB) exceeds single device "
                    f"memory ({self.device_memory_gb:.2f} GB) but only 1 device available"
                )
            return DistributedStrategy.PIPELINE_PARALLEL
        
        # Model fits on single device
        if self.num_devices == 1:
            # Only one device, no distribution needed
            return None
        
        # Multiple devices available, model fits on each
        if prefer_speed and batch_size >= self.num_devices * 4:
            # Large batch, data parallel will help
            return DistributedStrategy.DATA_PARALLEL
        
        # Default to data parallel for multi-device
        return DistributedStrategy.DATA_PARALLEL
    
    def print_analysis(self):
        """Print analysis of model and recommendations"""
        print(f"\n{'='*80}")
        print("Distributed Strategy Analysis")
        print(f"{'='*80}")
        print(f"Model parameters: {self.model_params:,}")
        print(f"Estimated model size: {self.model_size_gb:.2f} GB")
        print(f"Available devices: {self.num_devices}")
        print(f"Memory per device: {self.device_memory_gb:.2f} GB")
        print(f"Fits on single device: {self.fits_on_single_device()}")
        print(f"\nRecommended strategy: {self.recommend()}")
        print(f"{'='*80}")


def get_strategy_description(strategy: DistributedStrategy) -> dict:
    """
    Get detailed description of a distributed strategy
    
    Args:
        strategy: Strategy to describe
    
    Returns:
        Dictionary with description, use cases, pros, and cons
    """
    descriptions = {
        DistributedStrategy.DATA_PARALLEL: {
            "name": "Data Parallel",
            "description": "Replicate model on each device, split batch across devices",
            "use_case": "Speed up training with large batches",
            "pros": [
                "Near-linear speedup with number of devices",
                "Simple to implement",
                "Works with any model architecture"
            ],
            "cons": [
                "Model must fit on each device",
                "Communication overhead for gradient sync",
                "Doesn't help with model size"
            ],
            "best_for": "Models that fit on single device, large batch sizes"
        },
        DistributedStrategy.PIPELINE_PARALLEL: {
            "name": "Pipeline Parallel",
            "description": "Split model layers across devices",
            "use_case": "Train models larger than single device memory",
            "pros": [
                "Enables training of very large models",
                "Each device only holds subset of parameters",
                "Reduces per-device memory requirements"
            ],
            "cons": [
                "Sequential execution (pipeline bubbles)",
                "More complex to implement",
                "Requires careful layer distribution"
            ],
            "best_for": "Very large models that don't fit on single device"
        },
        DistributedStrategy.TENSOR_PARALLEL: {
            "name": "Tensor Parallel",
            "description": "Split individual layers/tensors across devices",
            "use_case": "Very large individual layers (e.g., huge embeddings)",
            "pros": [
                "Can handle very large individual layers",
                "Good for models with huge parameter tensors",
                "Can combine with other strategies"
            ],
            "cons": [
                "High communication overhead",
                "Complex implementation",
                "Requires fast interconnect between devices"
            ],
            "best_for": "Models with individual layers too large for one device"
        },
        DistributedStrategy.HYBRID: {
            "name": "Hybrid Parallel",
            "description": "Combine data parallel + pipeline parallel",
            "use_case": "Maximum scale - large models with fast training",
            "pros": [
                "Combines benefits of multiple strategies",
                "Highest scalability",
                "Best for production large-scale training"
            ],
            "cons": [
                "Most complex to implement",
                "Requires many devices",
                "Difficult to tune and debug"
            ],
            "best_for": "Production training of very large models (GPT-3, PaLM scale)"
        }
    }
    
    return descriptions.get(strategy, {})


def print_strategy_comparison():
    """Print comparison table of all strategies"""
    print(f"\n{'='*80}")
    print("Distributed Training Strategies Comparison")
    print(f"{'='*80}")
    
    for strategy in DistributedStrategy:
        desc = get_strategy_description(strategy)
        if desc:
            print(f"\n{desc['name'].upper()}")
            print(f"  Description: {desc['description']}")
            print(f"  Best for: {desc['best_for']}")
            print(f"  Pros:")
            for pro in desc['pros']:
                print(f"    + {pro}")
            print(f"  Cons:")
            for con in desc['cons']:
                print(f"    - {con}")
    
    print(f"\n{'='*80}")


def select_strategy(model: Module, devices: List[str],
                   batch_size: int = 32,
                   device_memory_gb: float = 16.0) -> DistributedStrategy:
    """
    Convenience function to select distributed strategy
    
    Args:
        model: Model to distribute
        devices: Available devices
        batch_size: Training batch size
        device_memory_gb: Memory per device in GB
    
    Returns:
        Recommended strategy
    
    Example:
        >>> strategy = select_strategy(model, ['xpu:0', 'xpu:1'], batch_size=64)
        >>> print(f"Use {strategy} for this model")
    """
    selector = StrategySelector(model, devices, device_memory_gb)
    return selector.recommend(batch_size=batch_size)