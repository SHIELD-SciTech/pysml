"""
Distributed Training Strategies — optimized analysis
- O(1) memory math
- Clear, fast recommendation path
"""

from enum import Enum
from typing import List, Optional
from ..nn import Module


class DistributedStrategy(Enum):
    DATA_PARALLEL = "data_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    TENSOR_PARALLEL = "tensor_parallel"
    HYBRID = "hybrid"

    def __str__(self):
        return self.value


class StrategySelector:
    def __init__(self, model: Module, devices: List[str], device_memory_gb: float = 16.0):
        self.model = model
        self.devices = devices
        self.device_memory_gb = float(device_memory_gb)
        self.num_devices = len(devices)

        # Count parameters with a single pass
        self.model_params = sum(p.size for p in model.parameters())
        self.model_size_gb = (self.model_params * 4.0) / (1024.0**3)  # float32

    def fits_on_single_device(self) -> bool:
        # Reserve ≈50% headroom for activations/optimizer states
        return self.model_size_gb < (self.device_memory_gb * 0.5)

    def recommend(self, batch_size: int = 32, prefer_speed: bool = True) -> Optional["DistributedStrategy"]:
        if self.num_devices <= 0:
            return None

        if not self.fits_on_single_device():
            if self.num_devices < 2:
                raise RuntimeError(
                    f"Model size ({self.model_size_gb:.2f} GB) exceeds single-device budget "
                    f"({self.device_memory_gb:.2f} GB) and only one device is available."
                )
            return DistributedStrategy.PIPELINE_PARALLEL

        if self.num_devices == 1:
            return None

        # Model fits everywhere; prefer DP for speed with sufficiently large batch
        if prefer_speed and batch_size >= max(4, 4 * self.num_devices):
            return DistributedStrategy.DATA_PARALLEL

        return DistributedStrategy.DATA_PARALLEL

    def print_analysis(self):
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
    # unchanged: returns metadata for printing help
    descriptions = {
        DistributedStrategy.DATA_PARALLEL: {
            "name": "Data Parallel",
            "description": "Replicate model on each device; split batch across devices",
            "use_case": "Speed up training with large batches",
            "pros": ["Near-linear speedup", "Simple", "Works for any architecture"],
            "cons": ["Model must fit on each device", "Grad sync overhead", "Doesn't help with model size"],
            "best_for": "Models that fit on one device; large batch sizes",
        },
        DistributedStrategy.PIPELINE_PARALLEL: {
            "name": "Pipeline Parallel",
            "description": "Split model layers across devices",
            "use_case": "Train models larger than per-device memory",
            "pros": ["Trains very large models", "Per-device parameter subset", "Lower per-device memory"],
            "cons": ["Pipeline bubbles", "More complex", "Careful layer mapping required"],
            "best_for": "Very large models that don't fit on a single device",
        },
        DistributedStrategy.TENSOR_PARALLEL: {
            "name": "Tensor Parallel",
            "description": "Split individual layers/tensors across devices",
            "use_case": "Huge layers (e.g., embeddings)",
            "pros": ["Handles very large layers", "Composable with other strategies"],
            "cons": ["High comms overhead", "Complex implementation", "Fast interconnect required"],
            "best_for": "Models with layers too large for any device",
        },
        DistributedStrategy.HYBRID: {
            "name": "Hybrid Parallel",
            "description": "Combine data + pipeline (optionally tensor) parallel",
            "use_case": "Max scale: large models with speed",
            "pros": ["Combines benefits", "Highest scalability"],
            "cons": ["Most complex", "Many devices", "Hard to tune/debug"],
            "best_for": "Production training at very large scale",
        },
    }
    return descriptions.get(strategy, {})


def print_strategy_comparison():
    print(f"\n{'='*80}")
    print("Distributed Training Strategies Comparison")
    print(f"{'='*80}")
    for strategy in DistributedStrategy:
        desc = get_strategy_description(strategy)
        if desc:
            print(f"\n{desc['name'].upper()}")
            print(f"  Description: {desc['description']}")
            print(f"  Best for: {desc['best_for']}")
            print("  Pros:")
            for pro in desc["pros"]:
                print(f"    + {pro}")
            print("  Cons:")
            for con in desc["cons"]:
                print(f"    - {con}")
    print(f"\n{'='*80}")


def select_strategy(model: Module, devices: List[str], batch_size: int = 32, device_memory_gb: float = 16.0) -> Optional[DistributedStrategy]:
    return StrategySelector(model, devices, device_memory_gb).recommend(batch_size=batch_size)
