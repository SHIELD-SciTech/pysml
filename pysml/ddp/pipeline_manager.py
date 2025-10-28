"""
PySML Distributed Pipeline Parallelism Manager
v0.4.8
-----------------------------------------------
Supports multi-device sequential model execution.
Compatible with PySML DDP and AMP (automatic mixed precision).
"""

import math
import time
from typing import List, Any, Dict

from pysml.engine import (
    set_device,
    get_backend,
    get_device,
    amp_enabled,
    autocast,
    get_scaler,
)

# =============================================================================
# Stage
# =============================================================================
class Stage:
    """
    Represents a model stage assigned to a specific device.

    Example:
        >>> from pysml.ddp.pipeline_manager import Stage
        >>> s = Stage(model, "cuda:0")
        >>> out = s.forward(x)
    """

    def __init__(self, model, device: str):
        self.model = model
        self.device = device
        set_device(device)
        self.backend = get_backend()
        self.model.to(device)

    def forward(self, x):
        set_device(self.device)
        if amp_enabled():
            with autocast("float16"):
                return self.model(x)
        return self.model(x)

    def backward(self, grad_output):
        set_device(self.device)
        grad_output.backward()

    def parameters(self):
        return list(self.model.parameters())

    def __repr__(self):
        return f"<Stage {self.device}, {self.model.__class__.__name__}>"

# =============================================================================
# Pipeline
# =============================================================================
class Pipeline:
    """
    Multi-device pipeline execution engine.

    Example:
        >>> from pysml.ddp.pipeline_manager import Pipeline, Stage
        >>> stages = [
        ...     Stage(model_part1, "cuda:0"),
        ...     Stage(model_part2, "cuda:1"),
        ... ]
        >>> pipe = Pipeline(stages)
        >>> y = pipe.forward(x)
    """

    def __init__(self, stages: List[Stage], microbatch_size: int = 1):
        self.stages = stages
        self.num_stages = len(stages)
        self.microbatch_size = microbatch_size
        self.scaler = get_scaler() if amp_enabled() else None

    # -------------------------------------------------------------------------
    # Forward pass with microbatch splitting
    # -------------------------------------------------------------------------
    def forward(self, inputs):
        """
        Split input into microbatches and run sequentially across stages.
        """
        if not isinstance(inputs, list):
            inputs = [inputs]

        microbatches = self._split_microbatches(inputs, self.microbatch_size)
        outputs = []

        for mb in microbatches:
            out = mb
            for stage in self.stages:
                out = stage.forward(out)
            outputs.append(out)

        return outputs if len(outputs) > 1 else outputs[0]

    # -------------------------------------------------------------------------
    # Backward pass
    # -------------------------------------------------------------------------
    def backward(self, loss):
        """
        Backward through pipeline.
        Uses AMP scaler if available.
        """
        if self.scaler is not None:
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()
        else:
            loss.backward()

    # -------------------------------------------------------------------------
    # Optimizer step across stages
    # -------------------------------------------------------------------------
    def step(self, optimizers: List[Any]):
        """
        Perform optimizer step for each stage.
        """
        if len(optimizers) != len(self.stages):
            raise ValueError("Number of optimizers must match number of stages")

        if self.scaler is not None:
            for opt in optimizers:
                self.scaler.step(opt)
            self.scaler.update()
        else:
            for opt in optimizers:
                opt.step()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _split_microbatches(self, inputs, microbatch_size):
        """Split inputs into microbatches for pipeline parallelism."""
        if isinstance(inputs[0], list):
            total = len(inputs[0])
        else:
            total = len(inputs)
        num_batches = math.ceil(total / microbatch_size)

        microbatches = []
        for i in range(num_batches):
            start = i * microbatch_size
            end = min(start + microbatch_size, total)
            if isinstance(inputs[0], list):
                batch = [x[start:end] for x in inputs]
            else:
                batch = inputs[start:end]
            microbatches.append(batch)
        return microbatches

    # -------------------------------------------------------------------------
    # Utility functions
    # -------------------------------------------------------------------------
    def synchronize(self):
        """Wait for all stages to finish (useful for CUDA/XPU)."""
        for stage in self.stages:
            b = stage.backend
            if hasattr(b, "synchronize"):
                try:
                    b.synchronize()
                except Exception:
                    pass

    def summary(self):
        print("============================================================")
        print("Pipeline Summary")
        print("============================================================")
        for i, s in enumerate(self.stages):
            print(f" Stage {i}: {s.model.__class__.__name__} on {s.device}")
        print("------------------------------------------------------------")
        print(f" Total stages: {self.num_stages}")
        print(f" Microbatch size: {self.microbatch_size}")
        print("============================================================")

    def __repr__(self):
        return f"<Pipeline stages={self.num_stages} microbatch={self.microbatch_size}>"
