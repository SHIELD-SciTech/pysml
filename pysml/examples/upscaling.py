"""Lightweight image upscaling demo for CPU, CUDA, and Intel XPU."""
from __future__ import annotations

import numpy as np

import pysml
from pysml import Tensor
from pysml.autograd import no_grad
from pysml.nn import Conv2d, Module, Sequential, Tanh, Upsample


def choose_device(preferred: str | None = None) -> str:
    """Select a working device string, preferring the requested one."""
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(["cuda", "xpu", "cpu"])

    for name in candidates:
        if name.startswith("cuda") and getattr(pysml.cuda, "is_available", lambda: False)():
            return name
        if name.startswith("xpu") and getattr(pysml.xpu, "is_available", lambda: False)():
            return name
        if name == "cpu":
            return name
    return "cpu"


class SimpleUpscaler(Module):
    """Two-stage network: resize then refine with a shallow conv block."""

    def __init__(self, scale_factor: int = 2) -> None:
        super().__init__()
        self.net = Sequential(
            Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False),
            Conv2d(3, 16, kernel_size=3, padding=1),
            Tanh(),
            Conv2d(16, 3, kernel_size=1, padding=0),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def run_demo(scale_factor: int = 2, preferred_device: str | None = None) -> Tensor:
    """Run the upscaler on a synthetic image and return the output tensor."""

    device = choose_device(preferred_device)
    print(f"Using device: {device}")

    model = SimpleUpscaler(scale_factor=scale_factor)
    model.to(device)

    pattern = np.linspace(0.0, 1.0, 32 * 32 * 3, dtype=np.float32).reshape(1, 3, 32, 32)
    input_image = Tensor(pattern, device=device)

    with no_grad():
        upscaled = model(input_image)

    print(f"Upscaled shape: {upscaled.shape}")
    return upscaled


if __name__ == "__main__":
    run_demo()
