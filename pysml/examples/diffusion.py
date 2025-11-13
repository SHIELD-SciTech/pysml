"""Diffusion U-Net walkthrough with detailed commentary.

Diffusion Example Overview
==========================

Just like the RWKV and Transformer modules, this script wraps a compact U-Net in
``ExampleConfig`` glue so you can flip between CPU, CUDA, and Intel XPU devices.
Docstrings highlight how to run local regression tests, how to enable the
prototype pipeline mode, and what each helper returns. Use it as a verbose
reference when porting larger diffusion architectures onto PySML.
"""
from __future__ import annotations

import numpy as np

import pysml
from pysml import Tensor
from pysml.autograd import no_grad
from pysml.nn import (
    Module,
    Conv2d,
    GroupNorm,
    UpsamplingBilinear2d,
    SiLU,
    MSELoss,
    AdamW,
    convolutional_residual_block,
)

from .parallel_utils import ExampleConfig


DEFAULT_STEPS = 5


class ResidualBlock(Module):
    """Basic residual block with GroupNorm and SiLU activations."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = GroupNorm(8, out_channels)
        self.act = SiLU()
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = GroupNorm(8, out_channels)
        self.residual = None
        if in_channels != out_channels:
            self.residual = Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return convolutional_residual_block(
            x,
            self.conv1,
            self.norm1,
            self.act,
            self.conv2,
            self.norm2,
            residual_conv=self.residual,
        )


class TinyDiffusionUNet(Module):
    """Minimal U-Net with a single downsample and upsample stage."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.conv_in = Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down = ResidualBlock(base_channels, base_channels * 2)
        self.downsample = Conv2d(
            base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1
        )
        self.mid = ResidualBlock(base_channels * 2, base_channels * 2)
        self.upsample = UpsamplingBilinear2d(scale_factor=2)
        self.up = ResidualBlock(base_channels * 2, base_channels)
        self.conv_out = Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        h1 = self.conv_in(x)
        h2 = self.down(h1)
        h3 = self.downsample(h2)
        h3 = self.mid(h3)
        h3 = self.upsample(h3)
        h3 = pysml.add(h3, h2)
        h3 = self.up(h3)
        return self.conv_out(h3)


class DownStage(Module):
    def __init__(self, conv_in: Module, down: Module, downsample: Module) -> None:
        super().__init__()
        self.conv_in = conv_in
        self.down = down
        self.downsample = downsample

    def forward(self, x: Tensor):
        h1 = self.conv_in(x)
        h2 = self.down(h1)
        h3 = self.downsample(h2)
        return h3, h2


class MidStage(Module):
    def __init__(self, mid: Module) -> None:
        super().__init__()
        self.mid = mid

    def forward(self, encoded: Tensor, skip: Tensor):
        return self.mid(encoded), skip


class UpStage(Module):
    def __init__(self, upsample: Module, up: Module, conv_out: Module) -> None:
        super().__init__()
        self.upsample = upsample
        self.up = up
        self.conv_out = conv_out

    def forward(self, encoded: Tensor, skip: Tensor):
        h3 = self.upsample(encoded)
        h3 = pysml.add(h3, skip)
        h3 = self.up(h3)
        return self.conv_out(h3)


def build_tensor(array: np.ndarray, requires_grad: bool = False) -> Tensor:
    return Tensor(array, requires_grad=requires_grad)


def generate_batch(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    *,
    rng: np.random.Generator | None = None,
):
    """Return noisy inputs and clean noise targets for diffusion loss."""
    rng = rng or np.random.default_rng()
    clean = rng.standard_normal(size=(batch_size, channels, height, width)).astype(
        np.float32
    )
    noise = rng.standard_normal(size=(batch_size, channels, height, width)).astype(
        np.float32
    )
    noisy = clean + 0.1 * noise
    return build_tensor(noisy, requires_grad=False), build_tensor(noise, requires_grad=False)


def build_unet_stages(model: TinyDiffusionUNet) -> list[Module]:
    """Split the U-Net into down, mid, and up pipeline chunks."""
    return [
        DownStage(model.conv_in, model.down, model.downsample),
        MidStage(model.mid),
        UpStage(model.upsample, model.up, model.conv_out),
    ]


def train_example(
    config: ExampleConfig, steps: int = DEFAULT_STEPS, *, use_pipeline: bool = False
) -> dict:
    """Train the toy U-Net for a few diffusion-style denoising steps."""
    batch_size = 2
    channels = 3
    height = width = 32

    np.random.seed(0)
    base_model = TinyDiffusionUNet(in_channels=channels, base_channels=32)
    if use_pipeline:
        model = config.apply(
            base_model,
            pipeline_stages=build_unet_stages(base_model),
            pipeline_kwargs={"partitions": [1, 1, 1]},
        )
    else:
        model = config.apply(base_model)
    optimizer = AdamW(model.parameters(), lr=2e-4)
    criterion = MSELoss()

    rng = np.random.default_rng(0)
    losses: list[float] = []
    for step in range(steps):
        inputs, targets = generate_batch(batch_size, channels, height, width, rng=rng)
        model.train()
        optimizer.zero_grad()
        pred_noise = model(inputs)
        loss = criterion(pred_noise, targets)
        loss.backward()
        optimizer.step()
        value = float(loss.item())
        losses.append(value)
        print(f"[diffusion/{config.device()}] step={step:02d} loss={value:.4f}")

    demonstrate_pipeline_unet(config=config)
    return {"final_loss": losses[-1], "loss_history": losses}


def deterministic_prediction(config: ExampleConfig, seed: int = 0) -> Tensor:
    """Create deterministic predictions for regression tests and docs."""
    np.random.seed(seed)
    model = TinyDiffusionUNet()
    model = config.apply(model)
    model.eval()
    rng = np.random.default_rng(seed)
    inputs, _ = generate_batch(1, 3, 16, 16, rng=rng)
    with no_grad():
        preds = model(inputs)
    return preds


def demonstrate_pipeline_unet(config: ExampleConfig | None = None) -> None:
    """Showcase the three-stage pipeline split and print profiling metrics."""
    cfg = config or ExampleConfig()
    model = TinyDiffusionUNet()
    pipeline = cfg.apply(
        model,
        pipeline_stages=build_unet_stages(model),
        pipeline_kwargs={"partitions": [1, 1, 1]},
    )
    inputs, _ = generate_batch(batch_size=2, channels=3, height=32, width=32)
    inputs = inputs.to(cfg.device())
    preds = pipeline(inputs)
    print("unet pipeline output shape:", preds.shape)
    print("unet pipeline metrics:", pipeline.profile())


def main() -> None:
    train_example(ExampleConfig())


if __name__ == "__main__":
    main()
