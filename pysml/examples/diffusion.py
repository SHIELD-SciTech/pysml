"""Diffusion-style U-Net example for PySML.

The script sketches a small noise-prediction network inspired by diffusion
models. It demonstrates how to wire convolutional residual blocks, bilinear
upsampling, and simple mean-squared-error training with synthetic data.
"""
import numpy as np

import pysml
from pysml import Tensor
from pysml.nn import (
    Module,
    Conv2d,
    GroupNorm,
    UpsamplingBilinear2d,
    SiLU,
    MSELoss,
    AdamW,
    PipelineModule,
)


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
        residual = x if self.residual is None else self.residual(x)
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + residual)


class TinyDiffusionUNet(Module):
    """Minimal U-Net with a single downsample and upsample stage."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.conv_in = Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down = ResidualBlock(base_channels, base_channels * 2)
        self.downsample = Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1)
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


def generate_batch(batch_size: int, channels: int, height: int, width: int):
    clean = np.random.randn(batch_size, channels, height, width).astype(np.float32)
    noise = np.random.randn(batch_size, channels, height, width).astype(np.float32)
    noisy = clean + 0.1 * noise
    return build_tensor(noisy, requires_grad=False), build_tensor(noise, requires_grad=False)


def main() -> None:
    batch_size = 2
    channels = 3
    height = width = 32

    model = TinyDiffusionUNet(in_channels=channels, base_channels=32)
    optimizer = AdamW(model.parameters(), lr=2e-4)
    criterion = MSELoss()

    for step in range(5):
        inputs, targets = generate_batch(batch_size, channels, height, width)
        model.train()
        optimizer.zero_grad()
        pred_noise = model(inputs)
        loss = criterion(pred_noise, targets)
        loss.backward()
        optimizer.step()
        print(f"step={step:02d} loss={loss.item():.4f}")

    demonstrate_pipeline_unet()


def demonstrate_pipeline_unet() -> None:
    model = TinyDiffusionUNet()
    stages = [
        DownStage(model.conv_in, model.down, model.downsample),
        MidStage(model.mid),
        UpStage(model.upsample, model.up, model.conv_out),
    ]
    pipeline = PipelineModule(stages, partitions=[1, 1, 1], schedule="gpipe", chunks=2)
    inputs, _ = generate_batch(batch_size=2, channels=3, height=32, width=32)
    preds = pipeline(inputs)
    print("unet pipeline output shape:", preds.shape)
    print("unet pipeline metrics:", pipeline.profile())


if __name__ == "__main__":
    main()
