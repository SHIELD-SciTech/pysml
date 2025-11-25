"""Transformer training demos that rely on the custom DDP runtime.

The functions below show how to combine ``PipelineParallel`` and
``DataParallel`` so the Transformer classifier can rehearse Intel XPU
splits without touching ``torch.distributed``. Each helper accepts an
optional device list (defaults to the first two XPUs reported by
``pysml.xpu.get_available_devices``) and falls back to CPU when hardware
is unavailable.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

import pysml
from pysml import ddp
from pysml.nn import CrossEntropyLoss
from pysml.nn.optim import AdamW

from . import transformer


def _preferred_devices(devices: Sequence[str] | None, minimum: int = 2) -> list[str]:
    if devices:
        return [str(device) for device in devices]
    if pysml.xpu.is_available():
        xp_devices = pysml.xpu.get_available_devices()
        if xp_devices:
            return xp_devices[: max(1, minimum)]
    if pysml.cuda.is_available():
        return [f"cuda:{idx}" for idx in range(minimum)]
    return ["cpu"]


def train_classifier_with_pipeline(
    *,
    devices: Sequence[str] | None = None,
    steps: int = 4,
    chunks: int = 2,
    seed: int = 0,
) -> dict:
    """Run a tiny Transformer classifier wrapped in ``PipelineParallel``.

    The helper mirrors :func:`pysml.examples.transformer.train_example` but
    wires the model through :class:`pysml.ddp.parallel.pipeline_parallel.PipelineParallel`
    so each stage runs on a specific XPU. It returns the loss history plus the
    resolved device mesh for debugging.
    """

    mesh = _preferred_devices(devices)
    ddp.register_global_communicator(mesh)
    pipeline_ctor = ddp.PipelineParallel(
        lambda: transformer.TinyTransformerClassifier(num_layers=3),
        devices=mesh,
        chunks=chunks,
    )
    model = pipeline_ctor()
    optimizer = AdamW(model.parameters(), lr=3e-4)
    criterion = CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    losses: list[float] = []

    for step in range(steps):
        tokens, targets = transformer.generate_batch(
            batch_size=8,
            seq_len=32,
            vocab_size=256,
            num_classes=4,
            rng=rng,
        )
        tokens = tokens.to(mesh[0])
        targets = targets.to(mesh[0])
        optimizer.zero_grad()
        logits = model(tokens)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        value = float(loss.item())
        losses.append(value)
        print(f"[transformer-ddp] step={step:02d} loss={value:.4f} devices={mesh}")

    return {"final_loss": losses[-1], "loss_history": losses, "devices": mesh}


def data_parallel_logits(
    *, devices: Sequence[str] | None = None, seed: int = 1, chunks: int = 2
):
    """Return batched logits computed via ``DataParallel`` replicas."""

    mesh = _preferred_devices(devices)
    ddp.register_global_communicator(mesh)
    pipeline_ctor = ddp.PipelineParallel(
        lambda: transformer.TinyTransformerClassifier(num_layers=3),
        devices=mesh,
        chunks=chunks,
    )
    data_parallel_ctor = ddp.DataParallel(lambda: pipeline_ctor(), devices=mesh)
    model = data_parallel_ctor()

    tokens, _ = transformer.generate_batch(
        batch_size=mesh and len(mesh) * 2 or 2,
        seq_len=16,
        vocab_size=256,
        num_classes=4,
        rng=np.random.default_rng(seed),
    )
    tokens = tokens.to(mesh[0])
    return model(tokens)


def main() -> None:
    stats = train_classifier_with_pipeline()
    print("final ddp loss:", stats["final_loss"], "on devices", stats["devices"])
    logits = data_parallel_logits()
    print("data-parallel logits shape:", logits.shape)


if __name__ == "__main__":
    main()
