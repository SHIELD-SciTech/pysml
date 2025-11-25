"""RWKV pipeline/data-parallel demos using the custom DDP stack."""
from __future__ import annotations

from typing import Sequence

import numpy as np

import pysml
from pysml import ddp
from pysml.nn import CrossEntropyLoss, Sequential
from pysml.nn.optim import Adam

from . import rwkv


def _preferred_devices(devices: Sequence[str] | None, minimum: int = 4) -> list[str]:
    if devices:
        return [str(device) for device in devices]
    if pysml.xpu.is_available():
        xp_devices = pysml.xpu.get_available_devices()
        if xp_devices:
            return xp_devices[: max(1, minimum)]
    if pysml.cuda.is_available():
        return [f"cuda:{idx}" for idx in range(minimum)]
    return ["cpu"]


def _rwkv_pipeline_ctor() -> Sequential:
    model = rwkv.TinyRWKVModel(num_layers=4)
    stages = rwkv.build_rwkv_stages(model)
    return Sequential(*stages)


def train_rwkv_with_pipeline(
    *,
    devices: Sequence[str] | None = None,
    steps: int = 4,
    chunks: int = 4,
    seed: int = 0,
) -> dict:
    """Train the RWKV classifier with explicit pipeline stage placement."""

    mesh = _preferred_devices(devices)
    ddp.register_global_communicator(mesh)
    pipeline_ctor = ddp.PipelineParallel(_rwkv_pipeline_ctor, devices=mesh, chunks=chunks)
    model = pipeline_ctor()
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    losses: list[float] = []

    for step in range(steps):
        tokens, targets = rwkv.generate_batch(
            batch_size=4,
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
        print(f"[rwkv-ddp] step={step:02d} loss={value:.4f} devices={mesh}")

    return {"final_loss": losses[-1], "loss_history": losses, "devices": mesh}


def data_parallel_rwkv_logits(
    *, devices: Sequence[str] | None = None, seed: int = 1, chunks: int = 4
):
    """Evaluate the RWKV stack with `DataParallel` replicas."""

    mesh = _preferred_devices(devices)
    ddp.register_global_communicator(mesh)
    pipeline_ctor = ddp.PipelineParallel(_rwkv_pipeline_ctor, devices=mesh, chunks=chunks)
    data_parallel_ctor = ddp.DataParallel(lambda: pipeline_ctor(), devices=mesh)
    model = data_parallel_ctor()

    tokens, _ = rwkv.generate_batch(
        batch_size=mesh and len(mesh) * 2 or 2,
        seq_len=16,
        vocab_size=256,
        num_classes=4,
        rng=np.random.default_rng(seed),
    )
    tokens = tokens.to(mesh[0])
    return model(tokens)


def main() -> None:
    stats = train_rwkv_with_pipeline()
    print("final ddp loss:", stats["final_loss"], "on devices", stats["devices"])
    logits = data_parallel_rwkv_logits()
    print("ddp logits shape:", logits.shape)


if __name__ == "__main__":
    main()
