"""RWKV example covering CPU, CUDA, and Intel XPU backends.

RWKV Example Overview
=====================

This module provides a bite-sized RWKV-like network together with helpers that
showcase PySML's :class:`~pysml.examples.parallel_utils.ExampleConfig`
orchestration. Each helper documents how to run on CPU, CUDA, or Intel XPU
devices and how to toggle prototype pipeline parallel execution. Although
``PipelineModule`` does not yet hand off tensors between ranks, the functions in
this file illustrate how to partition RWKV blocks, collect per-stage latency
metrics, and experiment with activation checkpointing.

Typical usage::

    from pysml.examples import rwkv
    from pysml.examples.parallel_utils import ExampleConfig

    config = ExampleConfig(backend="xpu", pipeline_parallel=4, pipeline_chunks=4)
    rwkv.train_example(config, steps=8, use_pipeline=True)

    # When you only need a quick latency breakdown per stage:
    rwkv.demonstrate_pipeline_segments(vocab_size=256, config=config)

The rest of the file spells out the building blocks (embedding, stacked RWKV
blocks, and the classifier head) so the partitioning points are easy to follow.
"""
from __future__ import annotations

import numpy as np

import pysml
from pysml import Tensor
from pysml.autograd import no_grad
from pysml.nn import (
    Module,
    ModuleList,
    Embedding,
    Linear,
    LayerNorm,
    CrossEntropyLoss,
    Adam,
    Sigmoid,
    Tanh,
    attention_free_time_mix,
)

from .parallel_utils import ExampleConfig


DEFAULT_STEPS = 5


class TinyRWKVBlock(Module):
    """Simplified RWKV block with time and channel mixing."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.time_norm = LayerNorm(hidden_size)
        self.channel_norm = LayerNorm(hidden_size)
        self.key = Linear(hidden_size, hidden_size)
        self.value = Linear(hidden_size, hidden_size)
        self.receptance = Linear(hidden_size, hidden_size)
        self.mix = Linear(hidden_size, hidden_size)
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        decay = np.full((hidden_size,), 0.5, dtype=np.float32)
        self.register_buffer("time_decay", Tensor(decay, requires_grad=False))

    def init_state(self, batch_size: int, device: str = "cpu"):
        zeros = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        state = {
            "avg_key": Tensor(zeros, requires_grad=False, device=device),
            "avg_value": Tensor(zeros, requires_grad=False, device=device),
            "output": Tensor(zeros, requires_grad=False, device=device),
        }
        return state

    def forward(self, x: Tensor, state):
        h = self.time_norm(x)
        k = self.key(h)
        v = self.value(h)
        r = self.sigmoid(self.receptance(h))

        mix, avg_key, avg_value = attention_free_time_mix(
            state["avg_key"], state["avg_value"], k, v, self.time_decay
        )
        gated = pysml.multiply(r, mix)

        channel = self.channel_norm(x + gated)
        channel = self.tanh(self.mix(channel))

        new_state = {
            "avg_key": avg_key,
            "avg_value": avg_value,
            "output": channel,
        }
        return channel, new_state


class TinyRWKVModel(Module):
    """Stack a few RWKV-style blocks and expose a classifier head."""

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_size: int = 64,
        num_layers: int = 3,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.embedding = Embedding(vocab_size, hidden_size)
        self.blocks = ModuleList([TinyRWKVBlock(hidden_size) for _ in range(num_layers)])
        self.head = Linear(hidden_size, num_classes)

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.embedding(tokens)
        batch_size = x.shape[0]
        states = [block.init_state(batch_size, device=x.active_device) for block in self.blocks]

        steps = pysml.split(x, 1, dim=1)
        last = None
        for step in steps:
            step = pysml.squeeze(step, axis=1)
            for idx, block in enumerate(self.blocks):
                step, states[idx] = block(step, states[idx])
            last = step

        return self.head(last)


class RWKVEmbeddingStage(Module):
    def __init__(self, embedding: Module) -> None:
        super().__init__()
        self.embedding = embedding

    def forward(self, tokens: Tensor) -> Tensor:
        return self.embedding(tokens)


class RWKVBlockStack(Module):
    def __init__(self, blocks: ModuleList) -> None:
        super().__init__()
        self.blocks = blocks

    def forward(self, hidden: Tensor) -> Tensor:
        batch_size = hidden.shape[0]
        states = [block.init_state(batch_size, device=hidden.active_device) for block in self.blocks]
        steps = pysml.split(hidden, 1, dim=1)
        outputs = []
        for step in steps:
            step = pysml.squeeze(step, axis=1)
            for idx, block in enumerate(self.blocks):
                step, states[idx] = block(step, states[idx])
            outputs.append(step)
        return pysml.stack(outputs, axis=1)


class RWKVHeadStage(Module):
    def __init__(self, head: Module) -> None:
        super().__init__()
        self.head = head

    def forward(self, hidden: Tensor) -> Tensor:
        last = pysml.squeeze(hidden[:, -1:, :], axis=1)
        return self.head(last)


def build_int_tensor(array: np.ndarray) -> Tensor:
    tensor = Tensor(array, requires_grad=False)
    tensor.data = array.astype(np.int64, copy=False)
    tensor._requires_grad = False
    tensor._grad = None
    return tensor


def generate_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_classes: int,
    *,
    rng: np.random.Generator | None = None,
):
    """Generate a deterministic batch of toy data for regression tests."""
    rng = rng or np.random.default_rng()
    tokens = rng.integers(0, vocab_size, size=(batch_size, seq_len), dtype=np.int64)
    targets = rng.integers(0, num_classes, size=(batch_size,), dtype=np.int64)
    return build_int_tensor(tokens), build_int_tensor(targets)


def build_rwkv_stages(model: TinyRWKVModel) -> list[Module]:
    """Split the RWKV model into four intuitive pipeline stages."""
    mid = len(model.blocks) // 2 or 1
    first = ModuleList(list(model.blocks)[:mid])
    second = ModuleList(list(model.blocks)[mid:])
    return [
        RWKVEmbeddingStage(model.embedding),
        RWKVBlockStack(first),
        RWKVBlockStack(second),
        RWKVHeadStage(model.head),
    ]


def train_example(
    config: ExampleConfig, steps: int = DEFAULT_STEPS, *, use_pipeline: bool = False
) -> dict:
    """Train the toy RWKV classifier with rich logging.

    Parameters
    ----------
    config:
        ``ExampleConfig`` describing the backend plus data/pipeline/tensor degrees.
    steps:
        Number of optimisation steps to run.
    use_pipeline:
        When ``True`` the model is wrapped in :class:`~pysml.nn.pipeline.PipelineModule`
        using ``build_rwkv_stages``. This remains single-process execution but
        exposes micro-batch scheduling and profiling hooks.

    Returns
    -------
    dict
        A ``{"final_loss": float, "loss_history": list}`` dictionary useful for
        quick regression tests.
    """
    batch_size = 4
    seq_len = 32
    vocab_size = 256
    num_classes = 4

    np.random.seed(0)
    base_model = TinyRWKVModel(vocab_size=vocab_size, num_classes=num_classes)
    if use_pipeline:
        model = config.apply(
            base_model,
            pipeline_stages=build_rwkv_stages(base_model),
            pipeline_kwargs={"partitions": [1, 1, 1, 1]},
        )
    else:
        model = config.apply(base_model)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = CrossEntropyLoss()

    rng = np.random.default_rng(0)
    losses = []
    for step in range(steps):
        input_ids, targets = generate_batch(
            batch_size, seq_len, vocab_size, num_classes, rng=rng
        )
        model.train()
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())
        losses.append(loss_value)
        print(f"[rwkv/{config.device()}] step={step:02d} loss={loss_value:.4f}")

    demonstrate_pipeline_segments(vocab_size, config=config)
    return {"final_loss": losses[-1], "loss_history": losses}


def deterministic_logits(config: ExampleConfig, seed: int = 0) -> Tensor:
    """Produce logits without randomness so tests can compare outputs."""
    np.random.seed(seed)
    model = TinyRWKVModel()
    model = config.apply(model)
    model.eval()
    rng = np.random.default_rng(seed)
    tokens, _ = generate_batch(2, 16, 256, 4, rng=rng)
    with no_grad():
        logits = model(tokens)
    return logits


def demonstrate_pipeline_segments(
    vocab_size: int, config: ExampleConfig | None = None
) -> None:
    """Print pipeline metrics for a staged RWKV model.

    The helper is intentionally side-effectful: it builds a reference
    ``TinyRWKVModel``, applies the provided ``ExampleConfig`` (or a default
    CPU-only config), and reports tensor shapes plus the latest pipeline metrics.
    Because this routine feeds random data you can call it in exploratory
    notebooks without touching the training loop.
    """
    cfg = config or ExampleConfig()
    model = TinyRWKVModel(vocab_size=vocab_size)
    pipeline = cfg.apply(
        model,
        pipeline_stages=build_rwkv_stages(model),
        pipeline_kwargs={"partitions": [1, 1, 1, 1]},
    )
    tokens, _ = generate_batch(
        batch_size=4,
        seq_len=32,
        vocab_size=vocab_size,
        num_classes=vocab_size,
    )
    tokens = tokens.to(cfg.device())
    logits = pipeline(tokens)
    print("rwkv pipeline logits shape:", logits.shape)
    print("rwkv pipeline metrics:", pipeline.profile())


def main() -> None:
    train_example(ExampleConfig())


if __name__ == "__main__":
    main()
