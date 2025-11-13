"""Minimal RWKV-style recurrent example for PySML.

The goal is to demonstrate how to assemble a light-weight RWKV-inspired block
with the primitives provided by ``pysml.nn``. The block keeps a running state to
blend new keys/values with an exponential decay, applies simple channel mixing,
and drives a classifier head.
"""
from __future__ import annotations

import numpy as np

import pysml
from pysml import Tensor
from pysml.distributed import ParallelStrategy
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
)


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

    def init_state(self, batch_size: int):
        zeros = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        state = {
            "avg_key": Tensor(zeros, requires_grad=False),
            "avg_value": Tensor(zeros, requires_grad=False),
            "output": Tensor(zeros, requires_grad=False),
        }
        return state

    def forward(self, x: Tensor, state):
        h = self.time_norm(x)
        k = self.key(h)
        v = self.value(h)
        r = self.sigmoid(self.receptance(h))

        decay = self.time_decay
        avg_key = pysml.add(pysml.multiply(state["avg_key"], decay), pysml.multiply(k, 1 - decay))
        avg_value = pysml.add(pysml.multiply(state["avg_value"], decay), pysml.multiply(v, 1 - decay))

        mix = pysml.multiply(avg_key, avg_value)
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
        states = [block.init_state(batch_size) for block in self.blocks]

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
        states = [block.init_state(batch_size) for block in self.blocks]
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


def generate_batch(batch_size: int, seq_len: int, vocab_size: int, num_classes: int):
    tokens = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int64)
    targets = np.random.randint(0, num_classes, size=(batch_size,), dtype=np.int64)
    return build_int_tensor(tokens), build_int_tensor(targets)


def main(strategy: ParallelStrategy | None = None) -> None:
    batch_size = 4
    seq_len = 32
    vocab_size = 256
    num_classes = 4

    model = TinyRWKVModel(vocab_size=vocab_size, num_classes=num_classes)
    if strategy is not None:
        model = strategy.apply(model)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = CrossEntropyLoss()

    for step in range(5):
        input_ids, targets = generate_batch(batch_size, seq_len, vocab_size, num_classes)
        model.train()
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        print(f"step={step:02d} loss={loss.item():.4f}")

    demonstrate_pipeline_segments(vocab_size)


def demonstrate_pipeline_segments(
    vocab_size: int, strategy: ParallelStrategy | None = None
) -> None:
    model = TinyRWKVModel(vocab_size=vocab_size)
    mid = len(model.blocks) // 2 or 1
    first = ModuleList(list(model.blocks)[:mid])
    second = ModuleList(list(model.blocks)[mid:])
    stages = [
        RWKVEmbeddingStage(model.embedding),
        RWKVBlockStack(first),
        RWKVBlockStack(second),
        RWKVHeadStage(model.head),
    ]
    pipeline_strategy = strategy or ParallelStrategy.pipeline(
        len(stages), schedule="gpipe", chunks=4
    )
    pipeline = pipeline_strategy.apply(
        pipeline_stages=stages,
        pipeline_kwargs={"partitions": [1, 1, 1, 1]},
    )
    tokens, _ = generate_batch(batch_size=4, seq_len=32, vocab_size=vocab_size, num_classes=vocab_size)
    logits = pipeline(tokens)
    print("rwkv pipeline logits shape:", logits.shape)
    print("rwkv pipeline metrics:", pipeline.profile())


if __name__ == "__main__":
    main()
