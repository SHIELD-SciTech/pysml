"""Transformer classifier and seq2seq walkthrough.

Transformer Example Overview
============================

This module contains two compact Transformer variants and demonstrates how to
drive them through ``ExampleConfig`` for different backends (CPU, CUDA, Intel
XPU) and degrees of data/pipeline/tensor parallelism. Every public helper has a
docstring with inline snippets so you can copy/paste the exact invocation when
experimenting with PySML's distributed features. The code is intentionally
verbose so the stage boundaries are obvious when feeding the network into
``PipelineModule``.
"""
from __future__ import annotations

import numpy as np

import pysml
from pysml import Tensor
from pysml.autograd import no_grad
from pysml.nn import (
    Module,
    Embedding,
    Linear,
    LayerNorm,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
    CrossEntropyLoss,
    AdamW,
    SinusoidalPositionalEncoding,
)

from .parallel_utils import ExampleConfig


DEFAULT_STEPS = 5


class TinyTransformerClassifier(Module):
    """Simple Transformer encoder classifier."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        num_classes: int = 4,
        max_length: int = 64,
    ) -> None:
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.position = SinusoidalPositionalEncoding(d_model, max_len=max_length)
        encoder_layer = TransformerEncoderLayer(
            d_model,
            num_heads,
            d_ff=4 * d_model,
            dropout=0.1,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        self.output_norm = LayerNorm(d_model)
        self.head = Linear(d_model, num_classes)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embedding(input_ids)
        x = self.position(x)
        x = self.encoder(x)
        x = self.output_norm(x)
        pooled = x.mean(axis=1)
        return self.head(pooled)


class TinySeq2Seq(Module):
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        max_length: int = 64,
    ) -> None:
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.position = SinusoidalPositionalEncoding(d_model, max_len=max_length)
        encoder_layer = TransformerEncoderLayer(
            d_model,
            num_heads,
            d_ff=4 * d_model,
            dropout=0.0,
            activation="gelu",
            norm_first=True,
        )
        decoder_layer = TransformerDecoderLayer(
            d_model,
            num_heads,
            d_ff=4 * d_model,
            dropout=0.0,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)
        self.head = Linear(d_model, vocab_size)

    def forward(self, src_tokens: Tensor, tgt_tokens: Tensor) -> Tensor:
        src = self.embedding(src_tokens)
        src = self.position(src)
        memory = self.encoder(src)
        tgt = self.embedding(tgt_tokens)
        tgt = self.position(tgt)
        hidden = self.decoder(tgt, memory)
        return self.head(hidden)


class SourceEmbeddingStage(Module):
    def __init__(self, embedding: Module, position: Module) -> None:
        super().__init__()
        self.embedding = embedding
        self.position = position

    def forward(self, src_tokens: Tensor, tgt_tokens: Tensor):
        src = self.embedding(src_tokens)
        src = self.position(src)
        return src, tgt_tokens


class EncoderStage(Module):
    def __init__(self, encoder: Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, src_hidden: Tensor, tgt_tokens: Tensor):
        memory = self.encoder(src_hidden)
        return memory, tgt_tokens


class DecoderStage(Module):
    def __init__(self, decoder: Module, embedding: Module, position: Module) -> None:
        super().__init__()
        self.decoder = decoder
        self.embedding = embedding
        self.position = position

    def forward(self, memory: Tensor, tgt_tokens: Tensor):
        tgt = self.embedding(tgt_tokens)
        tgt = self.position(tgt)
        hidden = self.decoder(tgt, memory)
        return hidden,


class ProjectionStage(Module):
    def __init__(self, head: Module) -> None:
        super().__init__()
        self.head = head

    def forward(self, hidden: Tensor):
        logits = self.head(hidden)
        pooled = logits.mean(axis=1)
        return pooled


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
    """Return integer tensors suitable for classifier or seq2seq tests."""
    rng = rng or np.random.default_rng()
    tokens = rng.integers(0, vocab_size, size=(batch_size, seq_len), dtype=np.int64)
    targets = rng.integers(0, num_classes, size=(batch_size,), dtype=np.int64)
    return build_int_tensor(tokens), build_int_tensor(targets)


def build_transformer_stages(vocab_size: int, d_model: int = 64) -> list[Module]:
    """Assemble four intuitive pipeline stages (src embed, encoder, decoder, head)."""
    src_embedding = Embedding(vocab_size, d_model)
    tgt_embedding = Embedding(vocab_size, d_model)
    src_position = SinusoidalPositionalEncoding(d_model, max_len=64)
    tgt_position = SinusoidalPositionalEncoding(d_model, max_len=64)
    encoder_layer = TransformerEncoderLayer(
        d_model,
        4,
        d_ff=4 * d_model,
        dropout=0.0,
        activation="gelu",
        norm_first=True,
    )
    decoder_layer = TransformerDecoderLayer(
        d_model,
        4,
        d_ff=4 * d_model,
        dropout=0.0,
        activation="gelu",
        norm_first=True,
    )
    encoder = TransformerEncoder(encoder_layer, 2)
    decoder = TransformerDecoder(decoder_layer, 2)
    head = Linear(d_model, vocab_size)
    return [
        SourceEmbeddingStage(src_embedding, src_position),
        EncoderStage(encoder),
        DecoderStage(decoder, tgt_embedding, tgt_position),
        ProjectionStage(head),
    ]


def train_example(
    config: ExampleConfig, steps: int = DEFAULT_STEPS, *, use_pipeline: bool = False
) -> dict:
    """Train the classifier for a few steps and return metrics.

    The helper mirrors ``rwkv.train_example``: pass a configured
    :class:`~pysml.examples.parallel_utils.ExampleConfig` and set
    ``use_pipeline=True`` to wrap the seq2seq model with ``PipelineModule``.
    Because PySML does not yet exchange tensors across ranks, pipeline mode
    should be viewed as instrumentation rather than true distributed execution.
    """
    batch_size = 8
    seq_len = 32
    vocab_size = 256
    num_classes = 4

    np.random.seed(0)
    base_model = TinyTransformerClassifier(
        vocab_size=vocab_size, num_classes=num_classes
    )
    if use_pipeline:
        model = config.apply(
            base_model,
            pipeline_stages=build_transformer_stages(vocab_size),
            pipeline_kwargs={"partitions": [1, 1, 1, 1]},
        )
    else:
        model = config.apply(base_model)
    optimizer = AdamW(model.parameters(), lr=3e-4)
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
        value = float(loss.item())
        losses.append(value)
        print(f"[transformer/{config.device()}] step={step:02d} loss={value:.4f}")

    demonstrate_pipeline_split(vocab_size, config=config)
    return {"final_loss": losses[-1], "loss_history": losses}


def deterministic_logits(config: ExampleConfig, seed: int = 0) -> Tensor:
    """Create deterministic logits for regression tests and docs."""
    np.random.seed(seed)
    model = TinyTransformerClassifier()
    model = config.apply(model)
    model.eval()
    rng = np.random.default_rng(seed)
    tokens, _ = generate_batch(2, 16, 256, 4, rng=rng)
    with no_grad():
        logits = model(tokens)
    return logits


def demonstrate_pipeline_split(
    vocab_size: int, config: ExampleConfig | None = None
) -> None:
    """Show how the seq2seq model can be partitioned into four stages."""
    cfg = config or ExampleConfig()
    pipeline = cfg.apply(
        TinyTransformerClassifier(vocab_size=vocab_size),
        pipeline_stages=build_transformer_stages(vocab_size),
        pipeline_kwargs={"partitions": [1, 1, 1, 1]},
    )
    src, _ = generate_batch(
        batch_size=4, seq_len=16, vocab_size=vocab_size, num_classes=vocab_size
    )
    tgt, _ = generate_batch(
        batch_size=4, seq_len=16, vocab_size=vocab_size, num_classes=vocab_size
    )
    src = src.to(cfg.device())
    tgt = tgt.to(cfg.device())
    logits = pipeline(src, tgt)
    metrics = pipeline.profile()
    print("pipeline logits shape:", logits.shape)
    print("pipeline metrics:", metrics)


def main() -> None:
    train_example(ExampleConfig())


if __name__ == "__main__":
    main()
