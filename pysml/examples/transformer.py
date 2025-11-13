"""Minimal Transformer encoder example for PySML.

This script builds a tiny Transformer-based text classifier using the high level
building blocks provided by ``pysml.nn``. It shows how to wire embeddings,
positional encodings, ``TransformerEncoderLayer`` stacks, and a classification
head together, then run a few dummy optimization steps.
"""
import numpy as np

import pysml
from pysml import Tensor
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
    PipelineModule,
)


class TinyTransformerClassifier(Module):
    """Simple Transformer encoder classifier.

    Parameters
    ----------
    vocab_size:
        Number of tokens available to the embedding table.
    d_model:
        Hidden size of the Transformer blocks.
    num_layers:
        Number of encoder layers in the stack.
    num_heads:
        Attention heads per layer.
    num_classes:
        Output classes for the classifier head.
    max_length:
        Maximum sequence length supported by the positional encoding buffer.
    """

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
        """Run the encoder and return logits of shape ``(batch, num_classes)``."""

        x = self.embedding(input_ids)
        x = self.position(x)
        x = self.encoder(x)
        x = self.output_norm(x)
        pooled = x.mean(axis=1)  # Average pool over sequence length
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
    """Create a Tensor wrapper around an integer array without casting to float."""

    tensor = Tensor(array, requires_grad=False)
    tensor.data = array.astype(np.int64, copy=False)
    tensor._requires_grad = False
    tensor._grad = None
    return tensor


def generate_batch(batch_size: int, seq_len: int, vocab_size: int, num_classes: int):
    tokens = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int64)
    targets = np.random.randint(0, num_classes, size=(batch_size,), dtype=np.int64)
    return build_int_tensor(tokens), build_int_tensor(targets)


def main() -> None:
    batch_size = 8
    seq_len = 32
    vocab_size = 256
    num_classes = 4

    model = TinyTransformerClassifier(vocab_size=vocab_size, num_classes=num_classes)
    optimizer = AdamW(model.parameters(), lr=3e-4)
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

    demonstrate_pipeline_split(vocab_size)


def demonstrate_pipeline_split(vocab_size: int) -> None:
    d_model = 64
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
    stages = [
        SourceEmbeddingStage(src_embedding, src_position),
        EncoderStage(encoder),
        DecoderStage(decoder, tgt_embedding, tgt_position),
        ProjectionStage(head),
    ]
    pipeline = PipelineModule(
        stages,
        partitions=[1, 1, 1, 1],
        schedule="1f1b",
        chunks=2,
        activation_checkpoint=True,
    )
    src, _ = generate_batch(batch_size=4, seq_len=16, vocab_size=vocab_size, num_classes=vocab_size)
    tgt, _ = generate_batch(batch_size=4, seq_len=16, vocab_size=vocab_size, num_classes=vocab_size)
    logits = pipeline(src, tgt)
    metrics = pipeline.profile()
    print("pipeline logits shape:", logits.shape)
    print("pipeline metrics:", metrics)


if __name__ == "__main__":
    main()
