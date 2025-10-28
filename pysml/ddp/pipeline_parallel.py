"""
Pipeline Parallel Training — optimized
- Minimal device transfers
- Optional micro-batching to reduce bubbles/memory
- Keeps API and printed summaries stable
"""

import pysml
import pysml.nn as nn
import pysml.nn.functional as F
from pysml.nn.models import TransformerBlock, LayerNorm
from typing import List, Optional


class PipelineModule:
    def __init__(self, devices: List[str]):
        self.devices = devices
        self.num_devices = len(devices)

    def distribute_layers(self, num_layers: int) -> List[tuple]:
        layers_per = num_layers // self.num_devices
        rem = num_layers % self.num_devices
        dist = []
        cur = 0
        for i, dev in enumerate(self.devices):
            n = layers_per + (1 if i < rem else 0)
            dist.append((cur, cur + n, dev))
            cur += n
        return dist

    def print_distribution(self, layer_names: Optional[List[str]] = None):
        print(f"\n{'='*80}")
        print("Pipeline Parallel Layer Distribution")
        print(f"{'='*80}")
        if layer_names:
            for idx, name in enumerate(layer_names):
                print(f"  Layer {idx:2d} ({name:20s})")
        print(f"{'='*80}")


class PipelineTransformer(nn.Module):
    """
    Pipeline-parallel Transformer with micro-batching.

    Args:
        devices: list of stage devices
        micro_batches: int micro-batches (>= num_devices recommended to reduce bubbles)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        devices: List[str],
        dropout: float = 0.1,
        micro_batches: int = 1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout_p = float(dropout)
        self.devices = devices
        self.num_devices = len(devices)
        self.micro_batches = max(1, int(micro_batches))

        print(f"\n{'='*80}")
        print(f"Initializing Pipeline-Parallel TransformerLM")
        print(f"{'='*80}")
        print(f"  vocab_size={vocab_size}, d_model={d_model}")
        print(f"  num_layers={num_layers}, num_heads={num_heads}")
        print(f"  d_ff={d_ff}, max_seq_len={max_seq_len}")
        print(f"  Devices: {devices}")

        layers_per_device = num_layers // self.num_devices
        remainder = num_layers % self.num_devices
        print(f"\nPipeline Parallelism Strategy:")
        print(f"  Layers per device: ~{layers_per_device}")

        # Embeddings on first device (avoid transfers during token lookup)
        self.token_embedding = pysml.randn(vocab_size, d_model, requires_grad=True) * 0.02
        self.pos_embedding = pysml.randn(max_seq_len, d_model, requires_grad=True) * 0.02
        self.embedding_device = devices[0]
        print(f"  Embeddings -> {self.embedding_device}")

        # Distribute transformer blocks
        self.blocks = nn.ModuleList()
        self.block_devices = []

        cur_dev = 0
        layers_on_cur = 0
        layers_for_cur = layers_per_device + (1 if cur_dev < remainder else 0)

        for layer_idx in range(num_layers):
            block = TransformerBlock(d_model, num_heads, d_ff, dropout)
            self.blocks.append(block)
            dev = devices[cur_dev]
            self.block_devices.append(dev)
            print(f"  Layer {layer_idx:2d} -> {dev}")
            layers_on_cur += 1
            if layers_on_cur >= layers_for_cur:
                cur_dev += 1
                if cur_dev < self.num_devices:
                    layers_on_cur = 0
                    layers_for_cur = layers_per_device + (1 if cur_dev < remainder else 0)

        self.ln_f = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.output_device = devices[-1]
        print(f"  Output layers -> {self.output_device}")

        total_params = sum(p.size for p in self.parameters())
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Parameters per device: ~{max(1, total_params // self.num_devices):,}")
        print(f"{'='*80}")

    def _maybe_to(self, x: pysml.Tensor, device: str) -> pysml.Tensor:
        return x if x.device == device else pysml.to_device(x, device)

    def _embed(self, x_tokens: pysml.Tensor) -> pysml.Tensor:
        # Keep lookup on embedding device
        if x_tokens.device != self.embedding_device:
            x_tokens = pysml.to_device(x_tokens, self.embedding_device)
        tok = F.embedding(x_tokens, self.token_embedding)  # relies on existing F.embedding
        return tok

    def forward(self, x_tokens: pysml.Tensor) -> pysml.Tensor:
        """
        Micro-batch pipeline pass (sequential in single-process, memory efficient).
        """
        bs, seqlen = x_tokens.shape

        # Precompute embeddings once per micro-batch on embedding device
        def make_pos_emb(n):
            pos = self.pos_embedding[:n]
            return pysml.reshape(pos, (1, n, self.d_model))

        outputs = []
        mb = max(1, bs // self.micro_batches)
        for start in range(0, bs, mb):
            end = min(bs, start + mb)
            xt = x_tokens[start:end]

            # Stage 0: embeddings + optional dropout (on emb device)
            tok = self._embed(xt)
            pos = make_pos_emb(xt.shape[1])
            h = tok + pos

            if self.training and self.dropout_p > 0:
                h = pysml.dropout(h, p=self.dropout_p)

            # Pass through blocks with minimal transfers
            cur_dev = self.embedding_device
            for block, dev in zip(self.blocks, self.block_devices):
                if cur_dev != dev:
                    h = pysml.to_device(h, dev)
                    cur_dev = dev
                h = block(h)

            # Final on output device
            if cur_dev != self.output_device:
                h = pysml.to_device(h, self.output_device)

            h = self.ln_f(h)
            h_flat = pysml.reshape(h, (h.shape[0] * h.shape[1], self.d_model))
            logits = self.lm_head(h_flat)
            logits = pysml.reshape(logits, (h.shape[0], h.shape[1], self.vocab_size))
            outputs.append(logits)

        # Concatenate micro-batch logits on output device
        if len(outputs) == 1:
            return outputs[0]
        return pysml.concatenate(outputs, axis=0)

    def get_memory_breakdown(self) -> dict:
        memory_map = {d: 0 for d in self.devices}
        memory_map[self.embedding_device] += self.token_embedding.size + self.pos_embedding.size
        for block, dev in zip(self.blocks, self.block_devices):
            memory_map[dev] += sum(p.size for p in block.parameters())
        memory_map[self.output_device] += sum(p.size for p in self.ln_f.parameters())
        memory_map[self.output_device] += sum(p.size for p in self.lm_head.parameters())
        return memory_map

    def print_memory_breakdown(self):
        mmap = self.get_memory_breakdown()
        total = sum(mmap.values()) or 1
        print(f"\n{'='*80}")
        print("Memory Distribution")
        print(f"{'='*80}")
        for dev, params in mmap.items():
            pct = (params / total) * 100.0
            print(f"  {dev}: {params:,} parameters ({pct:.1f}%)")
        print(f"  Total: {total:,} parameters")
        print(f"{'='*80}")
