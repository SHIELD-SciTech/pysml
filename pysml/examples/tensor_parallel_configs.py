"""Reference tensor-parallel friendly module configurations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .. import engine
from ..tensor import Tensor
from ..nn import LayerNorm, Module, SiLU
from ..nn.parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    TensorParallelMultiheadAttention,
)
from ..distributed import tensor_parallel as tp


class GPTTensorParallelBlock(Module):
    """GPT-style Transformer block highlighting tensor parallel sharding."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp_hidden = mlp_ratio * hidden_size
        self.attn_norm = LayerNorm(hidden_size)
        self.attn = TensorParallelMultiheadAttention(hidden_size, num_heads)
        self.mlp_norm = LayerNorm(hidden_size)
        self.mlp_fc = ColumnParallelLinear(hidden_size, self.mlp_hidden, gather_output=False)
        self.activation = SiLU()
        self.mlp_proj = RowParallelLinear(self.mlp_hidden, hidden_size, input_is_parallel=True)

    def forward(self, hidden: Tensor) -> Tensor:
        attn_input = self.attn_norm(hidden)
        attn_out = self.attn(attn_input)
        hidden = engine.add(hidden, attn_out)
        mlp_input = self.mlp_norm(hidden)
        mlp_out = self.mlp_fc(mlp_input)
        mlp_out = self.activation(mlp_out)
        mlp_out = self.mlp_proj(mlp_out)
        return engine.add(hidden, mlp_out)


class RWKVTimeMixTensorParallel(Module):
    """Tensor-parallel reference inspired by RWKV time-mix layers."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = LayerNorm(hidden_size)
        self.key = ColumnParallelLinear(hidden_size, hidden_size, gather_output=False)
        self.value = ColumnParallelLinear(hidden_size, hidden_size, gather_output=False)
        self.receptance = ColumnParallelLinear(hidden_size, hidden_size, gather_output=False)
        self.mix = RowParallelLinear(hidden_size, hidden_size, input_is_parallel=True)
        self.activation = SiLU()

    def forward(self, hidden: Tensor) -> Tensor:
        h = self.norm(hidden)
        k = self.key(h)
        v = self.value(h)
        r = engine.sigmoid(self.receptance(h))
        mix = engine.multiply(k, v)
        gated = engine.multiply(r, mix)
        mixed = self.mix(gated)
        return engine.add(hidden, self.activation(mixed))


class DiffusionUNetBottleneck(Module):
    """Simple tensor-parallel bottleneck used by diffusion U-Net examples."""

    def __init__(self, hidden_size: int, expansion: int = 2) -> None:
        super().__init__()
        mid = hidden_size * expansion
        self.norm1 = LayerNorm(hidden_size)
        self.up = ColumnParallelLinear(hidden_size, mid, gather_output=False)
        self.act = SiLU()
        self.norm2 = LayerNorm(mid)
        self.down = RowParallelLinear(mid, hidden_size, input_is_parallel=True)

    def forward(self, hidden: Tensor) -> Tensor:
        h = self.norm1(hidden)
        h = self.up(h)
        h = self.act(h)
        h = self.norm2(h)
        h = self.down(h)
        return engine.add(hidden, h)


@dataclass
class ReferenceConfig:
    description: str
    block: Module
    tensor_parallel_world_size: int


def build_reference_configs(hidden_size: int = 512, num_heads: int = 8) -> Dict[str, ReferenceConfig]:
    group = tp.get_tensor_parallel_group()
    return {
        "gpt": ReferenceConfig(
            description="GPT-style block with column/row parallel MLP",
            block=GPTTensorParallelBlock(hidden_size, num_heads),
            tensor_parallel_world_size=group.size,
        ),
        "rwkv": ReferenceConfig(
            description="RWKV time-mix with shared tensor parallel shards",
            block=RWKVTimeMixTensorParallel(hidden_size),
            tensor_parallel_world_size=group.size,
        ),
        "diffusion_unet": ReferenceConfig(
            description="Diffusion U-Net bottleneck built from partitioned linears",
            block=DiffusionUNetBottleneck(hidden_size),
            tensor_parallel_world_size=group.size,
        ),
    }


REFERENCE_CONFIGS = build_reference_configs()
