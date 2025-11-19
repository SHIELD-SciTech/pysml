"""Partitioned linear layers supporting tensor parallel sharding."""

from __future__ import annotations

import math
from typing import Optional

from ..module import Module, Parameter
from ... import engine
from ...tensor import Tensor
from . import utils


class ColumnParallelLinear(Module):
    """Shard the output dimension and optionally gather the partial results."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        gather_output: bool = True,
        group: Optional[object] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.group = group
        self.local_out_features = out_features

        self.weight = Parameter(self._init_weight(self.local_out_features, in_features))
        if bias:
            self.bias = Parameter(self._init_bias(self.local_out_features))
        else:
            self.bias = None

    def _init_weight(self, rows: int, cols: int) -> Tensor:
        from ...tensor import Tensor as _Tensor
        import numpy as np

        limit = math.sqrt(1.0 / cols)
        data = np.random.uniform(-limit, limit, (rows, cols)).astype(np.float32)
        return _Tensor(data, requires_grad=True)

    def _init_bias(self, size: int) -> Tensor:
        from ...tensor import Tensor as _Tensor
        import numpy as np

        data = np.zeros((size,), dtype=np.float32)
        return _Tensor(data, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        x, original_shape, single_sample = utils.reshape_for_linear(x, self.in_features)
        weight_t = self.weight.data.T()
        output = engine.matmul(x, weight_t)
        if self.bias is not None:
            output = engine.add(output, self.bias.data)
        output = utils.maybe_all_gather(output, self.gather_output, self.group)

        final_out_features = self.out_features if self.gather_output else self.local_out_features
        output = utils.restore_from_linear(output, original_shape, final_out_features, single_sample)
        return output


class RowParallelLinear(Module):
    """Shard the input dimension and reduce the partial projections."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        input_is_parallel: bool = False,
        group: Optional[object] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group = group
        self.input_is_parallel = input_is_parallel

        self.local_in_features = in_features
        self.effective_in_features = in_features

        self.weight = Parameter(self._init_weight(out_features, self.local_in_features))
        if bias:
            self.bias = Parameter(self._init_bias(out_features))
            pass
        else:
            self.bias = None

    def _init_weight(self, rows: int, cols: int) -> Tensor:
        from ...tensor import Tensor as _Tensor
        import numpy as np

        limit = math.sqrt(1.0 / cols)
        data = np.random.uniform(-limit, limit, (rows, cols)).astype(np.float32)
        return _Tensor(data, requires_grad=True)

    def _init_bias(self, size: int) -> Tensor:
        from ...tensor import Tensor as _Tensor
        import numpy as np

        data = np.zeros((size,), dtype=np.float32)
        return _Tensor(data, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        x, original_shape, single_sample = utils.reshape_for_linear(x, self.effective_in_features)
        weight_t = self.weight.data.T()
        output = engine.matmul(x, weight_t)
        if self.bias is not None:
            output = engine.add(output, self.bias.data)
        output = utils.restore_from_linear(output, original_shape, self.out_features, single_sample)
        return output
