"""Shared helpers for tensor-parallel aware modules."""

from __future__ import annotations

from typing import Tuple

from ...tensor import Tensor


def reshape_for_linear(input: Tensor, in_features: int) -> tuple[Tensor, Tuple[int, ...], bool]:
    """Flatten ``input`` to 2D while remembering the original shape."""

    input_shape = input.shape
    single_sample = len(input_shape) == 1
    if single_sample:
        input = input.reshape((1, -1))
    if input.shape[-1] != in_features:
        raise ValueError(f"Input dim mismatch: expected {in_features}, got {input.shape[-1]}")

    if len(input.shape) > 2:
        batch_size = 1
        for dim in input.shape[:-1]:
            batch_size *= dim
        input = input.reshape((batch_size, in_features))
    return input, input_shape, single_sample


def restore_from_linear(output: Tensor, original_shape: Tuple[int, ...], out_features: int, single_sample: bool) -> Tensor:
    if len(original_shape) > 2:
        output_shape = list(original_shape[:-1]) + [out_features]
        output = output.reshape(tuple(output_shape))
    if single_sample:
        output = output.reshape((out_features,))
    return output


def maybe_all_gather(tensor: Tensor, gather: bool, group) -> Tensor:
    if not gather:
        return tensor
    from ...distributed.tensor_parallel import gather_from_tensor_parallel_region

    return gather_from_tensor_parallel_region(tensor, dim=-1, group=group)


def reduce_partial_output(tensor: Tensor, group) -> Tensor:
    from ...distributed.tensor_parallel import reduce_from_tensor_parallel_region

    return reduce_from_tensor_parallel_region(tensor, op="sum", group=group)
