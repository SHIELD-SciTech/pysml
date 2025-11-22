"""Pipeline parallel execution helpers."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import pysml
from pysml import utils
from pysml.autograd import Function
from pysml.ddp.communication import primitives as dist_primitives
from pysml.tensor import Tensor

from .module import Module, ModuleList, Sequential


@dataclass
class _StageSpec:
    modules: Sequential
    device: Optional[str]
    rank: Optional[int]


class PipelineModule(Module):
    """Partition a sequential module graph into pipeline stages.

    Warning: Modules used in pipeline parallel schedules must be stateless
    with respect to their inputs. Avoid in-place operations (for example,
    ``+=`` or ``.add_()``) on tensors that are fed into a pipeline stage,
    because activation checkpointing will re-run the forward pass and expect
    unmodified inputs when computing gradients.
    """

    def __init__(
        self,
        modules: Sequence[Module] | ModuleList | Sequential,
        *,
        partitions: Optional[Sequence[int]] = None,
        num_stages: Optional[int] = None,
        strategy: str = "balanced",
        stage_devices: Optional[Sequence[Optional[str]]] = None,
        stage_ranks: Optional[Sequence[int]] = None,
        schedule: str = "gpipe",
        communication: str = "gather",
        chunks: int = 1,
        activation_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        flat = self._flatten_modules(modules)
        if not flat:
            raise ValueError("PipelineModule requires at least one module")

        counts = self._resolve_partitions(flat, partitions, num_stages, strategy)
        self._stages: list[_StageSpec] = []
        offset = 0
        for idx, count in enumerate(counts):
            stage_modules = Sequential(*flat[offset : offset + count])
            offset += count
            device = None if stage_devices is None else stage_devices[idx]
            rank = None if stage_ranks is None else stage_ranks[idx]
            self._stages.append(_StageSpec(stage_modules, device, rank))
            self.add_module(f"stage_{idx}", stage_modules)

        self.schedule = schedule.lower()
        if self.schedule not in {"gpipe", "1f1b"}:
            raise ValueError("schedule must be either 'gpipe' or '1f1b'")

        self.communication = communication
        self.chunks = max(1, chunks)
        self.activation_checkpoint = activation_checkpoint
        self._latest_metrics: Optional[utils.PipelineMetrics] = None
        self._stage_messages: list[list[Any]] = [[] for _ in self._stages]

    # ------------------------------------------------------------------
    # Partition helpers
    # ------------------------------------------------------------------
    def _flatten_modules(
        self, modules: Sequence[Module] | ModuleList | Sequential
    ) -> list[Module]:
        if isinstance(modules, Sequential):
            return list(modules._modules.values())
        if isinstance(modules, ModuleList):
            return list(modules._modules.values())
        if isinstance(modules, Module):
            return [modules]
        return list(modules)

    def _resolve_partitions(
        self,
        modules: Sequence[Module],
        partitions: Optional[Sequence[int]],
        num_stages: Optional[int],
        strategy: str,
    ) -> list[int]:
        if partitions is not None:
            counts = list(partitions)
            if sum(counts) != len(modules):
                raise ValueError("partition sizes must sum to the number of modules")
            if any(size <= 0 for size in counts):
                raise ValueError("partition sizes must be positive")
            return counts

        if num_stages is None:
            num_stages = 1
        num_stages = max(1, min(num_stages, len(modules)))
        if strategy == "parameter":
            counts = self._balance_by_params(modules, num_stages)
        else:
            counts = self._balance_by_depth(len(modules), num_stages)
        return counts

    def _balance_by_depth(self, total: int, stages: int) -> list[int]:
        base, remainder = divmod(total, stages)
        counts = []
        cursor = 0
        for idx in range(stages):
            size = base + (1 if idx < remainder else 0)
            size = max(1, size)
            counts.append(size)
            cursor += size
        counts[-1] += total - sum(counts)
        return counts

    def _balance_by_params(self, modules: Sequence[Module], stages: int) -> list[int]:
        param_counts = []
        for module in modules:
            total = 0
            for param in module.parameters():
                total += math.prod(param.shape)
            param_counts.append(total or 1)
        target = sum(param_counts) / stages
        counts: list[int] = []
        acc = 0
        bucket = 0
        for idx, count in enumerate(param_counts):
            acc += count
            bucket += 1
            if (acc >= target and len(counts) < stages - 1) or idx == len(param_counts) - 1:
                counts.append(bucket)
                bucket = 0
                acc = 0
        if bucket:
            counts[-1] += bucket
        while len(counts) < stages:
            counts.append(1)
        counts[-1] += len(modules) - sum(counts)
        return counts

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        stage_latencies: list[list[float]] = [[] for _ in self._stages]
        stage_memory: list[list[int]] = [[] for _ in self._stages]
        stage_messages: list[list[Any]] = [[] for _ in self._stages]
        micro_batches = self._chunk_arguments(args, kwargs)
        if self.schedule == "gpipe":
            outputs = self._run_gpipe(
                micro_batches, stage_latencies, stage_memory, stage_messages
            )
        else:
            outputs = self._run_1f1b(
                micro_batches, stage_latencies, stage_memory, stage_messages
            )
        merged = self._merge_outputs(outputs)
        self._latest_metrics = utils.build_pipeline_metrics(
            schedule=self.schedule,
            micro_batches=len(micro_batches),
            stage_latencies=[utils.reduce_samples(samples) for samples in stage_latencies],
            stage_memory=[utils.reduce_samples(samples) for samples in stage_memory],
            bubble=utils.estimate_pipeline_bubble(
                len(self._stages), len(micro_batches), self.schedule
            ),
        )
        return merged

    def profile(self) -> Optional[utils.PipelineMetrics]:
        return self._latest_metrics

    def _run_gpipe(
        self,
        micro_batches: list[tuple[tuple[Any, ...], dict[str, Any]]],
        stage_latencies: list[list[float]],
        stage_memory: list[list[int]],
        stage_messages: list[list[Any]],
    ):
        outputs = []
        for args, kwargs in micro_batches:
            payload = (args, kwargs)
            for idx, stage in enumerate(self._stages):
                payload = self._execute_stage(
                    idx, stage, payload, stage_latencies, stage_memory, stage_messages
                )
            outputs.append(self._denormalize(payload))
        return outputs

    def _run_1f1b(
        self,
        micro_batches: list[tuple[tuple[Any, ...], dict[str, Any]]],
        stage_latencies: list[list[float]],
        stage_memory: list[list[int]],
        stage_messages: list[list[Any]],
    ):
        in_flight: list[Optional[tuple[tuple[Any, ...], dict[str, Any]]]] = [None] * (
            len(self._stages)
        )
        outputs = []
        total_steps = len(micro_batches) + len(self._stages) - 1
        for clock in range(total_steps):
            for stage_idx in reversed(range(len(self._stages))):
                payload = in_flight[stage_idx]
                if payload is None:
                    continue
                payload = self._execute_stage(
                    stage_idx,
                    self._stages[stage_idx],
                    payload,
                    stage_latencies,
                    stage_memory,
                    stage_messages,
                )
                if stage_idx == len(self._stages) - 1:
                    outputs.append(self._denormalize(payload))
                else:
                    in_flight[stage_idx + 1] = payload
                in_flight[stage_idx] = None
            if clock < len(micro_batches):
                in_flight[0] = micro_batches[clock]
        return outputs

    def _denormalize(self, payload: tuple[tuple[Any, ...], dict[str, Any]]) -> Any:
        args, kwargs = payload
        if kwargs and not args:
            return kwargs
        if len(args) == 1 and not kwargs:
            return args[0]
        return args

    def _execute_stage(
        self,
        stage_idx: int,
        spec: _StageSpec,
        payload: tuple[tuple[Any, ...], dict[str, Any]],
        stage_latencies: list[list[float]],
        stage_memory: list[list[int]],
        stage_messages: list[list[Any]],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        args, kwargs = payload
        moved_args = self._move_to_device(args, spec.device)
        moved_kwargs = self._move_to_device(kwargs, spec.device)
        start = time.perf_counter()
        output = spec.modules(*moved_args, **moved_kwargs)
        latency = time.perf_counter() - start
        normalized = self._normalize_output(output)
        stage_latencies[stage_idx].append(latency)
        stage_memory[stage_idx].append(self._estimate_bytes(normalized))
        self._communicate(stage_idx, normalized, stage_messages)
        if self.activation_checkpoint:
            normalized = self._checkpoint_payload(stage_idx, normalized, spec)
        return normalized

    def _communicate(
        self,
        stage_idx: int,
        payload: tuple[tuple[Any, ...], dict[str, Any]],
        stage_messages: list[list[Any]],
    ) -> None:
        if stage_idx >= len(self._stages) - 1:
            return
        tensors = self._collect_tensors(payload)
        if not tensors:
            return
        transmitted: list[Any] = []
        if self.communication == "gather":
            for tensor in tensors:
                transmitted.append(dist_primitives.gather(tensor, dst=0))
        else:
            for tensor in tensors:
                transmitted.append(dist_primitives.send(tensor, dst=0))
        stage_messages[stage_idx] = transmitted

    def _collect_tensors(self, payload: tuple[tuple[Any, ...], dict[str, Any]]) -> list[Tensor]:
        tensors: list[Tensor] = []

        seen: set[int] = set()

        def _collect(obj: Any) -> None:
            obj_id = id(obj)
            if obj_id in seen:
                return
            seen.add(obj_id)
            if isinstance(obj, Tensor):
                tensors.append(obj)
            elif isinstance(obj, tuple) or isinstance(obj, list):
                for item in obj:
                    _collect(item)
            elif isinstance(obj, dict):
                for value in obj.values():
                    _collect(value)

        for arg in payload[0]:
            _collect(arg)
        for value in payload[1].values():
            _collect(value)
        return tensors

    def _move_to_device(self, obj: Any, device: Optional[str], _seen: Optional[dict[int, Any]] = None) -> Any:
        if device is None:
            return obj
        if _seen is None:
            _seen = {}
        obj_id = id(obj)
        if obj_id in _seen:
            return _seen[obj_id]
        if isinstance(obj, Tensor):
            moved = obj.to(device)
            _seen[obj_id] = moved
            return moved
        if isinstance(obj, tuple):
            moved_tuple = tuple(self._move_to_device(item, device, _seen) for item in obj)
            _seen[obj_id] = moved_tuple
            return moved_tuple
        if isinstance(obj, list):
            moved_list = [self._move_to_device(item, device, _seen) for item in obj]
            _seen[obj_id] = moved_list
            return moved_list
        if isinstance(obj, dict):
            moved_dict = {k: self._move_to_device(v, device, _seen) for k, v in obj.items()}
            _seen[obj_id] = moved_dict
            return moved_dict
        return obj

    def _normalize_output(
        self, output: Any
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if isinstance(output, tuple):
            return output, {}
        if isinstance(output, dict):
            return tuple(), output
        return (output,), {}

    def _checkpoint_payload(
        self,
        stage_idx: int,
        payload: tuple[tuple[Any, ...], dict[str, Any]],
        spec: _StageSpec,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        args, kwargs = payload

        saved_args = args
        saved_kwargs = kwargs

        stage_inputs: list[Optional[Tensor]] = []

        seen_inputs: set[int] = set()

        def _collect_inputs(obj: Any) -> None:
            obj_id = id(obj)
            if obj_id in seen_inputs:
                return
            seen_inputs.add(obj_id)
            if isinstance(obj, Tensor):
                stage_inputs.append(obj if obj._requires_grad else None)
            elif isinstance(obj, (tuple, list)):
                for item in obj:
                    _collect_inputs(item)
            elif isinstance(obj, dict):
                for value in obj.values():
                    _collect_inputs(value)

        _collect_inputs(args)
        _collect_inputs(kwargs)

        def _make_accessor(path: list[tuple[str, Any]]):
            path_tokens = tuple(path)

            def _access(payload_tree: tuple[tuple[Any, ...], dict[str, Any]], path_snapshot=path_tokens):
                payload_args, payload_kwargs = payload_tree
                token_type, token_key = path_snapshot[0]
                node: Any = payload_args if token_type == "args" else payload_kwargs
                node = node[token_key]
                for token_type, token_key in path_snapshot[1:]:
                    if token_type in {"tuple", "list"}:
                        node = node[token_key]
                    else:
                        node = node[token_key]
                return node

            return _access

        def _wrap_tensor(tensor: Tensor, path: list[tuple[str, Any]]):
            accessor = _make_accessor(path)

            def _checkpoint_backward(grad_output, *input_refs, accessor=accessor, saved_args=saved_args, saved_kwargs=saved_kwargs, **metadata):
                inputs = [ref() if ref is not None else None for ref in input_refs]
                for inp in inputs:
                    if inp is not None:
                        inp._grad = None

                recomputed = spec.modules(*saved_args, **saved_kwargs)
                recomputed_payload = self._normalize_output(recomputed)
                target = metadata["accessor"](recomputed_payload) if "accessor" in metadata else accessor(recomputed_payload)

                if isinstance(target, Tensor) and grad_output is not None:
                    target.backward(grad_output, retain_graph=False)

                grads = []
                for inp in inputs:
                    if inp is None:
                        grads.append(None)
                    else:
                        grads.append((inp, inp._grad))
                        inp._grad = None
                return grads

            grad_fn = Function(
                _checkpoint_backward,
                stage_inputs,
                metadata={"accessor": accessor},
            )

            wrapped = Tensor.__new__(Tensor)
            wrapped._backend = tensor._backend
            wrapped._dtype = tensor._dtype
            wrapped._requires_grad = tensor._requires_grad
            wrapped._grad = None
            wrapped._grad_fn = grad_fn if tensor._requires_grad else None
            wrapped.device = tensor.device
            wrapped.active_device = tensor.active_device
            wrapped._version = getattr(tensor, "_version", 0)
            wrapped.data = tensor.data
            return wrapped

        seen_wrap: dict[int, Any] = {}

        def _wrap(obj: Any, path: list[tuple[str, Any]]):
            obj_id = id(obj)
            if obj_id in seen_wrap:
                return seen_wrap[obj_id]
            if isinstance(obj, Tensor):
                wrapped_tensor = _wrap_tensor(obj, path)
                seen_wrap[obj_id] = wrapped_tensor
                return wrapped_tensor
            if isinstance(obj, tuple):
                wrapped_tuple = tuple(
                    _wrap(item, path + [("tuple", idx)]) for idx, item in enumerate(obj)
                )
                seen_wrap[obj_id] = wrapped_tuple
                return wrapped_tuple
            if isinstance(obj, list):
                wrapped_list = [
                    _wrap(item, path + [("list", idx)]) for idx, item in enumerate(obj)
                ]
                seen_wrap[obj_id] = wrapped_list
                return wrapped_list
            if isinstance(obj, dict):
                wrapped_dict = {
                    key: _wrap(value, path + [("dict", key)])
                    for key, value in obj.items()
                }
                seen_wrap[obj_id] = wrapped_dict
                return wrapped_dict
            return obj

        wrapped_args = tuple(_wrap(arg, [("args", idx)]) for idx, arg in enumerate(args))
        wrapped_kwargs = {k: _wrap(v, [("kwargs", k)]) for k, v in kwargs.items()}

        return wrapped_args, wrapped_kwargs

    def _chunk_arguments(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
        batch = self._discover_batch_size(args, kwargs)
        if batch is None or self.chunks <= 1 or batch <= 1:
            return [(args, kwargs)]
        chunks = min(self.chunks, batch)
        sizes = self._chunk_sizes(batch, chunks)
        tensor_cache: dict[int, list[Tensor]] = {}
        micro_batches = []
        for idx in range(len(sizes)):
            chunk_args = self._slice_structure(args, sizes, idx, tensor_cache)
            chunk_kwargs = self._slice_structure(kwargs, sizes, idx, tensor_cache)
            micro_batches.append((chunk_args, chunk_kwargs))
        return micro_batches

    def _discover_batch_size(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Optional[int]:
        for obj in list(args) + list(kwargs.values()):
            size = self._extract_batch(obj)
            if size is not None:
                return size
        return None

    def _extract_batch(self, obj: Any) -> Optional[int]:
        if isinstance(obj, Tensor) and obj.ndim > 0:
            return obj.shape[0]
        if isinstance(obj, (list, tuple)):
            for item in obj:
                size = self._extract_batch(item)
                if size is not None:
                    return size
        if isinstance(obj, dict):
            for value in obj.values():
                size = self._extract_batch(value)
                if size is not None:
                    return size
        return None

    def _chunk_sizes(self, batch: int, chunks: int) -> list[int]:
        base, remainder = divmod(batch, chunks)
        sizes = []
        for idx in range(chunks):
            size = base + (1 if idx < remainder else 0)
            if size > 0:
                sizes.append(size)
        if sum(sizes) != batch:
            sizes[-1] += batch - sum(sizes)
        return sizes

    def _slice_structure(
        self,
        obj: Any,
        sizes: Sequence[int],
        idx: int,
        cache: dict[int, list[Tensor]],
    ) -> Any:
        if isinstance(obj, Tensor) and obj.ndim > 0:
            splits = cache.get(id(obj))
            if splits is None:
                splits = pysml.split(obj, sizes, dim=0)
                cache[id(obj)] = splits
            return splits[idx]
        if isinstance(obj, tuple):
            return tuple(self._slice_structure(item, sizes, idx, cache) for item in obj)
        if isinstance(obj, list):
            return [self._slice_structure(item, sizes, idx, cache) for item in obj]
        if isinstance(obj, dict):
            return {k: self._slice_structure(v, sizes, idx, cache) for k, v in obj.items()}
        return obj

    def _merge_outputs(self, outputs: list[Any]) -> Any:
        if not outputs:
            return None
        template = outputs[0]
        if isinstance(template, Tensor):
            return pysml.concatenate(outputs, axis=0)
        if isinstance(template, tuple):
            transposed = list(zip(*outputs))
            return tuple(self._merge_outputs(list(chunk)) for chunk in transposed)
        if isinstance(template, list):
            transposed = list(zip(*outputs))
            return [self._merge_outputs(list(chunk)) for chunk in transposed]
        if isinstance(template, dict):
            merged = {}
            for key in template.keys():
                merged[key] = self._merge_outputs([output[key] for output in outputs])
            return merged
        return outputs[-1]

    def _estimate_bytes(self, payload: tuple[tuple[Any, ...], dict[str, Any]]) -> int:
        total = 0
        for arg in payload[0]:
            total += self._object_nbytes(arg)
        for value in payload[1].values():
            total += self._object_nbytes(value)
        return total

    def _object_nbytes(self, obj: Any) -> int:
        if isinstance(obj, Tensor):
            data = getattr(obj, "data", None)
            return int(getattr(data, "nbytes", 0) or 0)
        if isinstance(obj, tuple):
            return sum(self._object_nbytes(item) for item in obj)
        if isinstance(obj, list):
            return sum(self._object_nbytes(item) for item in obj)
        if isinstance(obj, dict):
            return sum(self._object_nbytes(item) for item in obj.values())
        return 0


__all__ = ["PipelineModule"]
