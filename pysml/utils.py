from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

def build_graph(tensor, nodes=None, edges=None):
        if nodes is None:
                nodes = set()
        if edges is None:
                edges = set()

        # Each tensor is a node
        nodes.add(tensor)

        # If it was created by an operation, add edges to parents
        if tensor._grad_fn:
                for parent_ref in (getattr(tensor._grad_fn, 'inputs', []) or []):
                        if parent_ref is None:
                                continue
                        parent = parent_ref()
                        if parent is None:
                                continue
                        edges.add((parent, tensor))
                        build_graph(parent, nodes, edges)

        return nodes, edges

def to_dot(root_tensor) -> str:
        nodes, edges = build_graph(root_tensor)

        def tensor_label(t):
                val = f"{t.data.tolist()}"
                return f"Tensor(value={val}, grad={None if t.grad is None else t.grad.data.tolist()})"

        lines = ["digraph ComputationGraph {", "rankdir=TB;"]

        for n in nodes:
                lines.append(f'"{id(n)}" [label="{tensor_label(n)}"];')

        for p, c in edges:
                lines.append(f'"{id(p)}" -> "{id(c)}";')

        lines.append("}")
        return "\n".join(lines)

@dataclass
class PipelineMetrics:
        schedule: str
        micro_batches: int
        num_stages: int
        bubble: float
        stage_latency: list[dict[str, float]]
        stage_memory: list[dict[str, float]]

def estimate_pipeline_bubble(num_stages: int, micro_batches: int, schedule: str) -> float:
        if micro_batches <= 0 or num_stages <= 0:
                return 0.0
        schedule = schedule.lower()
        if schedule == '1f1b':
                idle = max(num_stages - 1, 0)
                denom = micro_batches + num_stages - 1
                return idle / denom if denom else 0.0
        idle = (num_stages - 1) * 2
        denom = micro_batches + num_stages
        return idle / denom if denom else 0.0

def reduce_samples(samples: Iterable[float]) -> dict[str, float]:
        data = list(samples)
        if not data:
                return {"min": 0.0, "max": 0.0, "avg": 0.0}
        total = sum(data)
        return {"min": min(data), "max": max(data), "avg": total / len(data)}

def build_pipeline_metrics(
        *,
        schedule: str,
        micro_batches: int,
        stage_latencies: Iterable[dict[str, float]],
        stage_memory: Iterable[dict[str, float]],
        bubble: float,
) -> PipelineMetrics:
        latency = list(stage_latencies)
        memory = list(stage_memory)
        num_stages = max(len(latency), len(memory))
        while len(latency) < num_stages:
                latency.append({"min": 0.0, "max": 0.0, "avg": 0.0})
        while len(memory) < num_stages:
                memory.append({"min": 0.0, "max": 0.0, "avg": 0.0})
        return PipelineMetrics(
                schedule=schedule,
                micro_batches=micro_batches,
                num_stages=num_stages,
                bubble=bubble,
                stage_latency=latency,
                stage_memory=memory,
        )

__all__ = [
        'build_graph',
        'to_dot',
        'PipelineMetrics',
        'estimate_pipeline_bubble',
        'reduce_samples',
        'build_pipeline_metrics',
]
