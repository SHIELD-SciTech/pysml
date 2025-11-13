"""Distributed logging helpers for rank-aware metrics reporting."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

from ..tensor import Tensor
from . import collectives, process_group


def _tensor_to_list(tensor: Tensor) -> List[float]:
    data = getattr(tensor, "data", tensor)
    if hasattr(data, "tolist"):
        values = data.tolist()
    elif hasattr(data, "flatten"):
        values = list(data.flatten())
    else:  # Fallback to numpy for unknown containers
        try:
            import numpy as np

            values = np.asarray(data).tolist()
        except Exception:
            values = [float(data)]
    if isinstance(values, (int, float)):
        values = [float(values)]
    return [float(v) for v in values]


@dataclass
class AggregatedMetrics:
    """Container describing aggregated metrics across ranks."""

    samples: int
    total_loss: float
    total_time: float
    steps: int

    @property
    def average_loss(self) -> float:
        return 0.0 if self.steps == 0 else self.total_loss / max(self.steps, 1)

    @property
    def throughput(self) -> float:
        return 0.0 if self.total_time <= 0 else self.samples / self.total_time

    def as_dict(self) -> Dict[str, float]:
        return {
            "samples": float(self.samples),
            "total_loss": float(self.total_loss),
            "total_time": float(self.total_time),
            "steps": float(self.steps),
            "average_loss": float(self.average_loss),
            "throughput": float(self.throughput),
        }


class DistributedMonitor:
    """Track training progress across ranks with aggregated metrics."""

    def __init__(
        self,
        *,
        total_steps: Optional[int] = None,
        log_every: int = 10,
        writer: Optional[Callable[[str], None]] = None,
    ) -> None:
        process_group.lazy_init_from_env()
        self._communicator = process_group.get_backend()
        self.rank = self._communicator.rank
        self.world_size = max(self._communicator.world_size, 1)
        self.total_steps = total_steps
        self.log_every = max(1, int(log_every))
        self._writer = writer or (lambda msg: print(msg, flush=True))
        self._callbacks: List[Callable[[AggregatedMetrics], None]] = []
        self._reset_locals()

    def _reset_locals(self) -> None:
        self._local_samples = 0
        self._local_loss = 0.0
        self._local_time = 0.0
        self._local_steps = 0
        self._last_tick = time.perf_counter()

    def register_callback(self, callback: Callable[[AggregatedMetrics], None]) -> None:
        """Register a callback executed whenever aggregated metrics are reported."""

        if not callable(callback):
            raise TypeError("callback must be callable")
        self._callbacks.append(callback)

    @contextlib.contextmanager
    def track_batch(self, batch_size: int) -> Iterable[None]:
        """Context manager measuring elapsed time for a batch."""

        start = time.perf_counter()
        yield
        duration = max(time.perf_counter() - start, 1e-9)
        self._local_samples += int(batch_size)
        self._local_time += duration
        self._local_steps += 1

    def update(self, *, batch_size: int, loss: float, duration: Optional[float] = None) -> None:
        """Record a training step."""

        self._local_samples += int(batch_size)
        self._local_loss += float(loss)
        if duration is None:
            now = time.perf_counter()
            duration = max(now - self._last_tick, 1e-9)
            self._last_tick = now
        self._local_time += float(duration)
        self._local_steps += 1

        if self._local_steps % self.log_every == 0:
            self.report()

    def report(self, *, force: bool = False) -> AggregatedMetrics:
        """Aggregate metrics across ranks and optionally emit a progress line."""

        if self._local_steps == 0 and not force:
            return AggregatedMetrics(0, 0.0, 0.0, 0)

        payload = Tensor([self._local_samples, self._local_loss, self._local_time, self._local_steps], device="cpu")
        reduced = collectives.all_reduce(payload, op="sum")
        values = _tensor_to_list(reduced)
        metrics = AggregatedMetrics(
            samples=int(values[0]),
            total_loss=float(values[1]),
            total_time=float(values[2]),
            steps=int(values[3]),
        )
        if self.rank == 0:
            message = self._format_message(metrics)
            self._writer(message)
        for callback in self._callbacks:
            try:
                callback(metrics)
            except Exception:
                continue
        self._reset_locals()
        return metrics

    def _format_message(self, metrics: AggregatedMetrics) -> str:
        if self.total_steps is None:
            prefix = f"steps={metrics.steps}"
        else:
            percent = 0.0 if self.total_steps == 0 else (metrics.steps / self.total_steps) * 100.0
            prefix = f"steps={metrics.steps}/{self.total_steps} ({percent:4.1f}%)"
        return (
            f"[{prefix}] loss={metrics.average_loss:.4f} "
            f"throughput={metrics.throughput:.2f} samples/s world_size={self.world_size}"
        )

    def flush(self) -> None:
        """Force a metrics report regardless of the current step count."""

        self.report(force=True)


__all__ = ["DistributedMonitor", "AggregatedMetrics"]
