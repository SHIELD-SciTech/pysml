"""Benchmark helpers to compare single-device and pipeline runs.

The functions here intentionally favour readability over raw performance so you
can adapt them for your own experiments. ``compare_single_vs_pipeline`` produces
side-by-side statistics for CPU, CUDA, or Intel XPU launches – ideal when
validating that prototype pipeline schedules match the throughput you expect.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Dict, Iterable, List

from . import diffusion, rwkv, transformer
from .parallel_utils import ExampleConfig

Runner = Callable[..., dict]


@dataclass
class BenchmarkResult:
    example: str
    backend: str
    strategy: str
    steps: int
    duration_s: float
    throughput: float


RUNNERS: Dict[str, Runner] = {
    "rwkv": rwkv.train_example,
    "diffusion": diffusion.train_example,
    "transformer": transformer.train_example,
}


def benchmark(
    example: str,
    config: ExampleConfig,
    *,
    steps: int = 3,
    use_pipeline: bool = False,
) -> BenchmarkResult:
    """Run a single benchmark configuration and capture throughput stats."""
    runner = RUNNERS[example]
    start = time.perf_counter()
    runner(config, steps=steps, use_pipeline=use_pipeline)
    duration = time.perf_counter() - start
    return BenchmarkResult(
        example=example,
        backend=config.device(),
        strategy="pipeline" if use_pipeline else "single",
        steps=steps,
        duration_s=duration,
        throughput=steps / max(duration, 1e-6),
    )


def compare_single_vs_pipeline(
    example: str,
    backend: str,
    *,
    steps: int = 3,
    pipeline_parallel: int = 2,
    pipeline_chunks: int = 2,
) -> List[BenchmarkResult]:
    """Return both single-device and pipeline benchmark entries."""
    base = ExampleConfig(backend=backend)
    pipelined = ExampleConfig(
        backend=backend,
        pipeline_parallel=pipeline_parallel,
        pipeline_chunks=pipeline_chunks,
    )
    single = benchmark(example, base, steps=steps, use_pipeline=False)
    piped = benchmark(example, pipelined, steps=steps, use_pipeline=True)
    return [single, piped]


def summarize(results: Iterable[BenchmarkResult]) -> None:
    """Pretty-print benchmark results in a table-like format."""
    for result in results:
        print(
            f"{result.example:<12} {result.backend:<8} {result.strategy:<8} "
            f"steps={result.steps:<2d} time={result.duration_s:.3f}s "
            f"throughput={result.throughput:.2f} it/s"
        )


def main() -> None:
    all_results: list[BenchmarkResult] = []
    for example in RUNNERS:
        for backend in ("cpu", "cuda", "xpu"):
            all_results.extend(compare_single_vs_pipeline(example, backend))
    summarize(all_results)


if __name__ == "__main__":
    main()
