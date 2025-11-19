"""Dedicated launcher for diffusion on xpu."""
from __future__ import annotations

from .parallel_utils import ExampleConfig
from . import diffusion


def main() -> None:
    diffusion.train_example(ExampleConfig(backend="xpu"))


if __name__ == "__main__":
    main()
