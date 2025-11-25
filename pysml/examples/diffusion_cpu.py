"""Dedicated launcher for diffusion on cpu."""
from __future__ import annotations

from .parallel_utils import ExampleConfig
from . import diffusion


def main() -> None:
    diffusion.train_example(ExampleConfig(backend="cpu"))


if __name__ == "__main__":
    main()
