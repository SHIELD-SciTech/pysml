"""Dedicated launcher for transformer on cpu."""
from __future__ import annotations

from .parallel_utils import ExampleConfig
from . import transformer


def main() -> None:
    transformer.train_example(ExampleConfig(backend="cpu"))


if __name__ == "__main__":
    main()
