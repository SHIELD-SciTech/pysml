"""Dedicated launcher for transformer on xpu."""
from __future__ import annotations

from .parallel_utils import ExampleConfig
from . import transformer


def main() -> None:
    transformer.train_example(ExampleConfig(backend="xpu"))


if __name__ == "__main__":
    main()
