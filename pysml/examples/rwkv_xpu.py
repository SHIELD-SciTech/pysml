"""Dedicated launcher for rwkv on xpu."""
from __future__ import annotations

from .parallel_utils import ExampleConfig
from . import rwkv


def main() -> None:
    rwkv.train_example(ExampleConfig(backend="xpu"))


if __name__ == "__main__":
    main()
