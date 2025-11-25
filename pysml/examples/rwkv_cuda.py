"""Dedicated launcher for rwkv on cuda."""
from __future__ import annotations

from .parallel_utils import ExampleConfig
from . import rwkv


def main() -> None:
    rwkv.train_example(ExampleConfig(backend="cuda"))


if __name__ == "__main__":
    main()
