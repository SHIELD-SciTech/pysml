from __future__ import annotations

import unittest

import pytest

pytest.importorskip("numpy")

import pysml
from pysml.examples import diffusion, rwkv, transformer
from pysml.examples.parallel_utils import ExampleConfig


class ExampleDeterminismTests(unittest.TestCase):
    def assert_allclose(self, first, second, atol: float = 1e-5) -> None:
        diff = pysml.subtract(first, second)
        max_diff = float(pysml.max(pysml.abs(diff)).item())
        self.assertLessEqual(max_diff, atol)

    def test_rwkv_matches_across_backends(self) -> None:
        if not pysml.cuda.is_available():
            self.skipTest("CUDA backend not available for deterministic RWKV comparison")
        cpu_logits = rwkv.deterministic_logits(ExampleConfig(backend="cpu"), seed=7)
        cuda_logits = rwkv.deterministic_logits(ExampleConfig(backend="cuda"), seed=7)
        self.assert_allclose(cpu_logits, cuda_logits)

    def test_transformer_matches_across_backends(self) -> None:
        cpu_logits = transformer.deterministic_logits(
            ExampleConfig(backend="cpu"), seed=11
        )
        if not pysml.xpu.is_available():
            self.skipTest("XPU backend not available for deterministic Transformer comparison")

        xpu_logits = transformer.deterministic_logits(ExampleConfig(backend="xpu"), seed=11)
        self.assert_allclose(cpu_logits, xpu_logits)

    def test_diffusion_matches_across_backends(self) -> None:
        if not pysml.cuda.is_available():
            self.skipTest("CUDA backend not available for deterministic diffusion comparison")
        cpu_preds = diffusion.deterministic_prediction(
            ExampleConfig(backend="cpu"), seed=3
        )
        cuda_preds = diffusion.deterministic_prediction(ExampleConfig(backend="cuda"), seed=3)
        self.assert_allclose(cpu_preds, cuda_preds)


if __name__ == "__main__":
    unittest.main()
