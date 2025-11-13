import unittest

import pytest

np = pytest.importorskip("numpy")

from pysml import distributed
from pysml.distributed import DistributedDataParallel
from pysml.examples.diffusion import TinyDiffusionUNet, generate_batch as diffusion_batch
from pysml.examples.rwkv import TinyRWKVModel, generate_batch as rwkv_batch
from pysml.examples.transformer import TinyTransformerClassifier, generate_batch as transformer_batch
from pysml.nn import Adam, AdamW, CrossEntropyLoss, MSELoss


class DistributedExampleParityTests(unittest.TestCase):
    def setUp(self) -> None:
        distributed.shutdown()
        np.random.seed(0)

    def tearDown(self) -> None:
        distributed.shutdown()

    def _compare_step(self, model_ctor, data_fn, criterion, optimizer_ctor):
        base_model = model_ctor()
        ddp_model = model_ctor()
        ddp_model.load_state_dict(base_model.state_dict())
        ddp = DistributedDataParallel(ddp_model)

        base_optimizer = optimizer_ctor(base_model.parameters(), lr=1e-3)
        ddp_optimizer = optimizer_ctor(ddp.parameters(), lr=1e-3)

        inputs, targets = data_fn()

        base_optimizer.zero_grad()
        ddp_optimizer.zero_grad()

        base_loss = criterion(base_model(inputs), targets)
        ddp_loss = criterion(ddp(inputs), targets)

        base_loss.backward()
        ddp_loss.backward()
        ddp.synchronize_gradients()

        for (name_a, param_a), (name_b, param_b) in zip(
            base_model.named_parameters(), ddp.module.named_parameters()
        ):
            if param_a.grad is None or param_b.grad is None:
                continue
            np.testing.assert_allclose(
                param_a.grad.data,
                param_b.grad.data,
                atol=1e-6,
                err_msg=f"Gradient mismatch for {name_a} vs {name_b}",
            )

    def test_transformer_example_matches_single_device(self):
        def model_ctor():
            return TinyTransformerClassifier(vocab_size=128, d_model=32, num_layers=1, num_heads=2, num_classes=4)

        def batch_fn():
            return transformer_batch(batch_size=2, seq_len=8, vocab_size=128, num_classes=4)

        self._compare_step(model_ctor, batch_fn, CrossEntropyLoss(), AdamW)

    def test_rwkv_example_matches_single_device(self):
        def model_ctor():
            return TinyRWKVModel(vocab_size=64, hidden_size=32, num_layers=2, num_classes=4)

        def batch_fn():
            return rwkv_batch(batch_size=2, seq_len=8, vocab_size=64, num_classes=4)

        self._compare_step(model_ctor, batch_fn, CrossEntropyLoss(), Adam)

    def test_diffusion_example_matches_single_device(self):
        def model_ctor():
            return TinyDiffusionUNet(in_channels=3, base_channels=16)

        def batch_fn():
            return diffusion_batch(batch_size=1, channels=3, height=16, width=16)

        self._compare_step(model_ctor, batch_fn, MSELoss(), AdamW)


if __name__ == "__main__":
    unittest.main()
