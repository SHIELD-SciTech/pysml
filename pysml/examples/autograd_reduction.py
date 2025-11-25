"""Autograd sanity checks for reduction-aware gradients.

Run directly to see gradients for axis-aware sums/means and
broadcasted products on a chosen device.
"""

from __future__ import annotations

import numpy as np

import pysml
from pysml import Tensor


def reduction_checks(device: str = "cpu") -> dict[str, np.ndarray]:
    results = {}

    summed = Tensor(np.arange(1, 7, dtype=np.float32).reshape(2, 3), device=device, requires_grad=True)
    sum_loss = pysml.sum(summed, axis=1)
    sum_loss.backward(Tensor(np.ones_like(sum_loss.data)))
    results["sum_grad"] = summed.grad.numpy()

    averaged = Tensor(np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32), device=device, requires_grad=True)
    mean_loss = pysml.mean(averaged, axis=0)
    mean_loss.backward(Tensor(np.ones_like(mean_loss.data)))
    results["mean_grad"] = averaged.grad.numpy()

    left = Tensor(np.ones((2, 3), dtype=np.float32), device=device, requires_grad=True)
    right = Tensor(np.array([[2.0, 3.0, 4.0]], dtype=np.float32), device=device, requires_grad=True)
    prod_loss = pysml.sum(pysml.multiply(left, right))
    prod_loss.backward()
    results["broadcast_left_grad"] = left.grad.numpy()
    results["broadcast_right_grad"] = right.grad.numpy()

    return results


def print_results(device: str) -> None:
    grads = reduction_checks(device)
    print(f"Gradients on {device}:")
    for name, array in grads.items():
        print(f"- {name}:\n{array}\n")


if __name__ == "__main__":
    for device in ("cpu", "cuda", "xpu"):
        available = getattr(getattr(pysml, device, object()), "is_available", lambda: device == "cpu")()
        if not available:
            continue
        print_results(device)
