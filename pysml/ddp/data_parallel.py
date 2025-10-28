"""
Data Parallel Training — optimized
- Zero-copy batch slicing where possible
- Single graph build per split; in-place grad scaling (AllReduce sim)
- Optional custom loss_fn and micro-batching
"""

import pysml
from typing import List, Optional, Callable, Tuple, Union
from ..nn import Module
from .utils import split_batch

ArrayLike = Union[pysml.Tensor, "np.ndarray"]  # type: ignore


class DataParallelModel:
    """
    Replicates the model across devices logically (single process, simulated DDP)
    and splits the batch. This implementation minimizes host<->device churn and
    avoids unnecessary allocations.

    Args:
        model: Module on primary device
        devices: list of device strings
        loss_fn: callable(pred, target)->Tensor (defaults to pysml.mse_loss)
        micro_batches: int micro-batches per device split (reduces peak mem)
    """

    def __init__(
        self,
        model: Module,
        devices: List[str],
        loss_fn: Optional[Callable[[pysml.Tensor, pysml.Tensor], pysml.Tensor]] = None,
        micro_batches: int = 1,
    ):
        self.base_model = model
        self.devices = devices
        self.num_devices = len(devices)
        self.primary_device = devices[0]
        self.loss_fn = loss_fn or pysml.mse_loss
        self.micro_batches = max(1, int(micro_batches))

    def _to_device_tensor(self, x: ArrayLike, device: str, requires_grad=False) -> pysml.Tensor:
        if isinstance(x, pysml.Tensor):
            return x if x.device == device else pysml.to_device(x, device)
        t = pysml.Tensor(x, requires_grad=requires_grad)
        return pysml.to_device(t, device)

    def _iterate_micro_batches(
        self, X_split: ArrayLike, y_split: ArrayLike
    ) -> Tuple[pysml.Tensor, pysml.Tensor, int]:
        """Yield micro-batches (views) to reduce activation memory."""
        bs = X_split.shape[0]
        if self.micro_batches <= 1 or bs <= 1:
            yield X_split, y_split, bs
            return
        size = max(1, bs // self.micro_batches)
        for start in range(0, bs, size):
            end = min(bs, start + size)
            yield X_split[start:end], y_split[start:end], end - start

    def forward_and_backward(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        1) zero_grad once
        2) for each device split:
            - move split to device
            - forward -> loss -> backward (accumulating grads)
        3) divide grads by num_devices (AllReduce average)

        Returns mean loss across device splits.
        """
        bsz = X.shape[0]
        split_size = max(1, bsz // self.num_devices)

        # zero once
        self.base_model.zero_grad()

        losses = []
        # Simulate parallel: process sequentially on primary device (API stable)
        for i in range(self.num_devices):
            start = i * split_size
            end = bsz if i == self.num_devices - 1 else (i + 1) * split_size
            if start >= end:
                continue

            X_split = X[start:end]
            y_split = y[start:end]

            # micro-batch inside split
            dev = self.primary_device
            split_loss_sum = 0.0
            count = 0

            for Xm, ym, mbs in self._iterate_micro_batches(X_split, y_split):
                Xt = self._to_device_tensor(Xm, dev, requires_grad=False)
                yt = self._to_device_tensor(ym, dev, requires_grad=False)

                out = self.base_model(Xt)
                loss = self.loss_fn(out, yt)
                loss.backward()
                split_loss_sum += float(loss.item())
                count += 1

            losses.append(split_loss_sum / max(1, count))

        # AllReduce average (in-place scaling)
        for p in self.base_model.parameters():
            if p.grad is not None:
                p.grad.data /= self.num_devices

        return sum(losses) / max(1, len(losses))

    def __call__(self, X: ArrayLike, y: ArrayLike) -> float:
        return self.forward_and_backward(X, y)


class DistributedDataParallel:
    """
    PyTorch-like façade. Optimized for minimal overhead; actual parallelism is
    orchestrated outside (multi-proc/multi-thread not included here).
    """

    def __init__(self, model: Module, device_ids: List[str], output_device: Optional[str] = None):
        self.module = model
        self.device_ids = device_ids
        self.output_device = output_device or device_ids[0]
        self.num_devices = len(device_ids)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def parameters(self):
        return self.module.parameters()

    def train(self, mode: bool = True):
        self.module.train(mode)
        return self

    def eval(self):
        self.module.eval()
        return self

    def zero_grad(self):
        self.module.zero_grad()
