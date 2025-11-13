"""Debugging helpers for multi-process execution."""

from __future__ import annotations

import code
from typing import Any, Callable, Dict, List, Optional

from pysml import autograd
from ..nn.module import Module
from ..tensor import Tensor
from . import process_group


def launch_rank_repl(namespace: Optional[Dict[str, Any]] = None, *, interactive: bool = True, banner: Optional[str] = None) -> str:
    """Launch a rank-aware REPL for interactive debugging.

    When ``interactive`` is ``False`` the banner text is returned without
    starting the interactive console which makes the helper test friendly.
    """

    process_group.lazy_init_from_env()
    backend = process_group.get_backend()
    ns = dict(namespace or {})
    ns.setdefault("rank", backend.rank)
    ns.setdefault("world_size", backend.world_size)
    message = banner or f"PySML distributed REPL (rank {backend.rank}/{backend.world_size - 1})"
    if interactive:
        console = code.InteractiveConsole(ns)
        console.interact(message)
    return message


class GradientAnomalyError(RuntimeError):
    """Raised when gradients contain NaNs, infs, or explode beyond a threshold."""


def register_gradient_anomaly_detector(
    module: Module,
    *,
    detect_nan: bool = True,
    detect_inf: bool = True,
    max_norm: Optional[float] = None,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Tensor._PostBackwardHookHandle]:
    """Attach post-backward hooks that watch for gradient anomalies."""

    process_group.lazy_init_from_env()
    backend = process_group.get_backend()
    rank = backend.rank

    def _to_numpy(tensor: Tensor):
        data = getattr(tensor, "data", tensor)
        backend_impl = getattr(tensor, "_backend", None)
        to_numpy = getattr(backend_impl, "to_numpy", None)
        if callable(to_numpy):
            return to_numpy(data)
        try:
            import numpy as np

            return np.asarray(data)
        except Exception:
            return [float(data)]

    handles: List[Tensor._PostBackwardHookHandle] = []

    def _default_callback(event: Dict[str, Any]) -> None:
        raise GradientAnomalyError(
            f"Gradient anomaly detected on rank {rank} for {event['parameter']}: {event['reason']}"
        )

    reporter = callback or _default_callback

    for name, parameter in module.named_parameters():
        tensor = parameter.data
        tracked_parameter = parameter

        def _hook(current: Tensor, param_name: str = name, param_ref=tracked_parameter) -> None:
            grad = param_ref.grad
            if grad is None:
                return
            grad_array = _to_numpy(grad)
            reason = None
            if detect_nan and _has_nan(grad_array):
                reason = "NaN"
            elif detect_inf and _has_inf(grad_array):
                reason = "Inf"
            elif max_norm is not None and _norm_exceeds(grad_array, max_norm):
                reason = f"norm>{max_norm}"
            if reason is not None:
                payload = {"parameter": param_name, "reason": reason, "rank": rank}
                reporter(payload)

        handle = autograd.register_post_backward_hook(tensor, _hook)
        handles.append(handle)
    return handles


def _has_nan(array: Any) -> bool:
    try:
        import numpy as np

        arr = np.asarray(array)
        return bool(np.isnan(arr).any())
    except Exception:
        return False


def _has_inf(array: Any) -> bool:
    try:
        import numpy as np

        arr = np.asarray(array)
        return bool(np.isinf(arr).any())
    except Exception:
        return False


def _norm_exceeds(array: Any, threshold: float) -> bool:
    try:
        import numpy as np

        arr = np.asarray(array, dtype=float)
        norm = float(np.linalg.norm(arr))
        return norm > threshold
    except Exception:
        if isinstance(array, (int, float)):
            return abs(float(array)) > threshold
        return False


__all__ = [
    "launch_rank_repl",
    "register_gradient_anomaly_detector",
    "GradientAnomalyError",
]
