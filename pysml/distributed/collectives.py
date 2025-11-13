"""User facing collective communication helpers with fault tolerance."""

from __future__ import annotations

import contextlib
import os
import signal
import threading
import time
from typing import Any, Callable, Iterable, List, Optional, TypeVar

from .routing import get_communicator

_T = TypeVar("_T")


class CollectiveOperationError(RuntimeError):
    """Raised when a collective fails after exhausting retries."""


_DEFAULT_TIMEOUT = float(os.getenv("PYSML_COLLECTIVE_TIMEOUT", "0") or 0.0)
_DEFAULT_RETRIES = int(os.getenv("PYSML_COLLECTIVE_RETRIES", "0") or 0)
_FAILURE_STATE = {"active": False, "op": None, "error": None}
_FAILURE_LOCK = threading.Lock()


def _record_failure(op_name: str, exc: BaseException) -> None:
    with _FAILURE_LOCK:
        if not _FAILURE_STATE["active"]:
            _FAILURE_STATE["active"] = True
            _FAILURE_STATE["op"] = op_name
            _FAILURE_STATE["error"] = str(exc)


def _ensure_no_prior_failure() -> None:
    if _FAILURE_STATE["active"]:
        raise CollectiveOperationError(
            "Previous collective operation failed "
            f"({ _FAILURE_STATE['op'] }): {_FAILURE_STATE['error']}"
        )


@contextlib.contextmanager
def _deadline(seconds: Optional[float]):
    if not seconds or seconds <= 0:
        yield
        return
    if os.name != "posix" or not hasattr(signal, "SIGALRM"):
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        if elapsed > seconds:
            raise TimeoutError(f"Collective exceeded timeout ({seconds}s)")
        return

    def _handler(*_):  # pragma: no cover - depends on signal delivery
        raise TimeoutError(f"Collective exceeded timeout ({seconds}s)")

    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _run_collective(
    name: str,
    communicator,
    fn: Callable[[], _T],
    *,
    timeout: Optional[float] = None,
    retries: Optional[int] = None,
) -> _T:
    _ensure_no_prior_failure()
    allowed_retries = _DEFAULT_RETRIES if retries is None else max(int(retries), 0)
    timeout = _DEFAULT_TIMEOUT if timeout is None else timeout
    attempt = 0
    last_exc: Optional[BaseException] = None

    while attempt <= allowed_retries:
        try:
            with _deadline(timeout):
                return fn()
        except BaseException as exc:  # noqa: PERF203 - unified handling
            last_exc = exc
            attempt += 1
            if attempt > allowed_retries:
                _record_failure(name, exc)
                raise CollectiveOperationError(
                    f"{name} failed after {allowed_retries + 1} attempt(s): {exc}"
                ) from exc

    # This point is unreachable but satisfies the type-checker.
    raise CollectiveOperationError(f"{name} failed: {last_exc}")


def all_reduce(
    tensor: Any,
    op: str = "sum",
    *,
    timeout: Optional[float] = None,
    retries: Optional[int] = None,
) -> Any:
    communicator = get_communicator(tensor)
    return _run_collective(
        "all_reduce", communicator, lambda: communicator.all_reduce(tensor, op=op), timeout=timeout, retries=retries
    )


def broadcast(
    tensor: Any,
    src: int = 0,
    *,
    timeout: Optional[float] = None,
    retries: Optional[int] = None,
) -> Any:
    communicator = get_communicator(tensor)
    return _run_collective(
        "broadcast", communicator, lambda: communicator.broadcast(tensor, src=src), timeout=timeout, retries=retries
    )


def all_gather(
    tensor: Any,
    *,
    timeout: Optional[float] = None,
    retries: Optional[int] = None,
) -> Iterable[Any]:
    communicator = get_communicator(tensor)
    return _run_collective("all_gather", communicator, lambda: communicator.all_gather(tensor), timeout=timeout, retries=retries)


def reduce_scatter(
    tensors: Iterable[Any],
    op: str = "sum",
    *,
    timeout: Optional[float] = None,
    retries: Optional[int] = None,
) -> Any:
    tensors = list(tensors)
    if not tensors:
        raise ValueError("reduce_scatter requires a non-empty iterable")
    communicator = get_communicator(tensors[0])
    return _run_collective(
        "reduce_scatter",
        communicator,
        lambda: communicator.reduce_scatter(tensors, op=op),
        timeout=timeout,
        retries=retries,
    )


def barrier(
    device: str | None = None,
    *,
    timeout: Optional[float] = None,
    retries: Optional[int] = None,
) -> None:
    communicator = get_communicator(device)
    _run_collective("barrier", communicator, communicator.barrier, timeout=timeout, retries=retries)


def send(
    tensor: Any,
    dst: int,
    *,
    timeout: Optional[float] = None,
    retries: Optional[int] = None,
) -> Any:
    communicator = get_communicator(tensor)
    return _run_collective("send", communicator, lambda: communicator.send(tensor, dst), timeout=timeout, retries=retries)


def recv(
    tensor: Any,
    src: int,
    *,
    timeout: Optional[float] = None,
    retries: Optional[int] = None,
) -> Any:
    communicator = get_communicator(tensor)
    return _run_collective("recv", communicator, lambda: communicator.recv(tensor, src), timeout=timeout, retries=retries)


def gather(
    tensor: Any,
    dst: int = 0,
    *,
    timeout: Optional[float] = None,
    retries: Optional[int] = None,
) -> List[Any]:
    communicator = get_communicator(tensor)
    return _run_collective("gather", communicator, lambda: communicator.gather(tensor, dst=dst), timeout=timeout, retries=retries)


def scatter(
    tensors: Iterable[Any],
    src: int = 0,
    *,
    timeout: Optional[float] = None,
    retries: Optional[int] = None,
) -> Any:
    tensors = list(tensors)
    if not tensors:
        raise ValueError("scatter requires tensors")
    communicator = get_communicator(tensors[0])
    return _run_collective(
        "scatter",
        communicator,
        lambda: communicator.scatter(tensors, src=src),
        timeout=timeout,
        retries=retries,
    )


__all__ = [
    "all_reduce",
    "broadcast",
    "all_gather",
    "reduce_scatter",
    "barrier",
    "send",
    "recv",
    "gather",
    "scatter",
    "CollectiveOperationError",
]
