"""Core abstractions for distributed communication backends."""

from __future__ import annotations

import abc
from typing import Any, Callable, Optional


class CollectiveBackend(abc.ABC):
    """Abstract interface implemented by distributed communication backends."""

    #: Human readable backend identifier. Sub-classes should override.
    name: str = "unknown"
    #: Device types supported by the backend. Used for routing tensors.
    device_types: tuple[str, ...] = ("cpu",)

    def __init__(self) -> None:
        self._initialized: bool = False
        self._world_size: int = 1
        self._rank: int = 0
        self._notes: list[str] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if the backend dependencies are available."""

    def initialize(
        self,
        *,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        **_: Any,
    ) -> None:
        """Mark the backend as active.

        Concrete backends should extend this implementation to perform the
        actual initialization, but *must* invoke ``super().initialize`` to
        ensure book-keeping remains consistent.
        """

        self._rank = 0 if rank is None else rank
        self._world_size = 1 if world_size is None else world_size
        self._initialized = True

    def finalize(self) -> None:
        """Tear down the backend."""

        self._initialized = False
        self._world_size = 1
        self._rank = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def notes(self) -> tuple[str, ...]:
        return tuple(self._notes)

    def add_note(self, message: str) -> None:
        self._notes.append(message)

    # ------------------------------------------------------------------
    # Collective operations
    # ------------------------------------------------------------------
    def barrier(self) -> None:  # pragma: no cover - trivial default impl
        """Synchronize all ranks.

        Default implementation is a no-op which is appropriate for the
        single-process fallback. Real backends should provide a concrete
        implementation.
        """

    def _return_tensor(self, tensor: Any, *, updater: Optional[Callable[[Any], None]] = None) -> Any:
        if updater is not None:
            updater(tensor)
        return tensor

    def all_reduce(self, tensor: Any, op: str = "sum") -> Any:
        """Reduce ``tensor`` across processes and broadcast the result."""

        return self._return_tensor(tensor)

    def broadcast(self, tensor: Any, src: int = 0) -> Any:
        """Broadcast ``tensor`` from ``src`` to every rank."""

        return self._return_tensor(tensor)

    def all_gather(self, tensor: Any) -> list[Any]:
        """Gather ``tensor`` from every rank and return the ordered list."""

        return [tensor]

    def reduce_scatter(self, tensors: list[Any], op: str = "sum") -> Any:
        """Scatter ``tensors`` and reduce the received shard."""

        if not tensors:
            raise ValueError("reduce_scatter expects at least one tensor")
        return tensors[0]

    def send(self, tensor: Any, dst: int) -> Any:  # pragma: no cover - default no-op
        return self._return_tensor(tensor)

    def recv(self, tensor: Any, src: int) -> Any:  # pragma: no cover - default no-op
        return self._return_tensor(tensor)

    def gather(self, tensor: Any, dst: int = 0) -> list[Any]:  # pragma: no cover
        if dst != self.rank:
            return []
        return [tensor]

    def scatter(self, tensors: list[Any], src: int = 0) -> Any:  # pragma: no cover
        if not tensors:
            raise ValueError("scatter expects tensors to broadcast")
        return tensors[0]


class DummyBackend(CollectiveBackend):
    """Trivial backend used as a safe fallback."""

    name = "dummy"
    device_types = ("cpu", "cuda", "xpu")

    def is_available(self) -> bool:
        return True


__all__ = ["CollectiveBackend", "DummyBackend"]
