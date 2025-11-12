"""Device to communicator routing helpers."""

from __future__ import annotations

from typing import Any, Optional

from . import process_group


def get_device_type(device: Optional[str]) -> Optional[str]:
    if device is None:
        return None
    device = device.lower()
    if ":" in device:
        device = device.split(":", 1)[0]
    return device


def get_communicator(device_or_tensor: Any):
    """Return the communicator bound to ``device_or_tensor``."""

    device = None
    if hasattr(device_or_tensor, "active_device"):
        device = getattr(device_or_tensor, "active_device")
    elif isinstance(device_or_tensor, str):
        device = device_or_tensor

    device_type = get_device_type(device)
    if not process_group.is_initialized():
        process_group.lazy_init_from_env()
    return process_group.get_backend(device_type)


__all__ = ["get_communicator", "get_device_type"]
