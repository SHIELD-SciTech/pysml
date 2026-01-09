from . import backend
from .backend import get_available_devices, synchronize, get_device

AVAILABLE = backend.AVAILABLE

def is_available():
    return backend.AVAILABLE

def device_count():

    if not AVAILABLE:
        return 0
    return len(get_available_devices())

__all__ = [
    'backend',
    'is_available',
    'device_count',
    'get_available_devices',
    'synchronize',
    'get_device',
    'AVAILABLE',
]
