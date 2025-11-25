from . import backend
from .backend import get_available_devices, synchronize, get_device

def is_available():
    return backend.AVAILABLE