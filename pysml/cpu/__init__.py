

from . import backend

# CPU is always available
AVAILABLE = True

def is_available():

    return True

def get_device_count():

    return 1

def get_device_name():

    return "CPU"

def get_device_properties():

    import platform
    import multiprocessing
    
    return {
        'name': 'CPU',
        'processor': platform.processor(),
        'architecture': platform.machine(),
        'cores': multiprocessing.cpu_count(),
    }

__all__ = [
    'is_available',
    'get_device_count', 
    'get_device_name',
    'get_device_properties',
    'AVAILABLE',
]
