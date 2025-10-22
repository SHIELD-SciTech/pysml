"""
PySML CPU Backend Interface
"""

from . import backend

# CPU is always available
AVAILABLE = True

def is_available():
    """
    Check if CPU is available (always True)
    
    Returns:
        bool: True (CPU is always available)
    
    Example:
        >>> import pysml
        >>> print(pysml.cpu.is_available())
        True
    """
    return True

def get_device_count():
    """
    Get number of CPU devices (always 1)
    
    Returns:
        int: 1 (single CPU device)
    
    Example:
        >>> import pysml
        >>> print(pysml.cpu.get_device_count())
        1
    """
    return 1

def get_device_name():
    """
    Get CPU device name
    
    Returns:
        str: "CPU"
    
    Example:
        >>> import pysml
        >>> print(pysml.cpu.get_device_name())
        CPU
    """
    return "CPU"

def get_device_properties():
    """
    Get CPU device properties
    
    Returns:
        dict: Basic CPU information
    
    Example:
        >>> import pysml
        >>> props = pysml.cpu.get_device_properties()
        >>> print(props)
    """
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