"""
PySML Intel XPU Backend Interface
Complete implementation with all device management functions
"""

from . import backend

# Expose availability check
AVAILABLE = backend.AVAILABLE

def is_available():
    """
    Check if Intel XPU is available
    
    Returns:
        bool: True if Intel XPU/DPNP is available, False otherwise
    
    Example:
        >>> import pysml
        >>> if pysml.xpu.is_available():
        ...     print("Intel XPU is available!")
    """
    return AVAILABLE

def get_device_count():
    """
    Get number of Intel XPU devices
    
    Returns:
        int: Number of available XPU devices
    
    Example:
        >>> import pysml
        >>> if pysml.xpu.is_available():
        ...     print(f"Found {pysml.xpu.get_device_count()} XPU devices")
    """
    if not AVAILABLE:
        return 0
    
    try:
        import dpctl
        gpu_devices = dpctl.get_devices(device_type="gpu", backend="level_zero")
        return len(gpu_devices)
    except Exception:
        return 0

def get_device_name(device_id=0):
    """
    Get name of XPU device
    
    Args:
        device_id: Device index (default: 0)
    
    Returns:
        str: Device name or None if not available
    
    Example:
        >>> import pysml
        >>> if pysml.xpu.is_available():
        ...     print(pysml.xpu.get_device_name(0))
        Intel(R) Arc(TM) A770 Graphics
    """
    if not AVAILABLE:
        return None
    
    try:
        import dpctl
        gpu_devices = dpctl.get_devices(device_type="gpu", backend="level_zero")
        if device_id < len(gpu_devices):
            return gpu_devices[device_id].name
        return f"XPU Device {device_id}"
    except Exception:
        return f"XPU Device {device_id}"

def get_device_properties(device_id=0):
    """
    Get properties of XPU device
    
    Args:
        device_id: Device index (default: 0)
    
    Returns:
        dict: Device properties including name, memory, compute units, etc.
    
    Example:
        >>> import pysml
        >>> if pysml.xpu.is_available():
        ...     props = pysml.xpu.get_device_properties(0)
        ...     print(f"Name: {props['name']}")
        ...     print(f"Memory: {props['global_mem_size'] / 1e9:.2f} GB")
    """
    if not AVAILABLE:
        return {}
    
    try:
        import dpctl
        gpu_devices = dpctl.get_devices(device_type="gpu", backend="level_zero")
        if device_id < len(gpu_devices):
            device = gpu_devices[device_id]
            return {
                'name': device.name,
                'vendor': device.vendor,
                'driver_version': device.driver_version,
                'backend': device.backend.name,
                'device_type': device.device_type.name,
                'max_compute_units': device.max_compute_units,
                'max_work_item_dims': device.max_work_item_dims,
                'max_work_group_size': device.max_work_group_size,
                'max_num_sub_groups': device.max_num_sub_groups,
                'global_mem_size': device.global_mem_size,
                'local_mem_size': device.local_mem_size,
                'has_aspect_fp64': device.has_aspect_fp64,
                'has_aspect_fp16': device.has_aspect_fp16,
            }
        return {}
    except Exception as e:
        return {'error': str(e)}

def get_available_devices():
    """
    Get list of all available XPU devices
    
    Returns:
        list: List of device strings like ['xpu:0', 'xpu:1', ...]
    
    Example:
        >>> import pysml
        >>> if pysml.xpu.is_available():
        ...     devices = pysml.xpu.get_available_devices()
        ...     print(devices)
        ['xpu:0', 'xpu:1']
    """
    if not AVAILABLE:
        return []
    
    try:
        import dpctl
        gpu_devices = dpctl.get_devices(device_type="gpu", backend="level_zero")
        return [f"xpu:{i}" for i in range(len(gpu_devices))]
    except Exception:
        return []

def set_device(device_id=0):
    """
    Set active XPU device
    
    Args:
        device_id: Device index to activate
    
    Example:
        >>> import pysml
        >>> if pysml.xpu.is_available():
        ...     pysml.xpu.set_device(0)
    """
    if not AVAILABLE:
        raise RuntimeError("Intel XPU is not available")
    
    try:
        import dpctl
        gpu_devices = dpctl.get_devices(device_type="gpu", backend="level_zero")
        if device_id < len(gpu_devices):
            # Create a queue for the specified device
            queue = dpctl.SyclQueue(gpu_devices[device_id])
            dpctl.set_default_queue(queue)
    except Exception as e:
        raise RuntimeError(f"Failed to set XPU device {device_id}: {e}")

def synchronize():
    """
    Synchronize all XPU operations
    
    Waits for all queued XPU operations to complete.
    
    Example:
        >>> import pysml
        >>> if pysml.xpu.is_available():
        ...     # ... do XPU operations ...
        ...     pysml.xpu.synchronize()  # Wait for completion
    """
    if AVAILABLE:
        backend.synchronize()

def init():
    """
    Initialize Intel XPU backend
    
    Verifies that Intel XPU is available and can be used.
    Raises an error with helpful installation instructions if not available.
    
    Raises:
        RuntimeError: If Intel XPU is not available or initialization fails
    
    Example:
        >>> import pysml
        >>> if pysml.xpu.is_available():
        ...     pysml.xpu.init()
        ...     print("Intel XPU initialized successfully")
    """
    if not AVAILABLE:
        raise RuntimeError(
            "Intel XPU is not available. Please install dpnp and dpctl:\n"
            "  pip install dpnp dpctl\n\n"
            "Or from Intel's repository for best compatibility:\n"
            "  pip install -i https://software.repos.intel.com/python/pypi numpy dpnp dpctl\n\n"
            "For more information, visit:\n"
            "  https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html"
        )
    
    try:
        import dpnp
        import dpctl
        
        # Verify we can create arrays
        test_array = dpnp.zeros(1)
        
        # Check for available devices
        gpu_devices = dpctl.get_devices(device_type="gpu", backend="level_zero")
        if len(gpu_devices) == 0:
            raise RuntimeError(
                "No Intel XPU devices found. Please ensure:\n"
                "  1. Intel GPU drivers are installed\n"
                "  2. Intel oneAPI Base Toolkit is installed\n"
                "  3. Your system has an Intel GPU (Arc, Xe, or Data Center GPU)\n\n"
                "For driver installation, visit:\n"
                "  https://www.intel.com/content/www/us/en/download/726609/"
            )
        
        # Set default device
        queue = dpctl.SyclQueue(gpu_devices[0])
        dpctl.set_default_queue(queue)
        
    except ImportError as e:
        raise RuntimeError(f"Intel XPU dependencies not properly installed: {e}")
    except Exception as e:
        raise RuntimeError(f"Intel XPU initialization failed: {e}")

def get_memory_info(device_id=0):
    """
    Get memory information for XPU device
    
    Args:
        device_id: Device index (default: 0)
    
    Returns:
        dict: Memory information including total and free memory
    
    Example:
        >>> import pysml
        >>> if pysml.xpu.is_available():
        ...     mem = pysml.xpu.get_memory_info(0)
        ...     print(f"Total memory: {mem['total'] / 1e9:.2f} GB")
    """
    if not AVAILABLE:
        return {}
    
    props = get_device_properties(device_id)
    if 'global_mem_size' in props:
        return {
            'total': props['global_mem_size'],
            'device_id': device_id,
        }
    return {}

def print_info(device_id=0):
    """
    Print detailed information about XPU device
    
    Args:
        device_id: Device index to print info for (default: 0)
    
    Example:
        >>> import pysml
        >>> if pysml.xpu.is_available():
        ...     pysml.xpu.print_info(0)
        ================================================================================
        Intel XPU Device 0 Information
        ================================================================================
        Name: Intel(R) Arc(TM) A770 Graphics
        Vendor: Intel(R) Corporation
        Driver Version: 1.3.26918
        ...
        ================================================================================
    """
    if not AVAILABLE:
        print("Intel XPU is not available")
        return
    
    print("=" * 80)
    print(f"Intel XPU Device {device_id} Information")
    print("=" * 80)
    
    props = get_device_properties(device_id)
    
    if 'error' in props:
        print(f"Error getting device properties: {props['error']}")
        print("=" * 80)
        return
    
    if not props:
        print(f"Device {device_id} not found")
        print("=" * 80)
        return
    
    print(f"Name: {props.get('name', 'Unknown')}")
    print(f"Vendor: {props.get('vendor', 'Unknown')}")
    print(f"Driver Version: {props.get('driver_version', 'Unknown')}")
    print(f"Backend: {props.get('backend', 'Unknown')}")
    print(f"Device Type: {props.get('device_type', 'Unknown')}")
    
    print(f"\nCompute Capabilities:")
    print(f"  Max Compute Units: {props.get('max_compute_units', 'Unknown')}")
    print(f"  Max Work Group Size: {props.get('max_work_group_size', 'Unknown')}")
    print(f"  Max Work Item Dims: {props.get('max_work_item_dims', 'Unknown')}")
    print(f"  Max Sub Groups: {props.get('max_num_sub_groups', 'Unknown')}")
    
    print(f"\nMemory:")
    global_mem = props.get('global_mem_size', 0)
    local_mem = props.get('local_mem_size', 0)
    if global_mem:
        print(f"  Global Memory: {global_mem / 1e9:.2f} GB")
    if local_mem:
        print(f"  Local Memory: {local_mem / 1024:.2f} KB")
    
    print(f"\nPrecision Support:")
    print(f"  FP64: {'Yes' if props.get('has_aspect_fp64', False) else 'No'}")
    print(f"  FP16: {'Yes' if props.get('has_aspect_fp16', False) else 'No'}")
    
    print("=" * 80)

def print_all_devices():
    """
    Print information about all available XPU devices
    
    Example:
        >>> import pysml
        >>> if pysml.xpu.is_available():
        ...     pysml.xpu.print_all_devices()
    """
    if not AVAILABLE:
        print("Intel XPU is not available")
        return
    
    count = get_device_count()
    print(f"\nFound {count} Intel XPU device{'s' if count != 1 else ''}")
    print()
    
    for i in range(count):
        print_info(i)
        if i < count - 1:
            print()

__all__ = [
    'is_available',
    'get_device_count',
    'get_device_name',
    'get_device_properties',
    'get_available_devices',
    'set_device',
    'synchronize',
    'init',
    'get_memory_info',
    'print_info',
    'print_all_devices',
    'AVAILABLE',
    'backend',
]