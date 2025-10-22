"""
PySML CUDA Backend Interface
Complete implementation with all device management functions
"""

from . import backend

# Expose availability check
AVAILABLE = backend.AVAILABLE

def is_available(): # Must have cupy installed
    """
    Check if CUDA is available
    
    Returns:
        bool: True if CUDA/CuPy is available, False otherwise
    
    Example:
        >>> import pysml
        >>> if pysml.cuda.is_available():
        ...     print("CUDA is available!")
    """
    return AVAILABLE

def get_device_count():
    """
    Get number of CUDA devices
    
    Returns:
        int: Number of available CUDA devices
    
    Example:
        >>> import pysml
        >>> if pysml.cuda.is_available():
        ...     print(f"Found {pysml.cuda.get_device_count()} CUDA devices")
    """
    if not AVAILABLE:
        return 0
    
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount()
    except Exception:
        return 0

def get_device_name(device_id=0):
    """
    Get name of CUDA device
    
    Args:
        device_id: Device index (default: 0)
    
    Returns:
        str: Device name or None if not available
    
    Example:
        >>> import pysml
        >>> if pysml.cuda.is_available():
        ...     print(pysml.cuda.get_device_name(0))
        NVIDIA GeForce RTX 3080
    """
    if not AVAILABLE:
        return None
    
    try:
        import cupy as cp
        with cp.cuda.Device(device_id):
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            return props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    except Exception:
        return f"CUDA Device {device_id}"

def get_device_properties(device_id=0):
    """
    Get properties of CUDA device
    
    Args:
        device_id: Device index (default: 0)
    
    Returns:
        dict: Device properties including name, memory, compute capability, etc.
    
    Example:
        >>> import pysml
        >>> if pysml.cuda.is_available():
        ...     props = pysml.cuda.get_device_properties(0)
        ...     print(f"Name: {props['name']}")
        ...     print(f"Memory: {props['totalGlobalMem'] / 1e9:.2f} GB")
        ...     print(f"Compute Capability: {props['major']}.{props['minor']}")
    """
    if not AVAILABLE:
        return {}
    
    try:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        
        # Decode bytes to strings
        result = {}
        for key, value in props.items():
            if isinstance(value, bytes):
                result[key] = value.decode()
            else:
                result[key] = value
        
        return result
    except Exception as e:
        return {'error': str(e)}

def get_available_devices():
    """
    Get list of all available CUDA devices
    
    Returns:
        list: List of device strings like ['cuda:0', 'cuda:1', ...]
    
    Example:
        >>> import pysml
        >>> if pysml.cuda.is_available():
        ...     devices = pysml.cuda.get_available_devices()
        ...     print(devices)
        ['cuda:0', 'cuda:1']
    """
    if not AVAILABLE:
        return []
    
    try:
        import cupy as cp
        num_devices = cp.cuda.runtime.getDeviceCount()
        return [f"cuda:{i}" for i in range(num_devices)]
    except Exception:
        return []

def set_device(device_id=0):
    """
    Set active CUDA device
    
    Args:
        device_id: Device index to activate
    
    Example:
        >>> import pysml
        >>> if pysml.cuda.is_available():
        ...     pysml.cuda.set_device(0)
    """
    if not AVAILABLE:
        raise RuntimeError("CUDA is not available")
    
    try:
        import cupy as cp
        cp.cuda.Device(device_id).use()
    except Exception as e:
        raise RuntimeError(f"Failed to set CUDA device {device_id}: {e}")

def get_current_device():
    """
    Get currently active CUDA device
    
    Returns:
        int: Current device ID
    
    Example:
        >>> import pysml
        >>> if pysml.cuda.is_available():
        ...     current = pysml.cuda.get_current_device()
        ...     print(f"Current device: cuda:{current}")
    """
    if not AVAILABLE:
        return None
    
    try:
        import cupy as cp
        return cp.cuda.Device().id
    except Exception:
        return None

def synchronize():
    """
    Synchronize all CUDA operations
    
    Waits for all queued CUDA operations to complete.
    
    Example:
        >>> import pysml
        >>> if pysml.cuda.is_available():
        ...     # ... do CUDA operations ...
        ...     pysml.cuda.synchronize()  # Wait for completion
    """
    if AVAILABLE:
        backend.synchronize()

def init():
    """
    Initialize CUDA backend
    
    Verifies that CUDA is available and can be used.
    Raises an error with helpful installation instructions if not available.
    
    Raises:
        RuntimeError: If CUDA is not available or initialization fails
    
    Example:
        >>> import pysml
        >>> if pysml.cuda.is_available():
        ...     pysml.cuda.init()
        ...     print("CUDA initialized successfully")
    """
    if not AVAILABLE:
        raise RuntimeError(
            "CUDA is not available. Please install CuPy:\n"
            "  For CUDA 12.x: pip install cupy-cuda12x\n"
            "  For CUDA 11.x: pip install cupy-cuda11x\n\n"
            "Also ensure NVIDIA CUDA Toolkit is installed:\n"
            "  https://developer.nvidia.com/cuda-downloads\n\n"
            "For more information, visit:\n"
            "  https://docs.cupy.dev/en/stable/install.html"
        )
    
    try:
        import cupy as cp
        
        # Verify CUDA is working
        test_array = cp.zeros(1)
        
        # Check for available devices
        num_devices = cp.cuda.runtime.getDeviceCount()
        if num_devices == 0:
            raise RuntimeError(
                "No CUDA devices found. Please ensure:\n"
                "  1. NVIDIA GPU drivers are installed\n"
                "  2. CUDA Toolkit is installed\n"
                "  3. Your system has a CUDA-capable NVIDIA GPU\n\n"
                "For driver installation, visit:\n"
                "  https://www.nvidia.com/Download/index.aspx"
            )
        
        # Get current device and verify it works
        device = cp.cuda.Device()
        device.use()
        
    except ImportError as e:
        raise RuntimeError(f"CuPy not properly installed: {e}")
    except Exception as e:
        raise RuntimeError(f"CUDA initialization failed: {e}")

def get_memory_info(device_id=0):
    """
    Get memory information for CUDA device
    
    Args:
        device_id: Device index (default: 0)
    
    Returns:
        dict: Memory information including total, free, and used memory
    
    Example:
        >>> import pysml
        >>> if pysml.cuda.is_available():
        ...     mem = pysml.cuda.get_memory_info(0)
        ...     print(f"Total: {mem['total'] / 1e9:.2f} GB")
        ...     print(f"Free: {mem['free'] / 1e9:.2f} GB")
        ...     print(f"Used: {mem['used'] / 1e9:.2f} GB")
    """
    if not AVAILABLE:
        return {}
    
    try:
        import cupy as cp
        with cp.cuda.Device(device_id):
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            used_mem = total_mem - free_mem
            
            return {
                'total': total_mem,
                'free': free_mem,
                'used': used_mem,
                'used_percent': (used_mem / total_mem * 100) if total_mem > 0 else 0,
                'device_id': device_id,
            }
    except Exception as e:
        return {'error': str(e)}

def reset_device(device_id=0):
    """
    Reset CUDA device (clears memory)
    
    WARNING: This will clear all allocations on the device!
    
    Args:
        device_id: Device index to reset
    
    Example:
        >>> import pysml
        >>> if pysml.cuda.is_available():
        ...     pysml.cuda.reset_device(0)  # Clear all memory
    """
    if not AVAILABLE:
        raise RuntimeError("CUDA is not available")
    
    try:
        import cupy as cp
        with cp.cuda.Device(device_id):
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as e:
        raise RuntimeError(f"Failed to reset device {device_id}: {e}")

def print_info(device_id=0):
    """
    Print detailed information about CUDA device
    
    Args:
        device_id: Device index to print info for (default: 0)
    
    Example:
        >>> import pysml
        >>> if pysml.cuda.is_available():
        ...     pysml.cuda.print_info(0)
        ================================================================================
        CUDA Device 0 Information
        ================================================================================
        Name: NVIDIA GeForce RTX 3080
        Compute Capability: 8.6
        Total Memory: 10.00 GB
        ...
        ================================================================================
    """
    if not AVAILABLE:
        print("CUDA is not available")
        return
    
    print("=" * 80)
    print(f"CUDA Device {device_id} Information")
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
    print(f"Compute Capability: {props.get('major', '?')}.{props.get('minor', '?')}")
    
    total_mem = props.get('totalGlobalMem', 0)
    if total_mem:
        print(f"Total Memory: {total_mem / 1e9:.2f} GB")
    
    print(f"\nCompute Specifications:")
    print(f"  Multiprocessors: {props.get('multiProcessorCount', 'Unknown')}")
    print(f"  CUDA Cores per MP: {props.get('maxThreadsPerMultiProcessor', 'Unknown')}")
    print(f"  Warp Size: {props.get('warpSize', 'Unknown')}")
    print(f"  Max Threads per Block: {props.get('maxThreadsPerBlock', 'Unknown')}")
    print(f"  Max Block Dimensions: ({props.get('maxThreadsDim', ['?', '?', '?'])[0]}, "
          f"{props.get('maxThreadsDim', ['?', '?', '?'])[1]}, "
          f"{props.get('maxThreadsDim', ['?', '?', '?'])[2]})")
    print(f"  Max Grid Dimensions: ({props.get('maxGridSize', ['?', '?', '?'])[0]}, "
          f"{props.get('maxGridSize', ['?', '?', '?'])[1]}, "
          f"{props.get('maxGridSize', ['?', '?', '?'])[2]})")
    
    print(f"\nMemory Specifications:")
    print(f"  Shared Memory per Block: {props.get('sharedMemPerBlock', 0) / 1024:.2f} KB")
    print(f"  Constant Memory: {props.get('totalConstMem', 0) / 1024:.2f} KB")
    print(f"  L2 Cache Size: {props.get('l2CacheSize', 0) / 1024:.2f} KB")
    print(f"  Memory Clock Rate: {props.get('memoryClockRate', 0) / 1000:.2f} MHz")
    print(f"  Memory Bus Width: {props.get('memoryBusWidth', 'Unknown')} bits")
    
    print(f"\nClock Rates:")
    print(f"  GPU Clock Rate: {props.get('clockRate', 0) / 1000:.2f} MHz")
    print(f"  Memory Clock Rate: {props.get('memoryClockRate', 0) / 1000:.2f} MHz")
    
    # Get current memory usage
    mem_info = get_memory_info(device_id)
    if 'total' in mem_info:
        print(f"\nCurrent Memory Usage:")
        print(f"  Total: {mem_info['total'] / 1e9:.2f} GB")
        print(f"  Used: {mem_info['used'] / 1e9:.2f} GB ({mem_info['used_percent']:.1f}%)")
        print(f"  Free: {mem_info['free'] / 1e9:.2f} GB")
    
    print("=" * 80)

def print_all_devices():
    """
    Print information about all available CUDA devices
    
    Example:
        >>> import pysml
        >>> if pysml.cuda.is_available():
        ...     pysml.cuda.print_all_devices()
    """
    if not AVAILABLE:
        print("CUDA is not available")
        return
    
    count = get_device_count()
    print(f"\nFound {count} CUDA device{'s' if count != 1 else ''}")
    print()
    
    for i in range(count):
        print_info(i)
        if i < count - 1:
            print()

def get_cuda_version():
    """
    Get CUDA runtime version
    
    Returns:
        tuple: (major, minor) version numbers or None if not available
    
    Example:
        >>> import pysml
        >>> if pysml.cuda.is_available():
        ...     version = pysml.cuda.get_cuda_version()
        ...     print(f"CUDA Version: {version[0]}.{version[1]}")
    """
    if not AVAILABLE:
        return None
    
    try:
        import cupy as cp
        runtime_version = cp.cuda.runtime.runtimeGetVersion()
        major = runtime_version // 1000
        minor = (runtime_version % 1000) // 10
        return (major, minor)
    except Exception:
        return None

def get_cudnn_version():
    """
    Get cuDNN version if available
    
    Returns:
        int: cuDNN version number or None if not available
    
    Example:
        >>> import pysml
        >>> if pysml.cuda.is_available():
        ...     version = pysml.cuda.get_cudnn_version()
        ...     if version:
        ...         print(f"cuDNN Version: {version}")
    """
    if not AVAILABLE:
        return None
    
    try:
        import cupy as cp
        return cp.cuda.cudnn.getVersion()
    except Exception:
        return None

def empty_cache():
    """
    Empty CUDA memory cache
    
    Releases all unused cached memory so it can be used by other applications.
    This does NOT free memory that is currently in use.
    
    Example:
        >>> import pysml
        >>> if pysml.cuda.is_available():
        ...     pysml.cuda.empty_cache()
        ...     print("Cache cleared")
    """
    if not AVAILABLE:
        return
    
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass

def memory_summary(device_id=0):
    """
    Get detailed memory summary for device
    
    Args:
        device_id: Device index (default: 0)
    
    Returns:
        str: Formatted memory summary
    
    Example:
        >>> import pysml
        >>> if pysml.cuda.is_available():
        ...     print(pysml.cuda.memory_summary(0))
    """
    if not AVAILABLE:
        return "CUDA is not available"
    
    mem = get_memory_info(device_id)
    props = get_device_properties(device_id)
    
    if 'error' in mem or 'error' in props:
        return f"Error getting memory info for device {device_id}"
    
    summary = []
    summary.append("=" * 60)
    summary.append(f"CUDA Device {device_id} Memory Summary")
    summary.append("=" * 60)
    summary.append(f"Device: {props.get('name', 'Unknown')}")
    summary.append(f"Total Memory: {mem.get('total', 0) / 1e9:.2f} GB")
    summary.append(f"Used Memory: {mem.get('used', 0) / 1e9:.2f} GB ({mem.get('used_percent', 0):.1f}%)")
    summary.append(f"Free Memory: {mem.get('free', 0) / 1e9:.2f} GB")
    summary.append("=" * 60)
    
    return "\n".join(summary)

def print_memory_summary(device_id=0):
    """
    Print memory summary for device
    
    Args:
        device_id: Device index (default: 0)
    
    Example:
        >>> import pysml
        >>> if pysml.cuda.is_available():
        ...     pysml.cuda.print_memory_summary(0)
    """
    print(memory_summary(device_id))

__all__ = [
    'is_available',
    'get_device_count',
    'get_device_name',
    'get_device_properties',
    'get_available_devices',
    'set_device',
    'get_current_device',
    'synchronize',
    'init',
    'get_memory_info',
    'reset_device',
    'print_info',
    'print_all_devices',
    'get_cuda_version',
    'get_cudnn_version',
    'empty_cache',
    'memory_summary',
    'print_memory_summary',
    'AVAILABLE',
    'backend',
]