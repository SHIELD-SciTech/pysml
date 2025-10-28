"""
PySML CUDA Backend Interface
Optimized device control & memory inspection for CuPy backend
"""

from . import backend

AVAILABLE = backend.AVAILABLE

def is_available(): return AVAILABLE

def get_device_count():
    if not AVAILABLE: return 0
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount()
    except Exception:
        return 0

def get_device_name(device_id=0):
    if not AVAILABLE: return None
    try:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        name = props["name"]
        return name.decode() if isinstance(name, bytes) else name
    except Exception:
        return f"CUDA Device {device_id}"

def get_device_properties(device_id=0):
    if not AVAILABLE: return {}
    try:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        return {k: (v.decode() if isinstance(v, bytes) else v) for k, v in props.items()}
    except Exception as e:
        return {"error": str(e)}

def get_available_devices():
    if not AVAILABLE: return []
    try:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        return [f"cuda:{i}" for i in range(n)]
    except Exception:
        return []

def set_device(device_id=0):
    if not AVAILABLE:
        raise RuntimeError("CUDA not available")
    try:
        import cupy as cp
        cp.cuda.Device(device_id).use()
    except Exception as e:
        raise RuntimeError(f"Failed to set CUDA device {device_id}: {e}")

def synchronize(): backend.synchronize()

def init():
    if not AVAILABLE:
        raise RuntimeError(
            "CUDA not available.\n"
            "Install CuPy:\n"
            "  pip install cupy-cuda12x  # For CUDA 12.x\n"
            "  pip install cupy-cuda11x  # For CUDA 11.x"
        )
    try:
        import cupy as cp
        cp.zeros(1)
        if cp.cuda.runtime.getDeviceCount() == 0:
            raise RuntimeError("No CUDA-capable GPUs detected.")
    except Exception as e:
        raise RuntimeError(f"CUDA initialization failed: {e}")

def get_memory_info(device_id=0): return backend.get_memory_info(device_id)
def empty_cache(): backend.empty_cache()

def print_info(device_id=0):
    if not AVAILABLE:
        print("CUDA not available"); return
    props = get_device_properties(device_id)
    if "error" in props:
        print("Error:", props["error"]); return
    print("="*80)
    print(f"CUDA Device {device_id} Information")
    print("="*80)
    for k,v in props.items():
        if k.endswith("Mem"):
            print(f"{k}: {v/1e9:.2f} GB")
        else:
            print(f"{k}: {v}")
    print("="*80)

def print_all_devices():
    if not AVAILABLE:
        print("CUDA not available"); return
    n = get_device_count()
    print(f"\nFound {n} CUDA device{'s' if n!=1 else ''}\n")
    for i in range(n):
        print_info(i)
        if i < n-1: print()

__all__ = [
    "is_available","get_device_count","get_device_name","get_device_properties",
    "get_available_devices","set_device","synchronize","init",
    "get_memory_info","empty_cache","print_info","print_all_devices",
    "AVAILABLE","backend",
]
