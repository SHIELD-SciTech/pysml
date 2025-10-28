"""
PySML Intel XPU Backend Interface
Complete implementation with all device management functions
"""

from . import backend

# Expose availability check
AVAILABLE = backend.AVAILABLE

def is_available(): return AVAILABLE

def get_device_count():
    if not AVAILABLE: return 0
    try:
        import dpctl
        return len(dpctl.get_devices(device_type="gpu", backend="level_zero"))
    except Exception:
        return 0

def get_device_name(device_id=0):
    if not AVAILABLE: return None
    try:
        import dpctl
        gpus = dpctl.get_devices(device_type="gpu", backend="level_zero")
        if device_id < len(gpus):
            return gpus[device_id].name
        return f"XPU Device {device_id}"
    except Exception:
        return f"XPU Device {device_id}"

def get_device_properties(device_id=0):
    if not AVAILABLE: return {}
    try:
        import dpctl
        gpus = dpctl.get_devices(device_type="gpu", backend="level_zero")
        if device_id < len(gpus):
            d = gpus[device_id]
            return {
                "name": d.name,
                "vendor": d.vendor,
                "driver_version": d.driver_version,
                "backend": d.backend.name,
                "device_type": d.device_type.name,
                "max_compute_units": d.max_compute_units,
                "max_work_item_dims": d.max_work_item_dims,
                "max_work_group_size": d.max_work_group_size,
                "max_num_sub_groups": d.max_num_sub_groups,
                "global_mem_size": d.global_mem_size,
                "local_mem_size": d.local_mem_size,
                "has_aspect_fp64": d.has_aspect_fp64,
                "has_aspect_fp16": d.has_aspect_fp16,
            }
        return {}
    except Exception as e:
        return {"error": str(e)}

def get_available_devices():
    if not AVAILABLE: return []
    try:
        import dpctl
        gpus = dpctl.get_devices(device_type="gpu", backend="level_zero")
        return [f"xpu:{i}" for i in range(len(gpus))]
    except Exception:
        return []

def set_device(device_id=0):
    if not AVAILABLE:
        raise RuntimeError("Intel XPU not available")
    try:
        import dpctl
        gpus = dpctl.get_devices(device_type="gpu", backend="level_zero")
        if device_id < len(gpus):
            dpctl.set_default_queue(dpctl.SyclQueue(gpus[device_id]))
    except Exception as e:
        raise RuntimeError(f"Failed to set device {device_id}: {e}")

def synchronize(): backend.synchronize()

def init():
    if not AVAILABLE:
        raise RuntimeError(
            "Intel XPU not available. Install:\n"
            "  pip install dpnp dpctl\n"
            "or via Intel oneAPI repository."
        )
    try:
        import dpnp, dpctl
        dpnp.zeros(1)
        gpus = dpctl.get_devices(device_type="gpu", backend="level_zero")
        if not gpus:
            raise RuntimeError("No Intel GPU devices found.")
        dpctl.set_default_queue(dpctl.SyclQueue(gpus[0]))
    except Exception as e:
        raise RuntimeError(f"Intel XPU initialization failed: {e}")

def get_memory_info(device_id=0):
    if not AVAILABLE: return {}
    props = get_device_properties(device_id)
    if "global_mem_size" in props:
        return {"total": props["global_mem_size"], "device_id": device_id}
    return {}

def print_info(device_id=0):
    if not AVAILABLE:
        print("Intel XPU not available"); return
    props = get_device_properties(device_id)
    print("="*80)
    print(f"Intel XPU Device {device_id} Information")
    print("="*80)
    if "error" in props:
        print("Error:", props["error"]); return
    for k,v in props.items():
        if k.endswith("_size"):  # bytes → GB
            print(f"{k}: {v/1e9:.2f} GB")
        else:
            print(f"{k}: {v}")
    print("="*80)

def print_all_devices():
    if not AVAILABLE:
        print("Intel XPU not available"); return
    n = get_device_count()
    print(f"\nFound {n} Intel XPU device{'s' if n!=1 else ''}\n")
    for i in range(n):
        print_info(i)
        if i < n-1: print()

__all__ = [
    "is_available","get_device_count","get_device_name","get_device_properties",
    "get_available_devices","set_device","synchronize","init","get_memory_info",
    "print_info","print_all_devices","AVAILABLE","backend",
]
