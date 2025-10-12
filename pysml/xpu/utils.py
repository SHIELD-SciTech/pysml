try:
    import dpctl
except:
    dpctl = None

def is_available():
    """Check if an Intel XPU device is available."""
    try:
        devices = dpctl.get_devices("gpu")
        return len(devices) > 0
    except Exception:
        return False


def device_count():
    """Return the number of available Intel GPU/XPU devices."""
    try:
        devices = dpctl.get_devices(backend="level_zero", device_type="gpu")
        return len(devices)
    except Exception:
        return 0


def init():
    """Initialize the Intel XPU backend."""
    if not is_available():
        raise RuntimeError("No Intel XPU devices found or dpctl not available.")
    
    from pysml.tensor import TensorType
    TensorType.set_backend('xpu')
    actualdevices = device_count()
    print(f"[PySML] Intel XPU backend initialized ({actualdevices} device(s) found).")
