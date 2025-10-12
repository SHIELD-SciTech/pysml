import sys
try:
    import cupy
except:
    cupy = None


def is_available():
    try:
        import cupy as cp
        _ = cp.cuda.runtime.getDeviceCount()
        return True
    except ImportError:
        return False
    except Exception:
        return False

def device_count():
    if not is_available():
        return 0
    
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount()
    except Exception:
        return 0

def init():
    if not is_available():
        raise RuntimeError("CUDA not available. Make sure CUDA toolkit is installed and CuPy is installed correctly.")
    
    try:
        import cupy as cp
        test = cp.zeros(1)
        del test
        
        cp.cuda.Device(0).use()
        
        from pysml.tensor import TensorType
        TensorType.set_backend('cuda')
        
        print(f"[PySML] CUDA initialized successfully ({device_count()} device(s) found)")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize CUDA: {e}\n"
                         f"Make sure CUDA Toolkit is installed and nvrtc64_*.dll is in PATH")


def get_device_name(device_id=0):
    if not is_available():
        return "No CUDA device"
    
    try:
        import cupy as cp
        with cp.cuda.Device(device_id):
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            return props['name'].decode('utf-8')
    except Exception:
        return f"CUDA Device {device_id}"

