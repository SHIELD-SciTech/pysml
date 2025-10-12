from pysml.tensor import TensorType
import pysml.operations 
import dpctl
import re

class device:
    def __init__(self, device_str):
        if not isinstance(device_str, str):
            raise ValueError("Device must be a string: 'cpu', 'cuda:<id>', or 'xpu:<id>'")
        if not re.match(r"^(cpu|cuda:\d+|xpu:\d+)$", device_str):
            raise ValueError("Device must be 'cpu', 'cuda:<id>', or 'xpu:<id>'.")
        
        self.new_device = device_str
        self.old_device = TensorType.device
        self.old_backend = TensorType.backend
        self.queue = None # For XPU context

    def __enter__(self):
        TensorType.set_device(self.new_device)
        backend_name = self.new_device.split(':')[0]
        pysml.operations.update_backend(backend_name) 

        if 'xpu' in self.new_device:
            idx = int(self.new_device.split(':')[1])
            level_zero_gpus = dpctl.get_devices(backend="level_zero", device_type="gpu")
            if idx >= len(level_zero_gpus):
                raise ValueError(f"XPU device index {idx} out of range ({len(level_zero_gpus)} devices found).")
            dev = level_zero_gpus[idx]
            self.queue = dpctl.SyclQueue(dev)
            TensorType._current_queue = self.queue

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        TensorType.set_device(self.old_device)
        pysml.operations.update_backend(self.old_backend)
        TensorType.backend = self.old_backend
        self.queue = None
        TensorType._current_queue = None
