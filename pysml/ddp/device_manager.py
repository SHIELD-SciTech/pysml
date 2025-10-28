"""
Device Management — optimized
- Cached device discovery
- Predictable, side-effect-free getters
"""

import pysml
from typing import List, Optional


class DeviceManager:
    _cached_all = None  # small cache so repeated calls are cheap

    def __init__(self):
        if DeviceManager._cached_all is None:
            DeviceManager._cached_all = list(pysml.get_available_devices())
        self._all_devices = DeviceManager._cached_all

        # Stable filtered views
        self._xpu_devices = tuple(d for d in self._all_devices if "xpu" in d)
        self._cuda_devices = tuple(d for d in self._all_devices if "cuda" in d)
        self._cpu_devices = tuple(d for d in self._all_devices if d == "cpu")

    @property
    def all_devices(self) -> List[str]:
        return list(self._all_devices)

    @property
    def xpu_devices(self) -> List[str]:
        return list(self._xpu_devices)

    @property
    def cuda_devices(self) -> List[str]:
        return list(self._cuda_devices)

    @property
    def cpu_devices(self) -> List[str]:
        return list(self._cpu_devices)

    @property
    def num_xpus(self) -> int:
        return len(self._xpu_devices)

    @property
       def num_cudas(self) -> int:
        return len(self._cuda_devices)

    @property
    def has_accelerators(self) -> bool:
        return self.num_xpus > 0 or self.num_cudas > 0

    def _pick_devices(self, device_type: str) -> List[str]:
        if device_type == "auto":
            return (
                list(self._xpu_devices)
                or list(self._cuda_devices)
                or (list(self._cpu_devices) if self._cpu_devices else ["cpu"])
            )
        if device_type == "xpu":
            return list(self._xpu_devices)
        if device_type == "cuda":
            return list(self._cuda_devices)
        if device_type == "cpu":
            return list(self._cpu_devices) or ["cpu"]
        raise ValueError(f"Unknown device type: {device_type}")

    def get_devices(self, device_type: str = "auto", count: Optional[int] = None) -> List[str]:
        devices = self._pick_devices(device_type)
        if not devices:
            raise RuntimeError(f"No devices of type '{device_type}' available")
        if count is not None:
            if count > len(devices):
                raise ValueError(f"Requested {count} devices but only {len(devices)} available")
            devices = devices[:count]
        return devices

    def allocate_for_data_parallel(self, device_type: str = "auto", num_replicas: Optional[int] = None) -> List[str]:
        devices = self.get_devices(device_type)
        if num_replicas is None or num_replicas > len(devices):
            num_replicas = len(devices)
        return devices[:num_replicas]

    def allocate_for_pipeline_parallel(self, device_type: str = "auto", num_stages: Optional[int] = None) -> List[str]:
        devices = self.get_devices(device_type)
        if num_stages is None or num_stages > len(devices):
            num_stages = len(devices)
        return devices[:num_stages]

    def print_summary(self):
        print("=" * 80)
        print("Device Manager Summary")
        print("=" * 80)
        print(f"Total devices: {len(self._all_devices)}")

        print(f"\nXPU devices: {self.num_xpus}")
        for dev in self._xpu_devices:
            print(f"  - {dev}")

        print(f"\nCUDA devices: {self.num_cudas}")
        for dev in self._cuda_devices:
            print(f"  - {dev}")

        print(f"\nCPU devices: {len(self._cpu_devices)}")
        for dev in self._cpu_devices:
            print(f"  - {dev}")
        print("=" * 80)


def get_available_devices(device_type: Optional[str] = None) -> List[str]:
    all_devices = DeviceManager._cached_all or pysml.get_available_devices()
    if device_type is None:
        return list(all_devices)
    return [d for d in all_devices if device_type in d]


def print_device_info():
    dm = DeviceManager()
    dm.print_summary()
    return dm.xpu_devices, dm.cuda_devices
