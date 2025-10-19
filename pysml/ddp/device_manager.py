"""
Device Management for Distributed Training

Utilities for detecting, managing, and allocating devices across distributed training.
"""

import pysml
from typing import List, Optional


class DeviceManager:
    """
    Central manager for device allocation and management
    
    Example:
        >>> dm = DeviceManager()
        >>> devices = dm.get_devices('xpu', count=2)
        >>> dm.print_summary()
    """
    
    def __init__(self):
        """Initialize device manager"""
        self._all_devices = pysml.get_available_devices()
        self._xpu_devices = [d for d in self._all_devices if 'xpu' in d]
        self._cuda_devices = [d for d in self._all_devices if 'cuda' in d]
        self._cpu_devices = [d for d in self._all_devices if d == 'cpu']
    
    @property
    def all_devices(self) -> List[str]:
        """Get all available devices"""
        return self._all_devices.copy()
    
    @property
    def xpu_devices(self) -> List[str]:
        """Get all XPU devices"""
        return self._xpu_devices.copy()
    
    @property
    def cuda_devices(self) -> List[str]:
        """Get all CUDA devices"""
        return self._cuda_devices.copy()
    
    @property
    def cpu_devices(self) -> List[str]:
        """Get CPU device(s)"""
        return self._cpu_devices.copy()
    
    @property
    def num_xpus(self) -> int:
        """Number of XPU devices"""
        return len(self._xpu_devices)
    
    @property
    def num_cudas(self) -> int:
        """Number of CUDA devices"""
        return len(self._cuda_devices)
    
    @property
    def has_accelerators(self) -> bool:
        """Check if any accelerators (XPU or CUDA) are available"""
        return len(self._xpu_devices) > 0 or len(self._cuda_devices) > 0
    
    def get_devices(self, device_type: str = 'auto', count: Optional[int] = None) -> List[str]:
        """
        Get devices of specified type
        
        Args:
            device_type: 'xpu', 'cuda', 'cpu', or 'auto' (prefers XPU > CUDA > CPU)
            count: Number of devices to return (None = all available)
        
        Returns:
            List of device strings
        
        Example:
            >>> dm = DeviceManager()
            >>> devices = dm.get_devices('xpu', count=2)
            ['xpu:0', 'xpu:1']
        """
        if device_type == 'auto':
            if self.num_xpus > 0:
                devices = self._xpu_devices
            elif self.num_cudas > 0:
                devices = self._cuda_devices
            else:
                devices = self._cpu_devices if self._cpu_devices else ['cpu']
        elif device_type == 'xpu':
            devices = self._xpu_devices
        elif device_type == 'cuda':
            devices = self._cuda_devices
        elif device_type == 'cpu':
            devices = self._cpu_devices if self._cpu_devices else ['cpu']
        else:
            raise ValueError(f"Unknown device type: {device_type}")
        
        if not devices:
            raise RuntimeError(f"No devices of type '{device_type}' available")
        
        if count is not None:
            if count > len(devices):
                raise ValueError(f"Requested {count} devices but only {len(devices)} available")
            devices = devices[:count]
        
        return devices
    
    def allocate_for_data_parallel(self, device_type: str = 'auto', 
                                   num_replicas: Optional[int] = None) -> List[str]:
        """
        Allocate devices for data parallel training
        
        Args:
            device_type: Type of devices to use
            num_replicas: Number of model replicas (defaults to all available)
        
        Returns:
            List of devices for replicas
        """
        devices = self.get_devices(device_type)
        
        if num_replicas is None:
            num_replicas = len(devices)
        
        if num_replicas > len(devices):
            print(f"Warning: Requested {num_replicas} replicas but only {len(devices)} devices available")
            num_replicas = len(devices)
        
        return devices[:num_replicas]
    
    def allocate_for_pipeline_parallel(self, device_type: str = 'auto',
                                       num_stages: Optional[int] = None) -> List[str]:
        """
        Allocate devices for pipeline parallel training
        
        Args:
            device_type: Type of devices to use
            num_stages: Number of pipeline stages (defaults to all available)
        
        Returns:
            List of devices for pipeline stages
        """
        devices = self.get_devices(device_type)
        
        if num_stages is None:
            num_stages = len(devices)
        
        if num_stages > len(devices):
            print(f"Warning: Requested {num_stages} stages but only {len(devices)} devices available")
            num_stages = len(devices)
        
        return devices[:num_stages]
    
    def print_summary(self):
        """Print summary of available devices"""
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
    """
    Get available devices of specified type
    
    Args:
        device_type: 'xpu', 'cuda', 'cpu', or None (returns all)
    
    Returns:
        List of device strings
    
    Example:
        >>> devices = get_available_devices('xpu')
        ['xpu:0', 'xpu:1', 'xpu:2']
    """
    all_devices = pysml.get_available_devices()
    
    if device_type is None:
        return all_devices
    
    return [d for d in all_devices if device_type in d]


def print_device_info():
    """Print detailed information about available devices"""
    dm = DeviceManager()
    dm.print_summary()
    return dm.xpu_devices, dm.cuda_devices