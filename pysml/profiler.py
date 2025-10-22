"""
PySML Performance Profiler
Track operation timings and identify bottlenecks
"""

import time
from contextlib import contextmanager
from collections import defaultdict
from typing import Dict, List
import numpy as np


class Profiler:
    """
    Simple profiler for PySML operations
    
    Example:
        >>> profiler = Profiler()
        >>> profiler.enable()
        >>> 
        >>> with profiler.profile('forward_pass'):
        ...     output = model(input)
        >>> 
        >>> with profiler.profile('backward_pass'):
        ...     loss.backward()
        >>> 
        >>> profiler.report()
    """
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.enabled = False
        self.memory_tracking = False
        self.memory_usage = defaultdict(list)
    
    def enable(self):
        """Enable profiling"""
        self.enabled = True
    
    def disable(self):
        """Disable profiling"""
        self.enabled = False
    
    def enable_memory_tracking(self):
        """Enable memory usage tracking"""
        self.memory_tracking = True
    
    @contextmanager
    def profile(self, name: str):
        """
        Profile a code block
        
        Args:
            name: Name of the operation to profile
        
        Example:
            >>> with profiler.profile('matmul'):
            ...     result = x @ y
        """
        if not self.enabled:
            yield
            return
        
        # Track memory before
        mem_before = None
        if self.memory_tracking:
            try:
                import psutil
                process = psutil.Process()
                mem_before = process.memory_info().rss / 1024**2  # MB
            except ImportError:
                pass
        
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.timings[name].append(elapsed)
            
            # Track memory after
            if self.memory_tracking and mem_before is not None:
                try:
                    import psutil
                    process = psutil.Process()
                    mem_after = process.memory_info().rss / 1024**2  # MB
                    self.memory_usage[name].append(mem_after - mem_before)
                except ImportError:
                    pass
    
    def report(self, sort_by: str = 'total', top_k: int = None):
        """
        Print profiling report
        
        Args:
            sort_by: Sort by 'total', 'avg', or 'count'
            top_k: Only show top K operations
        """
        if not self.timings:
            print("No profiling data collected. Use profiler.enable() first.")
            return
        
        print("\n" + "="*100)
        print("Profiling Report")
        print("="*100)
        print(f"{'Operation':<40} {'Count':>10} {'Total (ms)':>15} {'Avg (ms)':>15} {'Min (ms)':>15} {'Max (ms)':>15}")
        print("-"*100)
        
        # Prepare data
        data = []
        for name, times in self.timings.items():
            count = len(times)
            total = sum(times) * 1000  # Convert to ms
            avg = total / count
            min_time = min(times) * 1000
            max_time = max(times) * 1000
            data.append((name, count, total, avg, min_time, max_time))
        
        # Sort
        if sort_by == 'total':
            data.sort(key=lambda x: x[2], reverse=True)
        elif sort_by == 'avg':
            data.sort(key=lambda x: x[3], reverse=True)
        elif sort_by == 'count':
            data.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to top K
        if top_k is not None:
            data = data[:top_k]
        
        # Print
        for name, count, total, avg, min_time, max_time in data:
            print(f"{name:<40} {count:>10} {total:>15.2f} {avg:>15.2f} {min_time:>15.2f} {max_time:>15.2f}")
        
        print("="*100)
        
        # Memory report if available
        if self.memory_tracking and self.memory_usage:
            print("\nMemory Usage Report")
            print("="*100)
            print(f"{'Operation':<40} {'Count':>10} {'Total (MB)':>15} {'Avg (MB)':>15}")
            print("-"*100)
            
            mem_data = []
            for name, mems in self.memory_usage.items():
                count = len(mems)
                total_mem = sum(mems)
                avg_mem = total_mem / count
                mem_data.append((name, count, total_mem, avg_mem))
            
            mem_data.sort(key=lambda x: x[2], reverse=True)
            
            if top_k is not None:
                mem_data = mem_data[:top_k]
            
            for name, count, total_mem, avg_mem in mem_data:
                print(f"{name:<40} {count:>10} {total_mem:>15.2f} {avg_mem:>15.2f}")
            
            print("="*100)
    
    def get_summary(self) -> Dict:
        """
        Get profiling summary as dictionary
        
        Returns:
            Dictionary with profiling statistics
        """
        summary = {}
        for name, times in self.timings.items():
            summary[name] = {
                'count': len(times),
                'total_ms': sum(times) * 1000,
                'avg_ms': (sum(times) / len(times)) * 1000,
                'min_ms': min(times) * 1000,
                'max_ms': max(times) * 1000,
            }
        return summary
    
    def clear(self):
        """Clear profiling data"""
        self.timings.clear()
        self.memory_usage.clear()
    
    def __enter__(self):
        """Context manager support"""
        self.enable()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.disable()
        return False


# Global profiler instance
_profiler = Profiler()

def enable_profiling():
    """Enable global profiler"""
    _profiler.enable()

def disable_profiling():
    """Disable global profiler"""
    _profiler.disable()

def get_profiler() -> Profiler:
    """Get global profiler instance"""
    return _profiler

def profile(name: str):
    """
    Decorator for profiling functions
    
    Example:
        >>> @profile('custom_op')
        ... def my_operation(x, y):
        ...     return x @ y
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with _profiler.profile(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class PerformanceMonitor:
    """
    Advanced performance monitor with statistics
    
    Tracks:
    - Operation timings
    - Memory usage
    - GPU utilization (if available)
    - Throughput metrics
    """
    
    def __init__(self):
        self.profiler = Profiler()
        self.batch_times = []
        self.epoch_times = []
        self.current_epoch_start = None
        self.current_batch_start = None
    
    def start_epoch(self):
        """Mark start of epoch"""
        self.current_epoch_start = time.perf_counter()
    
    def end_epoch(self):
        """Mark end of epoch"""
        if self.current_epoch_start is not None:
            elapsed = time.perf_counter() - self.current_epoch_start
            self.epoch_times.append(elapsed)
            self.current_epoch_start = None
    
    def start_batch(self):
        """Mark start of batch"""
        self.current_batch_start = time.perf_counter()
    
    def end_batch(self):
        """Mark end of batch"""
        if self.current_batch_start is not None:
            elapsed = time.perf_counter() - self.current_batch_start
            self.batch_times.append(elapsed)
            self.current_batch_start = None
    
    def get_throughput(self, batch_size: int) -> Dict:
        """
        Calculate throughput metrics
        
        Args:
            batch_size: Number of samples per batch
        
        Returns:
            Dictionary with throughput stats
        """
        if not self.batch_times:
            return {}
        
        avg_batch_time = np.mean(self.batch_times)
        samples_per_sec = batch_size / avg_batch_time
        
        return {
            'avg_batch_time_ms': avg_batch_time * 1000,
            'samples_per_second': samples_per_sec,
            'batches_per_second': 1.0 / avg_batch_time,
        }
    
    def report_training_stats(self, batch_size: int = None):
        """Print comprehensive training statistics"""
        print("\n" + "="*80)
        print("Training Performance Report")
        print("="*80)
        
        if self.epoch_times:
            print(f"\nEpoch Statistics:")
            print(f"  Total epochs: {len(self.epoch_times)}")
            print(f"  Avg epoch time: {np.mean(self.epoch_times):.2f}s")
            print(f"  Total training time: {sum(self.epoch_times):.2f}s")
        
        if self.batch_times:
            print(f"\nBatch Statistics:")
            print(f"  Total batches: {len(self.batch_times)}")
            print(f"  Avg batch time: {np.mean(self.batch_times)*1000:.2f}ms")
            print(f"  Min batch time: {np.min(self.batch_times)*1000:.2f}ms")
            print(f"  Max batch time: {np.max(self.batch_times)*1000:.2f}ms")
            
            if batch_size:
                throughput = self.get_throughput(batch_size)
                print(f"\nThroughput:")
                print(f"  Samples/sec: {throughput['samples_per_second']:.2f}")
                print(f"  Batches/sec: {throughput['batches_per_second']:.2f}")
        
        print("="*80)
    
    def clear(self):
        """Clear all statistics"""
        self.profiler.clear()
        self.batch_times.clear()
        self.epoch_times.clear()


__all__ = [
    'Profiler',
    'PerformanceMonitor',
    'enable_profiling',
    'disable_profiling',
    'get_profiler',
    'profile',
]