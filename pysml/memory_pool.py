import threading
from collections import defaultdict


class TensorBufferPool:
	def __init__(self):
		self._pools = defaultdict(list)  # (shape, dtype, device) -> [buffers]
		self._lock = threading.Lock()
		self._enabled = True
		
	def enable(self):
		self._enabled = True
		
	def disable(self):
		with self._lock:
			self._enabled = False
			self._pools.clear()
	
	def get_buffer(self, shape, dtype, backend, device):
		if getattr(backend, 'BACKEND_NAME', None) in ('cuda', 'xpu'):
			return None

		if not self._enabled:
			return None
			
		key = (tuple(shape), str(dtype), backend.BACKEND_NAME, str(device))
		
		with self._lock:
			pool = self._pools[key]
			if pool:
				return pool.pop()
		
		return None
	
	def return_buffer(self, buffer, shape, dtype, backend, device):
		if getattr(backend, 'BACKEND_NAME', None) in ('cuda', 'xpu'):
			return

		if not self._enabled:
			return
			
		key = (tuple(shape), str(dtype), backend.BACKEND_NAME, str(device))
		
		with self._lock:
			pool = self._pools[key]
			# Limit pool size to prevent unbounded memory growth
			if len(pool) < 32:  # Max 32 cached buffers per shape/dtype/device
				pool.append(buffer)
	
	def clear(self):
		"""Clear all cached buffers"""
		with self._lock:
			self._pools.clear()
	
	def get_stats(self):
		"""Get statistics about cached buffers"""
		with self._lock:
			return {
				'num_pools': len(self._pools),
				'total_buffers': sum(len(pool) for pool in self._pools.values())
			}


# Global buffer pool instance
_global_buffer_pool = TensorBufferPool()


def get_buffer_pool():
	return _global_buffer_pool


def enable_buffer_pool():
	_global_buffer_pool.enable()


def disable_buffer_pool():
	_global_buffer_pool.disable()


def clear_buffer_pool():
	_global_buffer_pool.clear()


def get_pool_stats():
	return _global_buffer_pool.get_stats()

