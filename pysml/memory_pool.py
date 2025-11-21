import threading
from collections import defaultdict, OrderedDict

DEFAULT_CAPACITY_BYTES = 2 * 1024 * 1024 * 1024  # 2GB total cache budget


class TensorBufferPool:
        def __init__(self, capacity_bytes: int = DEFAULT_CAPACITY_BYTES):
                # Per-key stacks for fast retrieval
                self._pools = defaultdict(list)  # (shape, dtype, device) -> [buffers]
                # Global LRU of all cached buffers for byte-aware eviction.
                self._lru = OrderedDict()  # buffer_id -> (key, buffer, nbytes)
                self._total_bytes = 0
                self._capacity_bytes = capacity_bytes
                self._lock = threading.Lock()
                self._enabled = True
		
	def enable(self):
		self._enabled = True
		
	def disable(self):
		with self._lock:
			self._enabled = False
			self._pools.clear()
	
        def get_buffer(self, shape, dtype, backend, device):
                if not self._enabled:
                        return None

                key = (tuple(shape), str(dtype), backend.BACKEND_NAME, str(device))

                with self._lock:
                        pool = self._pools[key]
                        if pool:
                                buffer = pool.pop()
                                buffer_id = id(buffer)
                                if buffer_id in self._lru:
                                        _, _, nbytes = self._lru.pop(buffer_id)
                                        self._total_bytes -= nbytes
                                return buffer

                return None

        def return_buffer(self, buffer, shape, dtype, backend, device):
                if not self._enabled:
                        return

                nbytes = self._buffer_nbytes(buffer)
                
                key = (tuple(shape), str(dtype), backend.BACKEND_NAME, str(device))

                with self._lock:
                        pool = self._pools[key]
                        pool.append(buffer)

                        buffer_id = id(buffer)
                        self._lru[buffer_id] = (key, buffer, nbytes)
                        self._total_bytes += nbytes

                        # Keep most recently used at the end
                        self._lru.move_to_end(buffer_id)

                        self._evict_if_needed()

        def clear(self):
                """Clear all cached buffers"""
                with self._lock:
                        self._pools.clear()
                        self._lru.clear()
                        self._total_bytes = 0
	
        def get_stats(self):
                """Get statistics about cached buffers"""
                with self._lock:
                        return {
                                'num_pools': len(self._pools),
                                'total_buffers': sum(len(pool) for pool in self._pools.values()),
                                'total_bytes': self._total_bytes,
                                'capacity_bytes': self._capacity_bytes,
                        }

        def _evict_if_needed(self):
                if self._total_bytes <= self._capacity_bytes:
                        return

                bytes_to_free = self._total_bytes - self._capacity_bytes
                freed = 0
                eviction_candidates = []

                # Collect eviction candidates before mutating the pool structures
                for buffer_id, (key, buffer, nbytes) in list(self._lru.items()):
                        eviction_candidates.append((buffer_id, key, buffer, nbytes))
                        freed += nbytes
                        if freed >= bytes_to_free:
                                break

                for buffer_id, key, buffer, nbytes in eviction_candidates:
                        self._lru.pop(buffer_id, None)
                        pool = self._pools.get(key)
                        if pool:
                                try:
                                        pool.remove(buffer)
                                except ValueError:
                                        pass
                        self._total_bytes -= nbytes

        @staticmethod
        def _buffer_nbytes(buffer) -> int:
                return int(getattr(buffer, 'nbytes', 0) or 0)


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

