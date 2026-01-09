

from __future__ import annotations

import os
import socket
import pickle
import struct
import threading
import time
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import numpy as np

class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    XPU = "xpu"

@dataclass
class DeviceInfo:

    device_type: DeviceType
    device_id: int
    rank: int
    hostname: str
    port: int

class CommunicationBackend:

    
    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: str = "localhost",
        master_port: int = 29500,
        device_type: str = "cuda",
        device_id: int = 0,
    ):
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.device_type = DeviceType(device_type)
        self.device_id = device_id
        
        self._initialized = False
        self._connections: Dict[int, socket.socket] = {}
        self._server_socket: Optional[socket.socket] = None
        self._listen_thread: Optional[threading.Thread] = None
        self._recv_buffers: Dict[int, List[bytes]] = {i: [] for i in range(world_size)}
        self._recv_lock = threading.Lock()
        self._backend = None
        
        self._init_backend()
    
    def _init_backend(self):

        if self.device_type == DeviceType.CUDA:
            try:
                from ..cuda import backend as cuda_backend
                self._backend = cuda_backend
            except ImportError:
                raise RuntimeError("CUDA backend not available")
        elif self.device_type == DeviceType.XPU:
            try:
                from ..xpu import backend as xpu_backend
                self._backend = xpu_backend
            except ImportError:
                raise RuntimeError("XPU backend not available")
        else:
            from ..cpu import backend as cpu_backend
            self._backend = cpu_backend
    
    def init_process_group(self):

        if self._initialized:
            return
        
        # Start server socket for receiving
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Each rank listens on master_port + rank
        listen_port = self.master_port + self.rank
        self._server_socket.bind(("0.0.0.0", listen_port))
        self._server_socket.listen(self.world_size)
        
        # Start listener thread
        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()
        
        # Connect to all other ranks
        time.sleep(0.5)  # Give servers time to start
        
        for target_rank in range(self.world_size):
            if target_rank == self.rank:
                continue
            
            target_port = self.master_port + target_rank
            max_retries = 10
            
            for retry in range(max_retries):
                try:
                    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    conn.connect((self.master_addr, target_port))
                    # Send our rank
                    conn.sendall(struct.pack("!I", self.rank))
                    self._connections[target_rank] = conn
                    break
                except ConnectionRefusedError:
                    if retry < max_retries - 1:
                        time.sleep(0.5)
                    else:
                        raise RuntimeError(f"Failed to connect to rank {target_rank}")
        
        self._initialized = True
        
        # Barrier to ensure all connections are established
        self.barrier()
    
    def _listen_loop(self):

        while True:
            try:
                conn, addr = self._server_socket.accept()
                # Receive sender rank
                rank_data = conn.recv(4)
                if len(rank_data) == 4:
                    sender_rank = struct.unpack("!I", rank_data)[0]
                    # Start receiver thread for this connection
                    recv_thread = threading.Thread(
                        target=self._recv_loop, 
                        args=(conn, sender_rank),
                        daemon=True
                    )
                    recv_thread.start()
            except Exception:
                break
    
    def _recv_loop(self, conn: socket.socket, sender_rank: int):

        while True:
            try:
                # Receive message length
                len_data = conn.recv(8)
                if len(len_data) < 8:
                    break
                msg_len = struct.unpack("!Q", len_data)[0]
                
                # Receive message data
                data = b""
                while len(data) < msg_len:
                    chunk = conn.recv(min(msg_len - len(data), 65536))
                    if not chunk:
                        break
                    data += chunk
                
                if len(data) == msg_len:
                    with self._recv_lock:
                        self._recv_buffers[sender_rank].append(data)
            except Exception:
                break
    
    def _send_to_rank(self, data: bytes, target_rank: int):

        if target_rank not in self._connections:
            raise RuntimeError(f"Not connected to rank {target_rank}")
        
        conn = self._connections[target_rank]
        # Send length + data
        msg = struct.pack("!Q", len(data)) + data
        conn.sendall(msg)
    
    def _recv_from_rank(self, source_rank: int, timeout: float = 30.0) -> bytes:

        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._recv_lock:
                if self._recv_buffers[source_rank]:
                    return self._recv_buffers[source_rank].pop(0)
            time.sleep(0.001)
        
        raise TimeoutError(f"Timeout waiting for data from rank {source_rank}")
    
    def _tensor_to_bytes(self, tensor) -> bytes:

        # Get numpy array from tensor
        if hasattr(tensor, 'data'):
            data = tensor.data
        else:
            data = tensor
        
        if hasattr(self._backend, 'asnumpy'):
            np_array = self._backend.asnumpy(data)
        else:
            np_array = np.asarray(data)
        
        # Serialize with pickle (includes shape and dtype)
        return pickle.dumps({
            'data': np_array.tobytes(),
            'shape': np_array.shape,
            'dtype': str(np_array.dtype),
        })
    
    def _bytes_to_tensor(self, data: bytes):

        info = pickle.loads(data)
        np_array = np.frombuffer(info['data'], dtype=info['dtype']).reshape(info['shape'])
        
        # Convert to device array
        return self._backend.asarray(np_array)
    
    # ========== Collective Operations ==========
    
    def broadcast(self, tensor, src: int = 0):

        if self.rank == src:
            # Send to all other ranks
            data = self._tensor_to_bytes(tensor)
            for target_rank in range(self.world_size):
                if target_rank != self.rank:
                    self._send_to_rank(data, target_rank)
            return tensor
        else:
            # Receive from src
            data = self._recv_from_rank(src)
            return self._bytes_to_tensor(data)
    
    def all_reduce(self, tensor, op: str = "sum"):

        # Ring all-reduce for efficiency
        # Step 1: Reduce-scatter
        # Step 2: All-gather
        
        # Simple implementation: gather all, reduce locally, broadcast
        # (Not optimal but correct)
        
        if hasattr(tensor, 'data'):
            data = tensor.data
        else:
            data = tensor
        
        # Send our data to rank 0
        if self.rank != 0:
            self._send_to_rank(self._tensor_to_bytes(data), 0)
        
        if self.rank == 0:
            # Collect all tensors
            all_tensors = [data]
            for src_rank in range(1, self.world_size):
                recv_data = self._recv_from_rank(src_rank)
                all_tensors.append(self._bytes_to_tensor(recv_data))
            
            # Reduce
            if op == "sum":
                result = all_tensors[0]
                for t in all_tensors[1:]:
                    result = self._backend.add(result, t)
            elif op == "mean":
                result = all_tensors[0]
                for t in all_tensors[1:]:
                    result = self._backend.add(result, t)
                result = self._backend.divide(result, self.world_size)
            elif op == "max":
                result = all_tensors[0]
                for t in all_tensors[1:]:
                    result = self._backend.maximum(result, t)
            elif op == "min":
                result = all_tensors[0]
                for t in all_tensors[1:]:
                    result = self._backend.minimum(result, t)
            else:
                raise ValueError(f"Unknown reduction op: {op}")
            
            # Broadcast result
            result_bytes = self._tensor_to_bytes(result)
            for target_rank in range(1, self.world_size):
                self._send_to_rank(result_bytes, target_rank)
            
            return result
        else:
            # Receive result from rank 0
            result_data = self._recv_from_rank(0)
            return self._bytes_to_tensor(result_data)
    
    def all_gather(self, tensor) -> List:

        if hasattr(tensor, 'data'):
            data = tensor.data
        else:
            data = tensor
        
        # Send to all, receive from all
        my_data = self._tensor_to_bytes(data)
        
        # Send to all other ranks
        for target_rank in range(self.world_size):
            if target_rank != self.rank:
                self._send_to_rank(my_data, target_rank)
        
        # Receive from all other ranks
        result = [None] * self.world_size
        result[self.rank] = data
        
        for src_rank in range(self.world_size):
            if src_rank != self.rank:
                recv_data = self._recv_from_rank(src_rank)
                result[src_rank] = self._bytes_to_tensor(recv_data)
        
        return result
    
    def reduce_scatter(self, tensor, op: str = "sum"):

        if hasattr(tensor, 'data'):
            data = tensor.data
        else:
            data = tensor
        
        # Split tensor into world_size chunks
        chunk_size = data.shape[0] // self.world_size
        chunks = [data[i * chunk_size:(i + 1) * chunk_size] for i in range(self.world_size)]
        
        # Send chunks to appropriate ranks
        for target_rank in range(self.world_size):
            if target_rank != self.rank:
                self._send_to_rank(self._tensor_to_bytes(chunks[target_rank]), target_rank)
        
        # Receive chunks from other ranks and reduce
        my_chunk = chunks[self.rank]
        
        for src_rank in range(self.world_size):
            if src_rank != self.rank:
                recv_data = self._recv_from_rank(src_rank)
                recv_chunk = self._bytes_to_tensor(recv_data)
                
                if op == "sum":
                    my_chunk = self._backend.add(my_chunk, recv_chunk)
                elif op == "mean":
                    my_chunk = self._backend.add(my_chunk, recv_chunk)
                elif op == "max":
                    my_chunk = self._backend.maximum(my_chunk, recv_chunk)
                elif op == "min":
                    my_chunk = self._backend.minimum(my_chunk, recv_chunk)
        
        if op == "mean":
            my_chunk = self._backend.divide(my_chunk, self.world_size)
        
        return my_chunk
    
    def send(self, tensor, dst: int, tag: int = 0):

        data = self._tensor_to_bytes(tensor)
        self._send_to_rank(data, dst)
    
    def recv(self, src: int, tag: int = 0):

        data = self._recv_from_rank(src)
        return self._bytes_to_tensor(data)
    
    def barrier(self):

        # Simple barrier using all-reduce of a scalar
        dummy = self._backend.asarray([1.0])
        self.all_reduce(dummy, op="sum")
    
    def destroy(self):

        for conn in self._connections.values():
            try:
                conn.close()
            except Exception:
                pass
        
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass
        
        self._initialized = False

# Global process group
_PROCESS_GROUP: Optional[CommunicationBackend] = None

def init_process_group(
    rank: int,
    world_size: int,
    master_addr: str = "localhost",
    master_port: int = 29500,
    device_type: str = "cuda",
    device_id: int = 0,
):

    global _PROCESS_GROUP
    
    _PROCESS_GROUP = CommunicationBackend(
        rank=rank,
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
        device_type=device_type,
        device_id=device_id,
    )
    _PROCESS_GROUP.init_process_group()
    
    return _PROCESS_GROUP

def get_process_group() -> Optional[CommunicationBackend]:

    return _PROCESS_GROUP

def destroy_process_group():

    global _PROCESS_GROUP
    if _PROCESS_GROUP is not None:
        _PROCESS_GROUP.destroy()
        _PROCESS_GROUP = None

def get_rank() -> int:

    if _PROCESS_GROUP is None:
        return 0
    return _PROCESS_GROUP.rank

def get_world_size() -> int:

    if _PROCESS_GROUP is None:
        return 1
    return _PROCESS_GROUP.world_size

def is_initialized() -> bool:

    return _PROCESS_GROUP is not None and _PROCESS_GROUP._initialized

__all__ = [
    'CommunicationBackend',
    'DeviceType',
    'DeviceInfo',
    'init_process_group',
    'get_process_group',
    'destroy_process_group',
    'get_rank',
    'get_world_size',
    'is_initialized',
]
