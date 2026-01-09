

from __future__ import annotations

from typing import Union, Tuple

from .module import Module
from ..tensor import Tensor

def _pair(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)

def _single(x):
    if isinstance(x, (tuple, list)):
        return x[0]
    return x

class MaxPool1d(Module):

    
    def __init__(
        self,
        kernel_size: int,
        stride: int = None,
        padding: int = 0
    ):
        super().__init__()
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride) if stride is not None else self.kernel_size
        self.padding = _single(padding)
    
    def forward(self, x: Tensor) -> Tensor:
        backend = x._backend
        N, C, L = x.shape
        
        # Pad if necessary
        if self.padding > 0:
            pad_data = backend.full((N, C, L + 2 * self.padding), float('-inf'), dtype=x.data.dtype)
            pad_data[:, :, self.padding:self.padding + L] = x.data
            x_padded = pad_data
        else:
            x_padded = x.data
        
        L_padded = x_padded.shape[2]
        L_out = (L_padded - self.kernel_size) // self.stride + 1
        
        output = backend.zeros((N, C, L_out), dtype=x.data.dtype)
        
        for i in range(L_out):
            start = i * self.stride
            end = start + self.kernel_size
            output[:, :, i] = backend.max(x_padded[:, :, start:end], axis=2)
        
        return x._new_like(output, requires_grad=x._requires_grad)
    
    def extra_repr(self) -> str:
        return f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}'

class MaxPool2d(Module):

    
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = None,
        padding: Union[int, Tuple[int, int]] = 0
    ):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) if stride is not None else self.kernel_size
        self.padding = _pair(padding)
    
    def forward(self, x: Tensor) -> Tensor:
        backend = x._backend
        N, C, H, W = x.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        
        # Pad if necessary
        if pH > 0 or pW > 0:
            pad_data = backend.full((N, C, H + 2 * pH, W + 2 * pW), float('-inf'), dtype=x.data.dtype)
            pad_data[:, :, pH:pH + H, pW:pW + W] = x.data
            x_padded = pad_data
        else:
            x_padded = x.data
        
        H_padded, W_padded = x_padded.shape[2], x_padded.shape[3]
        H_out = (H_padded - kH) // sH + 1
        W_out = (W_padded - kW) // sW + 1
        
        output = backend.zeros((N, C, H_out, W_out), dtype=x.data.dtype)
        
        for h in range(H_out):
            for w in range(W_out):
                h_start = h * sH
                w_start = w * sW
                window = x_padded[:, :, h_start:h_start + kH, w_start:w_start + kW]
                output[:, :, h, w] = backend.max(window.reshape(N, C, -1), axis=2)
        
        return x._new_like(output, requires_grad=x._requires_grad)
    
    def extra_repr(self) -> str:
        return f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}'

class AvgPool1d(Module):

    
    def __init__(
        self,
        kernel_size: int,
        stride: int = None,
        padding: int = 0
    ):
        super().__init__()
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride) if stride is not None else self.kernel_size
        self.padding = _single(padding)
    
    def forward(self, x: Tensor) -> Tensor:
        backend = x._backend
        N, C, L = x.shape
        
        # Pad if necessary
        if self.padding > 0:
            pad_data = backend.zeros((N, C, L + 2 * self.padding), dtype=x.data.dtype)
            pad_data[:, :, self.padding:self.padding + L] = x.data
            x_padded = pad_data
        else:
            x_padded = x.data
        
        L_padded = x_padded.shape[2]
        L_out = (L_padded - self.kernel_size) // self.stride + 1
        
        output = backend.zeros((N, C, L_out), dtype=x.data.dtype)
        
        for i in range(L_out):
            start = i * self.stride
            end = start + self.kernel_size
            output[:, :, i] = backend.mean(x_padded[:, :, start:end], axis=2)
        
        return x._new_like(output, requires_grad=x._requires_grad)
    
    def extra_repr(self) -> str:
        return f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}'

class AvgPool2d(Module):

    
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = None,
        padding: Union[int, Tuple[int, int]] = 0
    ):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) if stride is not None else self.kernel_size
        self.padding = _pair(padding)
    
    def forward(self, x: Tensor) -> Tensor:
        backend = x._backend
        N, C, H, W = x.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        
        # Pad if necessary
        if pH > 0 or pW > 0:
            pad_data = backend.zeros((N, C, H + 2 * pH, W + 2 * pW), dtype=x.data.dtype)
            pad_data[:, :, pH:pH + H, pW:pW + W] = x.data
            x_padded = pad_data
        else:
            x_padded = x.data
        
        H_padded, W_padded = x_padded.shape[2], x_padded.shape[3]
        H_out = (H_padded - kH) // sH + 1
        W_out = (W_padded - kW) // sW + 1
        
        output = backend.zeros((N, C, H_out, W_out), dtype=x.data.dtype)
        
        for h in range(H_out):
            for w in range(W_out):
                h_start = h * sH
                w_start = w * sW
                window = x_padded[:, :, h_start:h_start + kH, w_start:w_start + kW]
                output[:, :, h, w] = backend.mean(window.reshape(N, C, -1), axis=2)
        
        return x._new_like(output, requires_grad=x._requires_grad)
    
    def extra_repr(self) -> str:
        return f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}'

class AdaptiveAvgPool1d(Module):

    
    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size
    
    def forward(self, x: Tensor) -> Tensor:
        backend = x._backend
        N, C, L = x.shape
        
        output = backend.zeros((N, C, self.output_size), dtype=x.data.dtype)
        
        for i in range(self.output_size):
            start = int(i * L / self.output_size)
            end = int((i + 1) * L / self.output_size)
            if end > start:
                output[:, :, i] = backend.mean(x.data[:, :, start:end], axis=2)
            else:
                output[:, :, i] = x.data[:, :, start]
        
        return x._new_like(output, requires_grad=x._requires_grad)
    
    def extra_repr(self) -> str:
        return f'output_size={self.output_size}'

class AdaptiveAvgPool2d(Module):

    
    def __init__(self, output_size: Union[int, Tuple[int, int]]):
        super().__init__()
        self.output_size = _pair(output_size)
    
    def forward(self, x: Tensor) -> Tensor:
        backend = x._backend
        N, C, H, W = x.shape
        oH, oW = self.output_size
        
        output = backend.zeros((N, C, oH, oW), dtype=x.data.dtype)
        
        for h in range(oH):
            for w in range(oW):
                h_start = int(h * H / oH)
                h_end = int((h + 1) * H / oH)
                w_start = int(w * W / oW)
                w_end = int((w + 1) * W / oW)
                
                if h_end > h_start and w_end > w_start:
                    window = x.data[:, :, h_start:h_end, w_start:w_end]
                    output[:, :, h, w] = backend.mean(window.reshape(N, C, -1), axis=2)
                else:
                    output[:, :, h, w] = x.data[:, :, h_start, w_start]
        
        return x._new_like(output, requires_grad=x._requires_grad)
    
    def extra_repr(self) -> str:
        return f'output_size={self.output_size}'

__all__ = ['MaxPool1d', 'MaxPool2d', 'AvgPool1d', 'AvgPool2d', 
           'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d']
