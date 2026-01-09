

from __future__ import annotations

import math
from typing import Union, Tuple

from .module import Module, Parameter
from ..tensor import Tensor
from ..dtype import bf16
from .. import engine

def _pair(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)

def _single(x):
    if isinstance(x, (tuple, list)):
        return x[0]
    return x

class Conv1d(Module):

    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        dtype=None,
        device: str = 'cpu'
    ):
        super().__init__()
        
        if in_channels % groups != 0:
            raise ValueError(f'in_channels ({in_channels}) must be divisible by groups ({groups})')
        if out_channels % groups != 0:
            raise ValueError(f'out_channels ({out_channels}) must be divisible by groups ({groups})')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.dilation = _single(dilation)
        self.groups = groups
        
        if dtype is None:
            dtype = bf16()
        
        # Initialize weight: (out_channels, in_channels/groups, kernel_size)
        k = groups / (in_channels * self.kernel_size)
        bound = math.sqrt(k)
        
        from ..cpu import backend as cpu_backend
        weight_shape = (out_channels, in_channels // groups, self.kernel_size)
        weight_data = cpu_backend.uniform(-bound, bound, weight_shape, dtype=None, device=None)
        self.weight = Parameter(Tensor(weight_data, dtype=dtype, device=device, requires_grad=True))
        
        if bias:
            bias_data = cpu_backend.uniform(-bound, bound, (out_channels,), dtype=None, device=None)
            self.bias = Parameter(Tensor(bias_data, dtype=dtype, device=device, requires_grad=True))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: Tensor) -> Tensor:

        backend = x._backend
        
        N, C_in, L = x.shape
        
        # Calculate output length
        L_out = (L + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        
        # Pad input if necessary
        if self.padding > 0:
            pad_data = backend.zeros((N, C_in, L + 2 * self.padding), dtype=x.data.dtype)
            pad_data[:, :, self.padding:self.padding + L] = x.data
            x_padded = pad_data
        else:
            x_padded = x.data
        
        # Use im2col-style unfolding for efficient convolution
        # Unfold input: extract patches
        patches = []
        for i in range(L_out):
            start = i * self.stride
            end = start + self.kernel_size * self.dilation
            indices = list(range(start, end, self.dilation))
            patch = x_padded[:, :, indices]  # (N, C_in, kernel_size)
            patches.append(patch.reshape(N, -1))  # (N, C_in * kernel_size)
        
        # Stack patches: (N, L_out, C_in * kernel_size)
        unfolded = backend.stack(patches, axis=1)
        
        # Reshape weight: (out_channels, C_in * kernel_size)
        weight_flat = self.weight.data.reshape(self.out_channels, -1)
        
        # Convolution as matrix multiplication
        # (N, L_out, C_in * K) @ (C_in * K, out_channels) -> (N, L_out, out_channels)
        output = backend.matmul(unfolded, weight_flat.T)
        
        # Transpose to (N, out_channels, L_out)
        output = backend.transpose(output, (0, 2, 1))
        
        if self.bias is not None:
            output = output + self.bias.data.reshape(1, -1, 1)
        
        return x._new_like(output, requires_grad=x._requires_grad or self.weight._requires_grad)
    
    def extra_repr(self) -> str:
        return (f'{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, '
                f'stride={self.stride}, padding={self.padding}')

class Conv2d(Module):

    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        dtype=None,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        
        if dtype is None:
            dtype = bf16()
        
        # Initialize weight: (out_channels, in_channels/groups, kH, kW)
        k = groups / (in_channels * self.kernel_size[0] * self.kernel_size[1])
        bound = math.sqrt(k)
        
        from ..cpu import backend as cpu_backend
        weight_shape = (out_channels, in_channels // groups, self.kernel_size[0], self.kernel_size[1])
        weight_data = cpu_backend.uniform(-bound, bound, weight_shape, dtype=None, device=None)
        self.weight = Parameter(Tensor(weight_data, dtype=dtype, device=device, requires_grad=True))
        
        if bias:
            bias_data = cpu_backend.uniform(-bound, bound, (out_channels,), dtype=None, device=None)
            self.bias = Parameter(Tensor(bias_data, dtype=dtype, device=device, requires_grad=True))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: Tensor) -> Tensor:

        backend = x._backend
        
        N, C_in, H, W = x.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        dH, dW = self.dilation
        
        # Calculate output dimensions
        H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        
        # Pad input if necessary
        if pH > 0 or pW > 0:
            pad_data = backend.zeros((N, C_in, H + 2 * pH, W + 2 * pW), dtype=x.data.dtype)
            pad_data[:, :, pH:pH + H, pW:pW + W] = x.data
            x_padded = pad_data
        else:
            x_padded = x.data
        
        # im2col: unfold input into columns
        # Output shape: (N, C_in * kH * kW, H_out * W_out)
        cols = []
        for h in range(H_out):
            for w in range(W_out):
                h_start = h * sH
                w_start = w * sW
                
                # Extract patch with dilation
                patch_indices_h = [h_start + i * dH for i in range(kH)]
                patch_indices_w = [w_start + j * dW for j in range(kW)]
                
                # Gather patch: (N, C_in, kH, kW)
                patch = x_padded[:, :, patch_indices_h[0]:patch_indices_h[-1]+1:dH, 
                                 patch_indices_w[0]:patch_indices_w[-1]+1:dW]
                cols.append(patch.reshape(N, -1))  # (N, C_in * kH * kW)
        
        # Stack: (N, H_out * W_out, C_in * kH * kW)
        col_matrix = backend.stack(cols, axis=1)
        
        # Reshape weight: (out_channels, C_in * kH * kW)
        weight_flat = self.weight.data.reshape(self.out_channels, -1)
        
        # Convolution as matrix multiplication
        # (N, H_out * W_out, C_in * kH * kW) @ (C_in * kH * kW, out_channels)
        output = backend.matmul(col_matrix, weight_flat.T)  # (N, H_out * W_out, out_channels)
        
        # Reshape to (N, out_channels, H_out, W_out)
        output = backend.transpose(output, (0, 2, 1))
        output = output.reshape(N, self.out_channels, H_out, W_out)
        
        if self.bias is not None:
            output = output + self.bias.data.reshape(1, -1, 1, 1)
        
        return x._new_like(output, requires_grad=x._requires_grad or self.weight._requires_grad)
    
    def extra_repr(self) -> str:
        return (f'{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, '
                f'stride={self.stride}, padding={self.padding}')

class ConvTranspose1d(Module):

    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        dtype=None,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.output_padding = _single(output_padding)
        self.groups = groups
        self.dilation = _single(dilation)
        
        if dtype is None:
            dtype = bf16()
        
        k = groups / (in_channels * self.kernel_size)
        bound = math.sqrt(k)
        
        from ..cpu import backend as cpu_backend
        weight_shape = (in_channels, out_channels // groups, self.kernel_size)
        weight_data = cpu_backend.uniform(-bound, bound, weight_shape, dtype=None, device=None)
        self.weight = Parameter(Tensor(weight_data, dtype=dtype, device=device, requires_grad=True))
        
        if bias:
            bias_data = cpu_backend.zeros((out_channels,))
            self.bias = Parameter(Tensor(bias_data, dtype=dtype, device=device, requires_grad=True))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: Tensor) -> Tensor:
        backend = x._backend
        
        N, C_in, L = x.shape
        
        # Calculate output length
        L_out = (L - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + self.output_padding + 1
        
        # Initialize output
        output = backend.zeros((N, self.out_channels, L_out), dtype=x.data.dtype)
        
        # Transpose convolution is like scattering input values through the kernel
        weight = self.weight.data  # (in_channels, out_channels/groups, kernel_size)
        
        for i in range(L):
            for k in range(self.kernel_size):
                out_pos = i * self.stride + k * self.dilation - self.padding
                if 0 <= out_pos < L_out:
                    # x[:, :, i] @ weight[:, :, k] -> output[:, :, out_pos]
                    contrib = backend.matmul(
                        x.data[:, :, i:i+1].transpose(0, 2, 1),  # (N, 1, C_in)
                        weight[:, :, k]  # (C_in, C_out/groups)
                    )  # (N, 1, C_out/groups)
                    output[:, :, out_pos] += contrib.squeeze(1)
        
        if self.bias is not None:
            output = output + self.bias.data.reshape(1, -1, 1)
        
        return x._new_like(output, requires_grad=x._requires_grad or self.weight._requires_grad)
    
    def extra_repr(self) -> str:
        return (f'{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, '
                f'stride={self.stride}, padding={self.padding}')

class ConvTranspose2d(Module):

    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Union[int, Tuple[int, int]] = 1,
        dtype=None,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.output_padding = _pair(output_padding)
        self.groups = groups
        self.dilation = _pair(dilation)
        
        if dtype is None:
            dtype = bf16()
        
        kH, kW = self.kernel_size
        k = groups / (in_channels * kH * kW)
        bound = math.sqrt(k)
        
        from ..cpu import backend as cpu_backend
        weight_shape = (in_channels, out_channels // groups, kH, kW)
        weight_data = cpu_backend.uniform(-bound, bound, weight_shape, dtype=None, device=None)
        self.weight = Parameter(Tensor(weight_data, dtype=dtype, device=device, requires_grad=True))
        
        if bias:
            bias_data = cpu_backend.zeros((out_channels,))
            self.bias = Parameter(Tensor(bias_data, dtype=dtype, device=device, requires_grad=True))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: Tensor) -> Tensor:
        backend = x._backend
        
        N, C_in, H, W = x.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        opH, opW = self.output_padding
        dH, dW = self.dilation
        
        # Calculate output dimensions
        H_out = (H - 1) * sH - 2 * pH + dH * (kH - 1) + opH + 1
        W_out = (W - 1) * sW - 2 * pW + dW * (kW - 1) + opW + 1
        
        # Initialize output
        output = backend.zeros((N, self.out_channels, H_out, W_out), dtype=x.data.dtype)
        
        weight = self.weight.data  # (in_channels, out_channels/groups, kH, kW)
        
        for h in range(H):
            for w in range(W):
                for kh in range(kH):
                    for kw in range(kW):
                        out_h = h * sH + kh * dH - pH
                        out_w = w * sW + kw * dW - pW
                        if 0 <= out_h < H_out and 0 <= out_w < W_out:
                            # Compute contribution
                            inp = x.data[:, :, h, w]  # (N, C_in)
                            w_slice = weight[:, :, kh, kw]  # (C_in, C_out/groups)
                            contrib = backend.matmul(inp, w_slice)  # (N, C_out)
                            output[:, :, out_h, out_w] += contrib
        
        if self.bias is not None:
            output = output + self.bias.data.reshape(1, -1, 1, 1)
        
        return x._new_like(output, requires_grad=x._requires_grad or self.weight._requires_grad)
    
    def extra_repr(self) -> str:
        return (f'{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, '
                f'stride={self.stride}, padding={self.padding}')

__all__ = ['Conv1d', 'Conv2d', 'ConvTranspose1d', 'ConvTranspose2d']
