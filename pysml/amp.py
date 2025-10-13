import numpy as np
from pysml.tensor import Tensor, dtype
import warnings
from typing import Optional, Dict, Any


class GradScaler:
    def __init__(self, 
                 init_scale: float = 2**16,
                 growth_factor: float = 2.0,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 2000,
                 enabled: bool = True):
        self._scale = init_scale
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._enabled = enabled
        
        # Track steps since last scale update
        self._growth_tracker = 0
        self._steps_since_last_scale_update = 0
        
        # Cache for scaled losses
        self._cached_scaled_loss = None
    
    def scale(self, loss: Tensor) -> Tensor:
        if not self._enabled:
            return loss
        
        # Scale loss
        scaled_loss = loss * self._scale
        self._cached_scaled_loss = scaled_loss
        return scaled_loss
    
    def unscale_(self, optimizer):
        if not self._enabled:
            return
        
        inv_scale = 1.0 / self._scale
        
        for param in optimizer.params:
            if param.grad is not None:
                param.grad.data *= inv_scale
    
    def step(self, optimizer):
        if not self._enabled:
            optimizer.step()
            return
        
        # Unscale gradients
        self.unscale_(optimizer)
        
        # Check for inf/nan in gradients
        found_inf = self._check_inf_gradients(optimizer)
        
        if found_inf:
            # Skip optimizer step and reduce scale
            self._scale *= self._backoff_factor
            self._growth_tracker = 0
            warnings.warn(f"Gradient overflow detected. Reducing scale to {self._scale}")
        else:
            # Perform optimizer step
            optimizer.step()
            
            # Update scale tracking
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker = 0
    
    def update(self):
        pass
    
    def _check_inf_gradients(self, optimizer) -> bool:
        for param in optimizer.params:
            if param.grad is not None:
                if np.any(np.isnan(param.grad.data)) or np.any(np.isinf(param.grad.data)):
                    return True
        return False
    
    def get_scale(self) -> float:
        return self._scale
    
    def state_dict(self) -> Dict[str, Any]:
        return {
            'scale': self._scale,
            'growth_tracker': self._growth_tracker,
            '_growth_factor': self._growth_factor,
            '_backoff_factor': self._backoff_factor,
            '_growth_interval': self._growth_interval,
            '_enabled': self._enabled
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._scale = state_dict['scale']
        self._growth_tracker = state_dict['growth_tracker']
        self._growth_factor = state_dict['_growth_factor']
        self._backoff_factor = state_dict['_backoff_factor']
        self._growth_interval = state_dict['_growth_interval']
        self._enabled = state_dict['_enabled']


class autocast:
    def __init__(self, enabled: bool = True, dtype=np.float16):
        self.enabled = enabled
        self.target_dtype = dtype
        self.prev_dtype = None
        self._prev_cache = None
    
    def __enter__(self):
        if self.enabled:
            # Save current default dtype
            from pysml.tensor import STANDARD_DTYPE
            self.prev_dtype = STANDARD_DTYPE
            
            # Set autocast dtype
            import pysml.tensor as tensor_module
            tensor_module.STANDARD_DTYPE = self.target_dtype
            
            # Mark that we're in autocast context
            _amp_state['autocast_enabled'] = True
            _amp_state['autocast_dtype'] = self.target_dtype
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            # Restore previous dtype
            import pysml.tensor as tensor_module
            tensor_module.STANDARD_DTYPE = self.prev_dtype
            
            # Mark that we're out of autocast context
            _amp_state['autocast_enabled'] = False
            _amp_state['autocast_dtype'] = None
        
        return False


# Global state for AMP
_amp_state = {
    'autocast_enabled': False,
    'autocast_dtype': None
}


def is_autocast_enabled() -> bool:
    return _amp_state['autocast_enabled']


def get_autocast_dtype():
    return _amp_state['autocast_dtype']


class AMPContext:
    def __init__(self, enabled: bool = True, dtype=np.float16, init_scale: float = 2**16):
        self.enabled = enabled
        self.scaler = GradScaler(init_scale=init_scale, enabled=enabled)
        self.dtype = dtype
        self._autocast = None
    
    def autocast(self):
        if self._autocast is None:
            self._autocast = autocast(enabled=self.enabled, dtype=self.dtype)
        return self._autocast
    
    def scale(self, loss: Tensor) -> Tensor:
        return self.scaler.scale(loss)
    
    def unscale_(self, optimizer):
        return self.scaler.unscale_(optimizer)
    
    def step(self, optimizer):
        return self.scaler.step(optimizer)
    
    def update(self):
        return self.scaler.update()
    
    def state_dict(self):
        return {
            'enabled': self.enabled,
            'dtype': str(self.dtype),
            'scaler': self.scaler.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.enabled = state_dict['enabled']
        self.scaler.load_state_dict(state_dict['scaler'])


def convert_model_to_fp16(model):
    for param in model.parameters():
        if param.dtype == np.float32:
            param.data = param.data.astype(np.float16)
            param.dtype = np.float16
    
    return model


def convert_model_to_fp32(model):
    for param in model.parameters():
        if param.dtype == np.float16:
            param.data = param.data.astype(np.float32)
            param.dtype = np.float32
    
    return model


class AutocastLinear:
    def __init__(self, in_features, out_features, bias=True):
        from pysml.nn.linear import Linear
        self.linear = Linear(in_features, out_features, bias=bias, dtype=dtype.float32)
    
    def __call__(self, x: Tensor):
        # Save input dtype
        input_dtype = x.dtype
        
        # Cast to fp16 for computation if in autocast
        if is_autocast_enabled():
            x_compute = Tensor(x.data.astype(np.float16), requires_grad=x.requires_grad)
            weight_compute = Tensor(self.linear.weight.data.astype(np.float16))
            
            # Compute in fp16
            y = x_compute @ weight_compute
            
            if self.linear.bias is not None:
                bias_compute = Tensor(self.linear.bias.data.astype(np.float16))
                y = y + bias_compute
            
            # Cast back to fp32 for stability
            y = Tensor(y.data.astype(np.float32), requires_grad=y.requires_grad, _ctx=y._ctx)
        else:
            # Normal fp32 computation
            y = self.linear(x)
        
        return y


# Utility functions for gradient clipping (often used with AMP)
def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0):
    parameters = list(parameters)
    
    if len(parameters) == 0:
        return 0.0
    
    # Compute total norm
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            # Flatten the gradient and compute norm
            grad_flat = p.grad.data.flatten()
            if norm_type == 2.0:
                param_norm = np.sqrt(np.sum(grad_flat ** 2))
            elif norm_type == float('inf'):
                param_norm = np.max(np.abs(grad_flat))
            else:
                param_norm = np.sum(np.abs(grad_flat) ** norm_type) ** (1.0 / norm_type)
            
            total_norm += param_norm ** norm_type
    
    total_norm = total_norm ** (1.0 / norm_type)
    
    # Clip gradients
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data *= clip_coef
    
    return total_norm


def clip_grad_value_(parameters, clip_value: float):
    for p in parameters:
        if p.grad is not None:
            p.grad.data = np.clip(p.grad.data, -clip_value, clip_value)