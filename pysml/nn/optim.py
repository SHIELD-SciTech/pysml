import math
from __future__ import annotations

from typing import Dict, Optional

from pysml.tensor import Tensor


class Optimizer:

        def __init__(self, params, defaults):
                self.defaults = defaults
                self.state = {}
                self._dist_world_size = 1
                self._dist_rank = 0
                self._state_sharding_enabled = False
                self._state_offload_device: Optional[str] = None
                self._owned_param_ids: set[int] = set()
		
		# Convert parameter iterator to list
		if hasattr(params, '__iter__') and not isinstance(params, list):
			params = list(params)
		
		# Store parameters
		self.param_groups = []
		if len(params) == 0:
			raise ValueError("Optimizer got empty parameter list")
		
		if isinstance(params[0], dict):
			# List of parameter groups
			self.param_groups = params
		else:
			# Single parameter group
			self.param_groups = [{'params': params}]
		
		# Apply defaults to all groups
                for group in self.param_groups:
                        for key, value in defaults.items():
                                group.setdefault(key, value)

        def configure_state_sharding(self, *, shard: bool = False, offload_to_cpu: bool = False) -> None:
                """Configure optional optimizer state sharding/offload."""

                self._state_sharding_enabled = shard
                self._state_offload_device = "cpu" if offload_to_cpu else None
                self._owned_param_ids.clear()
                if shard and self._dist_world_size > 0:
                        index = 0
                        for group in self.param_groups:
                                for param in group['params']:
                                        if index % self._dist_world_size == self._dist_rank:
                                                self._owned_param_ids.add(id(param))
                                        index += 1

        def _ensure_state_device(self, param_state: Dict, param) -> None:
                if not param_state:
                        return
                target = getattr(param.data, 'active_device', None)
                for key, value in list(param_state.items()):
                        if isinstance(value, Tensor) and value.active_device != target:
                                param_state[key] = value.to(target)

        def _maybe_offload_state(self, param_state: Dict, param) -> None:
                if not param_state:
                        return
                should_offload = False
                if self._state_sharding_enabled and id(param) not in self._owned_param_ids:
                        should_offload = True
                if self._state_offload_device is not None:
                        should_offload = True
                if not should_offload:
                        return
                device = self._state_offload_device or 'cpu'
                for key, value in list(param_state.items()):
                        if isinstance(value, Tensor) and value.active_device != device:
                                param_state[key] = value.to(device)
	
	def zero_grad(self):
		for group in self.param_groups:
			for p in group['params']:
				if p.grad is not None:
					p.zero_grad()
	
	def step(self):
		raise NotImplementedError("Optimizer subclass must implement step()")
	
	def state_dict(self):
		return {
			'state': self.state,
			'param_groups': self.param_groups
		}
	
	def load_state_dict(self, state_dict):
		self.state = state_dict['state']
		self.param_groups = state_dict['param_groups']
	
	def add_param_group(self, param_group):
		# Apply defaults
		for key, value in self.defaults.items():
			param_group.setdefault(key, value)
		
		self.param_groups.append(param_group)


class SGD(Optimizer):
	def __init__(self, params, lr=0.01, momentum=0.0, dampening=0.0,
				 weight_decay=0.0, nesterov=False):
		if lr < 0.0:
			raise ValueError(f"Invalid learning rate: {lr}")
		if momentum < 0.0:
			raise ValueError(f"Invalid momentum value: {momentum}")
		if weight_decay < 0.0:
			raise ValueError(f"Invalid weight_decay value: {weight_decay}")
		
		defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
					   weight_decay=weight_decay, nesterov=nesterov)
		
		if nesterov and (momentum <= 0 or dampening != 0):
			raise ValueError("Nesterov momentum requires a momentum and zero dampening")
		
		super().__init__(params, defaults)
	
	def step(self):
		for group in self.param_groups:
			weight_decay = group['weight_decay']
			momentum = group['momentum']
			dampening = group['dampening']
			nesterov = group['nesterov']
			lr = group['lr']
			
			for p in group['params']:
				if p.grad is None:
					continue
				
				d_p = p.grad
				
				# Add weight decay
				if weight_decay != 0:
					from .. import engine
					d_p = engine.add(d_p, engine.multiply(p.data, weight_decay))
				
                                # Apply momentum
                                if momentum != 0:
                                        param_state = self.state.get(id(p), {})
                                        self._ensure_state_device(param_state, p)

                                        if 'momentum_buffer' not in param_state:
                                                from .. import Tensor
                                                buf = Tensor.__new__(Tensor)
                                                buf._backend = p.data._backend
						buf._dtype = p.data._dtype
						buf.device = p.data.device
						buf.active_device = p.data.active_device
						buf.data = p.data._backend.copy(d_p.data)
						buf._requires_grad = False
						buf._grad = None
						param_state['momentum_buffer'] = buf
					else:
						buf = param_state['momentum_buffer']
						from .. import engine
						# buf = momentum * buf + (1 - dampening) * d_p
                                                buf.data = engine.add(
                                                        engine.multiply(buf, momentum),
                                                        engine.multiply(d_p, 1 - dampening)
                                                ).data

					if nesterov:
						from .. import engine
						d_p = engine.add(d_p, engine.multiply(buf, momentum))
                                        else:
                                                d_p = buf

                                        self.state[id(p)] = param_state
                                        self._maybe_offload_state(param_state, p)
				
				# Update parameters
				from .. import engine
				p.data.data = engine.subtract(p.data, engine.multiply(d_p, lr)).data


class Adam(Optimizer):
	def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
				 weight_decay=0.0, amsgrad=False):
		if lr < 0.0:
			raise ValueError(f"Invalid learning rate: {lr}")
		if eps < 0.0:
			raise ValueError(f"Invalid epsilon value: {eps}")
		if not 0.0 <= betas[0] < 1.0:
			raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
		if not 0.0 <= betas[1] < 1.0:
			raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
		if weight_decay < 0.0:
			raise ValueError(f"Invalid weight_decay value: {weight_decay}")
		
		defaults = dict(lr=lr, betas=betas, eps=eps,
					   weight_decay=weight_decay, amsgrad=amsgrad)
		super().__init__(params, defaults)
	
	def step(self):
		for group in self.param_groups:
			beta1, beta2 = group['betas']
			lr = group['lr']
			eps = group['eps']
			weight_decay = group['weight_decay']
			amsgrad = group['amsgrad']
			
			for p in group['params']:
				if p.grad is None:
					continue
				
				grad = p.grad
				
                                # Initialize state
                                param_state = self.state.get(id(p), {})
                                self._ensure_state_device(param_state, p)
				
				if len(param_state) == 0:
					param_state['step'] = 0
					
					# Exponential moving average of gradient values
					from .. import Tensor
					m = Tensor.__new__(Tensor)
					m._backend = p.data._backend
					m._dtype = p.data._dtype
					m.device = p.data.device
					m.active_device = p.data.active_device
					m.data = p.data._backend.zeros_like(p.data.data)
					m._requires_grad = False
					m._grad = None
					param_state['exp_avg'] = m
					
					# Exponential moving average of squared gradient values
					v = Tensor.__new__(Tensor)
					v._backend = p.data._backend
					v._dtype = p.data._dtype
					v.device = p.data.device
					v.active_device = p.data.active_device
					v.data = p.data._backend.zeros_like(p.data.data)
					v._requires_grad = False
					v._grad = None
					param_state['exp_avg_sq'] = v
					
					if amsgrad:
						# Max of squared gradients
						v_max = Tensor.__new__(Tensor)
						v_max._backend = p.data._backend
						v_max._dtype = p.data._dtype
						v_max.device = p.data.device
						v_max.active_device = p.data.active_device
						v_max.data = p.data._backend.zeros_like(p.data.data)
						v_max._requires_grad = False
						v_max._grad = None
						param_state['max_exp_avg_sq'] = v_max
				
				exp_avg = param_state['exp_avg']
				exp_avg_sq = param_state['exp_avg_sq']
				
				if amsgrad:
					max_exp_avg_sq = param_state['max_exp_avg_sq']
				
				param_state['step'] += 1
				step = param_state['step']
				
				# Add weight decay (L2 regularization)
				if weight_decay != 0:
					from .. import engine
					grad = engine.add(grad, engine.multiply(p.data, weight_decay))
				
				# Update biased first moment estimate
				# m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
				from .. import engine
				exp_avg.data = engine.add(
					engine.multiply(exp_avg, beta1),
					engine.multiply(grad, 1 - beta1)
				).data
				
				# Update biased second raw moment estimate
				# v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
				exp_avg_sq.data = engine.add(
					engine.multiply(exp_avg_sq, beta2),
					engine.multiply(engine.square(grad), 1 - beta2)
				).data
				
				if amsgrad:
					# Use max of v_t for denominator
					backend = p.data._backend
					max_exp_avg_sq.data = backend.maximum(max_exp_avg_sq.data, exp_avg_sq.data)
					denom_data = max_exp_avg_sq.data
				else:
					denom_data = exp_avg_sq.data
				
				# Bias correction
				bias_correction1 = 1 - beta1 ** step
				bias_correction2 = 1 - beta2 ** step
				
				# Compute step size
				step_size = lr / bias_correction1
				
				# Create denominator tensor
				from .. import Tensor
				denom = Tensor.__new__(Tensor)
				denom._backend = p.data._backend
				denom._dtype = p.data._dtype
				denom.device = p.data.device
				denom.active_device = p.data.active_device
				denom.data = denom_data
				denom._requires_grad = False
				denom._grad = None
				
				# Compute update: -step_size * m_t / (sqrt(v_t / bias_correction2) + eps)
				backend = p.data._backend
				denom_corrected = engine.add(
					engine.sqrt(engine.divide(denom, bias_correction2)),
					eps
				)
				
                                update = engine.divide(exp_avg, denom_corrected)
                                p.data.data = engine.subtract(p.data, engine.multiply(update, step_size)).data

                                self.state[id(p)] = param_state
                                self._maybe_offload_state(param_state, p)
                                self._maybe_offload_state(param_state, p)


class AdamW(Optimizer):
	def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
				 weight_decay=0.01, amsgrad=False):
		if lr < 0.0:
			raise ValueError(f"Invalid learning rate: {lr}")
		if eps < 0.0:
			raise ValueError(f"Invalid epsilon value: {eps}")
		if not 0.0 <= betas[0] < 1.0:
			raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
		if not 0.0 <= betas[1] < 1.0:
			raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
		if weight_decay < 0.0:
			raise ValueError(f"Invalid weight_decay value: {weight_decay}")
		
		defaults = dict(lr=lr, betas=betas, eps=eps,
					   weight_decay=weight_decay, amsgrad=amsgrad)
		super().__init__(params, defaults)
	
	def step(self):
		for group in self.param_groups:
			beta1, beta2 = group['betas']
			lr = group['lr']
			eps = group['eps']
			weight_decay = group['weight_decay']
			amsgrad = group['amsgrad']
			
			for p in group['params']:
				if p.grad is None:
					continue
				
				grad = p.grad
				
                                # Initialize state
                                param_state = self.state.get(id(p), {})
                                self._ensure_state_device(param_state, p)

                                if len(param_state) == 0:
					param_state['step'] = 0
					
					# Exponential moving average of gradient values
					from .. import Tensor
					m = Tensor.__new__(Tensor)
					m._backend = p.data._backend
					m._dtype = p.data._dtype
					m.device = p.data.device
					m.active_device = p.data.active_device
					m.data = p.data._backend.zeros_like(p.data.data)
					m._requires_grad = False
					m._grad = None
					param_state['exp_avg'] = m
					
					# Exponential moving average of squared gradient values
					v = Tensor.__new__(Tensor)
					v._backend = p.data._backend
					v._dtype = p.data._dtype
					v.device = p.data.device
					v.active_device = p.data.active_device
					v.data = p.data._backend.zeros_like(p.data.data)
					v._requires_grad = False
					v._grad = None
					param_state['exp_avg_sq'] = v
					
					if amsgrad:
						# Max of squared gradients
						v_max = Tensor.__new__(Tensor)
						v_max._backend = p.data._backend
						v_max._dtype = p.data._dtype
						v_max.device = p.data.device
						v_max.active_device = p.data.active_device
						v_max.data = p.data._backend.zeros_like(p.data.data)
						v_max._requires_grad = False
						v_max._grad = None
						param_state['max_exp_avg_sq'] = v_max
				
				exp_avg = param_state['exp_avg']
				exp_avg_sq = param_state['exp_avg_sq']
				
				if amsgrad:
					max_exp_avg_sq = param_state['max_exp_avg_sq']
				
				param_state['step'] += 1
				step = param_state['step']
				
				# Decoupled weight decay (AdamW)
				# Apply weight decay directly to parameters, not to gradients
				if weight_decay != 0:
					from .. import engine
					p.data.data = engine.multiply(p.data, 1 - lr * weight_decay).data
				
				# Update biased first moment estimate
				from .. import engine
				exp_avg.data = engine.add(
					engine.multiply(exp_avg, beta1),
					engine.multiply(grad, 1 - beta1)
				).data
				
				# Update biased second raw moment estimate
				exp_avg_sq.data = engine.add(
					engine.multiply(exp_avg_sq, beta2),
					engine.multiply(engine.square(grad), 1 - beta2)
				).data
				
				if amsgrad:
					# Use max of v_t for denominator
					backend = p.data._backend
					max_exp_avg_sq.data = backend.maximum(max_exp_avg_sq.data, exp_avg_sq.data)
					denom_data = max_exp_avg_sq.data
				else:
					denom_data = exp_avg_sq.data
				
				# Bias correction
				bias_correction1 = 1 - beta1 ** step
				bias_correction2 = 1 - beta2 ** step
				
				# Compute step size
				step_size = lr / bias_correction1
				
				# Create denominator tensor
				from .. import Tensor
				denom = Tensor.__new__(Tensor)
				denom._backend = p.data._backend
				denom._dtype = p.data._dtype
				denom.device = p.data.device
				denom.active_device = p.data.active_device
				denom.data = denom_data
				denom._requires_grad = False
				denom._grad = None
				
				# Compute update
				denom_corrected = engine.add(
					engine.sqrt(engine.divide(denom, bias_correction2)),
					eps
				)
				
				update = engine.divide(exp_avg, denom_corrected)
				p.data.data = engine.subtract(p.data, engine.multiply(update, step_size)).data
				
				self.state[id(p)] = param_state


class RMSprop(Optimizer):
	def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8,
				 weight_decay=0.0, momentum=0.0, centered=False):
		if lr < 0.0:
			raise ValueError(f"Invalid learning rate: {lr}")
		if eps < 0.0:
			raise ValueError(f"Invalid epsilon value: {eps}")
		if momentum < 0.0:
			raise ValueError(f"Invalid momentum value: {momentum}")
		if alpha < 0.0:
			raise ValueError(f"Invalid alpha value: {alpha}")
		if weight_decay < 0.0:
			raise ValueError(f"Invalid weight_decay value: {weight_decay}")
		
		defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay,
					   momentum=momentum, centered=centered)
		super().__init__(params, defaults)
	
	def step(self):
		for group in self.param_groups:
			alpha = group['alpha']
			lr = group['lr']
			eps = group['eps']
			weight_decay = group['weight_decay']
			momentum = group['momentum']
			centered = group['centered']
			
			for p in group['params']:
				if p.grad is None:
					continue
				
				grad = p.grad
				
				# Initialize state
				param_state = self.state.get(id(p), {})
				
				if len(param_state) == 0:
					param_state['step'] = 0
					
					# Running average of squared gradients
					from .. import Tensor
					v = Tensor.__new__(Tensor)
					v._backend = p.data._backend
					v._dtype = p.data._dtype
					v.device = p.data.device
					v.active_device = p.data.active_device
					v.data = p.data._backend.zeros_like(p.data.data)
					v._requires_grad = False
					v._grad = None
					param_state['square_avg'] = v
					
					if momentum > 0:
						# Momentum buffer
						buf = Tensor.__new__(Tensor)
						buf._backend = p.data._backend
						buf._dtype = p.data._dtype
						buf.device = p.data.device
						buf.active_device = p.data.active_device
						buf.data = p.data._backend.zeros_like(p.data.data)
						buf._requires_grad = False
						buf._grad = None
						param_state['momentum_buffer'] = buf
					
					if centered:
						# Running average of gradients
						grad_avg = Tensor.__new__(Tensor)
						grad_avg._backend = p.data._backend
						grad_avg._dtype = p.data._dtype
						grad_avg.device = p.data.device
						grad_avg.active_device = p.data.active_device
						grad_avg.data = p.data._backend.zeros_like(p.data.data)
						grad_avg._requires_grad = False
						grad_avg._grad = None
						param_state['grad_avg'] = grad_avg
				
				square_avg = param_state['square_avg']
				param_state['step'] += 1
				
				# Add weight decay
				if weight_decay != 0:
					from .. import engine
					grad = engine.add(grad, engine.multiply(p.data, weight_decay))
				
				# Update running average of squared gradients
				from .. import engine
				square_avg.data = engine.add(
					engine.multiply(square_avg, alpha),
					engine.multiply(engine.square(grad), 1 - alpha)
				).data
				
				if centered:
					grad_avg = param_state['grad_avg']
					# Update running average of gradients
					grad_avg.data = engine.add(
						engine.multiply(grad_avg, alpha),
						engine.multiply(grad, 1 - alpha)
					).data
					# Compute centered variance
					avg = engine.subtract(square_avg, engine.square(grad_avg))
				else:
					avg = square_avg
				
				# Compute update
				if momentum > 0:
					buf = param_state['momentum_buffer']
					# buf = momentum * buf + grad / (sqrt(avg) + eps)
					update = engine.divide(grad, engine.add(engine.sqrt(avg), eps))
					buf.data = engine.add(
						engine.multiply(buf, momentum),
						update
					).data
					p.data.data = engine.subtract(p.data, engine.multiply(buf, lr)).data
				else:
					# p = p - lr * grad / (sqrt(avg) + eps)
					update = engine.divide(grad, engine.add(engine.sqrt(avg), eps))
					p.data.data = engine.subtract(p.data, engine.multiply(update, lr)).data
				
				self.state[id(p)] = param_state


__all__ = [
	'Optimizer',
	'SGD',
	'Adam',
	'AdamW',
	'RMSprop',
]