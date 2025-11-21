from .module import Module
import math


class Loss(Module):
	
	def __init__(self, reduction='mean'):
		super().__init__()
		if reduction not in ['none', 'mean', 'sum']:
			raise ValueError(f"Invalid reduction mode: {reduction}")
		self.reduction = reduction
	
	def _reduce(self, loss):
		"""Apply reduction to loss tensor"""
		if self.reduction == 'none':
			return loss
		elif self.reduction == 'mean':
			return loss.mean()
		elif self.reduction == 'sum':
			return loss.sum()


class MSELoss(Loss):
	
	def __init__(self, reduction='mean'):
		super().__init__(reduction)
	
	def forward(self, input, target):
		from .. import engine
		
		# (input - target)^2
		diff = engine.subtract(input, target)
		squared_diff = engine.square(diff)
		
		return self._reduce(squared_diff)


class L1Loss(Loss):
	
	def __init__(self, reduction='mean'):
		super().__init__(reduction)
	
	def forward(self, input, target):
		from .. import engine
		
		# |input - target|
		diff = engine.subtract(input, target)
		abs_diff = engine.abs(diff)
		
		return self._reduce(abs_diff)


class SmoothL1Loss(Loss):
	
	def __init__(self, reduction='mean', beta=1.0):
		super().__init__(reduction)
		self.beta = beta
	
	def forward(self, input, target):
		from .. import engine
		backend = input._backend
		
		diff = engine.subtract(input, target)
		abs_diff = engine.abs(diff)
		
		# If |diff| < beta: 0.5 * diff^2 / beta
		# Else: |diff| - 0.5 * beta
		
		mask = backend.less(abs_diff.data, self.beta)
		
		from .. import Tensor
		mask_tensor = Tensor.__new__(Tensor)
		mask_tensor._backend = backend
		mask_tensor._dtype = input._dtype
		mask_tensor.device = input.device
		mask_tensor.active_device = input.active_device
		mask_tensor.data = mask
		mask_tensor._requires_grad = False
		mask_tensor._grad = None
		
		# Small diff: 0.5 * diff^2 / beta
		small_loss = engine.divide(
			engine.multiply(0.5, engine.square(diff)),
			self.beta
		)
		
		# Large diff: |diff| - 0.5 * beta
		large_loss = engine.subtract(abs_diff, 0.5 * self.beta)
		
		loss = engine.where(mask_tensor, small_loss, large_loss)
		
		return self._reduce(loss)


class CrossEntropyLoss(Loss):

	def __init__(self, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
		super().__init__(reduction)
		self.weight = weight
		self.ignore_index = ignore_index
		self.label_smoothing = label_smoothing
	
	def forward(self, input, target):
		from .. import engine
		import numpy as np
		backend = input._backend
		
		# input: (batch, num_classes) or (batch, num_classes, d1, d2, ...)
		# target: (batch,) or (batch, d1, d2, ...) with class indices
		
		# Flatten if needed
		original_shape = input.shape
		if len(original_shape) > 2:
			batch_size = original_shape[0]
			num_classes = original_shape[1]
			# Reshape to (batch * spatial, num_classes)
			input_2d = input.reshape((batch_size * np.prod(original_shape[2:]), num_classes))
			target_flat = target.reshape((-1,))
		else:
			input_2d = input
			target_flat = target
			batch_size = original_shape[0]
			num_classes = original_shape[1]
		
		# Log softmax
		log_probs = engine.log_softmax(input_2d, axis=-1)
		
		# Get target probabilities (one-hot with label smoothing)
		target_np = target_flat.numpy()
		
		if self.label_smoothing > 0:
			# Label smoothing: soft targets
			smooth_prob = self.label_smoothing / num_classes
			target_probs = np.full((len(target_np), num_classes), smooth_prob)
			for i in range(len(target_np)):
				if target_np[i] != self.ignore_index:
					target_probs[i, int(target_np[i])] = 1.0 - self.label_smoothing + smooth_prob
		else:
			# One-hot encoding
			target_probs = np.zeros((len(target_np), num_classes))
			for i in range(len(target_np)):
				if target_np[i] != self.ignore_index:
					target_probs[i, int(target_np[i])] = 1.0
		
		from .. import Tensor
		target_probs_tensor = Tensor.__new__(Tensor)
		target_probs_tensor._backend = backend
		target_probs_tensor._dtype = input._dtype
		target_probs_tensor.device = input.device
		target_probs_tensor.active_device = input.active_device
		target_probs_tensor.data = backend.asarray(target_probs)
		target_probs_tensor._requires_grad = False
		target_probs_tensor._grad = None
		
		# Negative log likelihood: -sum(target * log_probs)
		loss = engine.negative(
			engine.sum_with_grad(
				engine.multiply(target_probs_tensor, log_probs),
				axis=-1
			)
		)
		
		# Apply class weights if provided
		if self.weight is not None:
			weight_np = self.weight.numpy()
			weights = np.array([weight_np[int(t)] if t != self.ignore_index else 0.0 
							   for t in target_np])
			
			weight_tensor = Tensor.__new__(Tensor)
			weight_tensor._backend = backend
			weight_tensor._dtype = input._dtype
			weight_tensor.device = input.device
			weight_tensor.active_device = input.active_device
			weight_tensor.data = backend.asarray(weights)
			weight_tensor._requires_grad = False
			weight_tensor._grad = None
			
			loss = engine.multiply(loss, weight_tensor)
		
		return self._reduce(loss)


class NLLLoss(Loss):
	
	def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
		super().__init__(reduction)
		self.weight = weight
		self.ignore_index = ignore_index
	
	def forward(self, input, target):
		from .. import engine
		import numpy as np
		backend = input._backend
		
		# input: (batch, num_classes) - log probabilities
		# target: (batch,) - class indices
		
		batch_size = input.shape[0]
		input_np = input.numpy()
		target_np = target.numpy()
		
		# Gather log probs for target classes
		loss_np = np.zeros(batch_size)
		for i in range(batch_size):
			target_class = int(target_np[i])
			if target_class != self.ignore_index:
				loss_np[i] = -input_np[i, target_class]
				
				# Apply weight if provided
				if self.weight is not None:
					weight_np = self.weight.numpy()
					loss_np[i] *= weight_np[target_class]
		
		from .. import Tensor
		loss = Tensor.__new__(Tensor)
		loss._backend = backend
		loss._dtype = input._dtype
		loss.device = input.device
		loss.active_device = input.active_device
		loss.data = backend.asarray(loss_np)
		loss._requires_grad = input._requires_grad
		loss._grad = None
		
		return self._reduce(loss)


class BCELoss(Loss):
	
	def __init__(self, weight=None, reduction='mean'):
		super().__init__(reduction)
		self.weight = weight
	
	def forward(self, input, target):
		from .. import engine
		
		# BCE = -[y*log(x) + (1-y)*log(1-x)]
		# Numerically stable version
		
		# Clip to avoid log(0)
		input_clipped = engine.clip(input, 1e-7, 1.0 - 1e-7)
		
		# Calculate loss
		term1 = engine.multiply(target, engine.log(input_clipped))
		term2 = engine.multiply(
			engine.subtract(1.0, target),
			engine.log(engine.subtract(1.0, input_clipped))
		)
		
		loss = engine.negative(engine.add(term1, term2))
		
		if self.weight is not None:
			loss = engine.multiply(loss, self.weight)
		
		return self._reduce(loss)


class BCEWithLogitsLoss(Loss):
	
	def __init__(self, weight=None, reduction='mean', pos_weight=None):
		super().__init__(reduction)
		self.weight = weight
		self.pos_weight = pos_weight
	
	def forward(self, input, target):
		from .. import engine
		
		# BCE with logits: log_loss = max(x, 0) - x * y + log(1 + exp(-|x|))
		# More stable than sigmoid + BCE
		
		# max(input, 0)
		max_val = engine.maximum(input, 0)
		
		# input * target
		input_target = engine.multiply(input, target)
		
		# log(1 + exp(-|input|))
		abs_input = engine.abs(input)
		log_term = engine.log(engine.add(1.0, engine.exp(engine.negative(abs_input))))
		
		# Combine
		loss = engine.add(
			engine.subtract(max_val, input_target),
			log_term
		)
		
		# Apply positive class weight if provided
		if self.pos_weight is not None:
			# Multiply positive examples by pos_weight
			pos_loss = engine.multiply(target, loss)
			pos_loss = engine.multiply(pos_loss, self.pos_weight)
			
			neg_loss = engine.multiply(engine.subtract(1.0, target), loss)
			
			loss = engine.add(pos_loss, neg_loss)
		
		if self.weight is not None:
			loss = engine.multiply(loss, self.weight)
		
		return self._reduce(loss)


class KLDivLoss(Loss):
	
	def __init__(self, reduction='mean', log_target=False):
		super().__init__(reduction)
		self.log_target = log_target
	
	def forward(self, input, target):
		from .. import engine
		
		# KL(P||Q) = sum(P * (log(P) - log(Q)))
		# input is log(Q), target is P (or log(P) if log_target=True)
		
		if self.log_target:
			# target is already log(P)
			loss = engine.multiply(
				engine.exp(target),
				engine.subtract(target, input)
			)
		else:
			# target is P
			loss = engine.multiply(
				target,
				engine.subtract(engine.log(engine.clip(target, 1e-7, 1.0)), input)
			)
		
		return self._reduce(loss)


class HingeLoss(Loss):
	
	def __init__(self, margin=1.0, reduction='mean'):
		super().__init__(reduction)
		self.margin = margin
	
	def forward(self, input, target):
		from .. import engine
		
		# Hinge loss: max(0, margin - y * f(x))
		# target should be -1 or 1
		
		# margin - target * input
		loss = engine.subtract(
			self.margin,
			engine.multiply(target, input)
		)
		
		# max(0, loss)
		loss = engine.maximum(loss, 0)
		
		return self._reduce(loss)


class CosineEmbeddingLoss(Loss):
	
	def __init__(self, margin=0.0, reduction='mean'):
		super().__init__(reduction)
		self.margin = margin
	
	def forward(self, input1, input2, target):
		from .. import engine
		
		# Compute cosine similarity
		# cos = (x1 · x2) / (||x1|| * ||x2||)
		
		dot_product = engine.sum_with_grad(
			engine.multiply(input1, input2),
			axis=-1
		)
		
		norm1 = engine.sqrt(engine.sum_with_grad(engine.square(input1), axis=-1))
		norm2 = engine.sqrt(engine.sum_with_grad(engine.square(input2), axis=-1))
		
		cos_sim = engine.divide(dot_product, engine.multiply(norm1, norm2))
		
		# Loss: 1 - cos_sim if target=1, max(0, cos_sim - margin) if target=-1
		backend = input1._backend
		
		# Positive pairs (target = 1): loss = 1 - cos_sim
		pos_loss = engine.subtract(1.0, cos_sim)
		
		# Negative pairs (target = -1): loss = max(0, cos_sim - margin)
		neg_loss = engine.maximum(engine.subtract(cos_sim, self.margin), 0)
		
		# Select based on target
		mask = backend.greater(target.data, 0)
		
		from .. import Tensor
		mask_tensor = Tensor.__new__(Tensor)
		mask_tensor._backend = backend
		mask_tensor._dtype = input1._dtype
		mask_tensor.device = input1.device
		mask_tensor.active_device = input1.active_device
		mask_tensor.data = mask
		mask_tensor._requires_grad = False
		mask_tensor._grad = None
		
		loss = engine.where(mask_tensor, pos_loss, neg_loss)
		
		return self._reduce(loss)


class TripletMarginLoss(Loss):
	
	def __init__(self, margin=1.0, p=2, reduction='mean'):
		super().__init__(reduction)
		self.margin = margin
		self.p = p
	
	def forward(self, anchor, positive, negative):
		from .. import engine
		
		# Distance between anchor and positive
		if self.p == 2:
			# L2 distance
			diff_pos = engine.subtract(anchor, positive)
			dist_pos = engine.sqrt(engine.sum_with_grad(engine.square(diff_pos), axis=-1))
			
			diff_neg = engine.subtract(anchor, negative)
			dist_neg = engine.sqrt(engine.sum_with_grad(engine.square(diff_neg), axis=-1))
		else:
			# L1 distance
			diff_pos = engine.abs(engine.subtract(anchor, positive))
			dist_pos = engine.sum_with_grad(diff_pos, axis=-1)
			
			diff_neg = engine.abs(engine.subtract(anchor, negative))
			dist_neg = engine.sum_with_grad(diff_neg, axis=-1)
		
		# Loss: max(0, dist_pos - dist_neg + margin)
		loss = engine.maximum(
			engine.add(engine.subtract(dist_pos, dist_neg), self.margin),
			0
		)
		
		return self._reduce(loss)


class CTCLoss(Loss):
	
	def __init__(self, blank=0, reduction='mean', zero_infinity=False):
		super().__init__(reduction)
		self.blank = blank
		self.zero_infinity = zero_infinity
	
        def forward(self, log_probs, targets, input_lengths, target_lengths):
                from .. import engine
                from ..tensor import Tensor

                backend = log_probs._backend
                time, batch_size, num_classes = log_probs.shape

                targets_np = targets.numpy() if isinstance(targets, Tensor) else targets
                input_lens = (
                        input_lengths.numpy().tolist()
                        if isinstance(input_lengths, Tensor)
                        else list(input_lengths)
                )
                target_lens = (
                        target_lengths.numpy().tolist()
                        if isinstance(target_lengths, Tensor)
                        else list(target_lengths)
                )

                mask_data = backend.zeros(log_probs.shape, dtype=log_probs.data.dtype)
                valid_mask = backend.ones(batch_size, dtype=log_probs.data.dtype)

                for b in range(batch_size):
                        max_time = min(int(input_lens[b]), time)
                        max_targets = min(int(target_lens[b]), getattr(targets_np, 'shape', (0, 0))[1])

                        if max_time <= 0 or max_targets <= 0:
                                valid_mask[b] = 0.0 if self.zero_infinity else valid_mask[b]
                                continue

                        steps = min(max_time, max_targets)
                        for t in range(steps):
                                cls = int(targets_np[b, t])
                                if cls < 0 or cls >= num_classes:
                                        continue
                                mask_data[t, b, cls] = 1.0

                        if self.zero_infinity and max_time < max_targets:
                                valid_mask[b] = 0.0

                mask = Tensor.__new__(Tensor)
                mask._backend = backend
                mask._dtype = log_probs._dtype
                mask._requires_grad = False
                mask._grad = None
                mask.device = log_probs.device
                mask.active_device = log_probs.active_device
                mask.data = mask_data

                selected = engine.multiply(log_probs, mask)
                log_probs_per_sample = engine.sum_with_grad(selected, axis=(0, 2))
                losses = engine.negative(log_probs_per_sample)

                if self.zero_infinity:
                        weight = Tensor.__new__(Tensor)
                        weight._backend = backend
                        weight._dtype = log_probs._dtype
                        weight._requires_grad = False
                        weight._grad = None
                        weight.device = log_probs.device
                        weight.active_device = log_probs.active_device
                        weight.data = valid_mask
                        losses = engine.multiply(losses, weight)

                return self._reduce(losses)


class FocalLoss(Loss):
	"""Focal Loss for addressing class imbalance"""
	
	def __init__(self, alpha=1, gamma=2, reduction='mean'):
		super().__init__(reduction)
		self.alpha = alpha
		self.gamma = gamma
	
	def forward(self, input, target):
		from .. import engine
		
		# Focal Loss: -alpha * (1-p)^gamma * log(p)
		# where p is the probability of the true class
		
		# Apply softmax to get probabilities
		probs = engine.softmax(input, axis=-1)
		
		# Get probabilities for target classes
		import numpy as np
		backend = input._backend
		
		batch_size = input.shape[0]
		probs_np = probs.numpy()
		target_np = target.numpy()
		
		# Extract target class probabilities
		target_probs = np.zeros(batch_size)
		for i in range(batch_size):
			target_probs[i] = probs_np[i, int(target_np[i])]
		
		from .. import Tensor
		pt = Tensor.__new__(Tensor)
		pt._backend = backend
		pt._dtype = input._dtype
		pt.device = input.device
		pt.active_device = input.active_device
		pt.data = backend.asarray(target_probs)
		pt._requires_grad = input._requires_grad
		pt._grad = None
		
		# Focal loss: -alpha * (1-pt)^gamma * log(pt)
		focal_weight = engine.power(engine.subtract(1.0, pt), self.gamma)
		loss = engine.negative(
			engine.multiply(
				engine.multiply(self.alpha, focal_weight),
				engine.log(engine.clip(pt, 1e-7, 1.0))
			)
		)
		
		return self._reduce(loss)


# Alias for common usage
CrossEntropy = CrossEntropyLoss
MSE = MSELoss
L1 = L1Loss
BCE = BCELoss
NLL = NLLLoss


__all__ = [
	'Loss',
	'MSELoss', 'MSE',
	'L1Loss', 'L1',
	'SmoothL1Loss',
	'CrossEntropyLoss', 'CrossEntropy',
	'NLLLoss', 'NLL',
	'BCELoss', 'BCE',
	'BCEWithLogitsLoss',
	'KLDivLoss',
	'HingeLoss',
	'CosineEmbeddingLoss',
	'TripletMarginLoss',
	'CTCLoss',
	'FocalLoss',
]