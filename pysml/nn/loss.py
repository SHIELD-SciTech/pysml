from .module import Module
from ..autograd import Function, is_grad_enabled
import math
import numpy as np


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


def _log_sum_exp(values):
        if len(values) == 0:
                return -np.inf

        result = values[0]
        for val in values[1:]:
                result = np.logaddexp(result, val)
        return result


def _compute_ctc_loss_and_grad(log_probs, targets_np, input_lens, target_lens, blank, zero_infinity):
        time, batch_size, num_classes = log_probs.shape
        grad = np.zeros_like(log_probs)
        losses = np.zeros(batch_size, dtype=log_probs.dtype)

        for b in range(batch_size):
                T = min(int(input_lens[b]), time)
                L = min(int(target_lens[b]), targets_np.shape[1])

                if T <= 0 or L < 0:
                        losses[b] = 0.0 if zero_infinity else np.inf
                        continue

                target_seq = targets_np[b, :L]
                extended = np.full(2 * L + 1, blank, dtype=np.int64)
                extended[1::2] = target_seq

                S = extended.shape[0]
                alpha = np.full((T, S), -np.inf, dtype=log_probs.dtype)
                beta = np.full((T, S), -np.inf, dtype=log_probs.dtype)

                alpha[0, 0] = log_probs[0, b, blank]
                if S > 1:
                        alpha[0, 1] = log_probs[0, b, extended[1]]

                for t in range(1, T):
                        for s in range(S):
                                candidates = [alpha[t - 1, s]]
                                if s - 1 >= 0:
                                        candidates.append(alpha[t - 1, s - 1])
                                if s - 2 >= 0 and extended[s] != blank and extended[s] != extended[s - 2]:
                                        candidates.append(alpha[t - 1, s - 2])

                                alpha[t, s] = log_probs[t, b, extended[s]] + _log_sum_exp(candidates)

                beta[T - 1, S - 1] = log_probs[T - 1, b, extended[S - 1]]
                if S > 1:
                        beta[T - 1, S - 2] = log_probs[T - 1, b, extended[S - 2]]

                for t in range(T - 2, -1, -1):
                        for s in range(S):
                                candidates = [beta[t + 1, s]]
                                if s + 1 < S:
                                        candidates.append(beta[t + 1, s + 1])
                                if s + 2 < S and extended[s] != blank and extended[s] != extended[s + 2]:
                                        candidates.append(beta[t + 1, s + 2])

                                beta[t, s] = log_probs[t, b, extended[s]] + _log_sum_exp(candidates)

                loglike = alpha[T - 1, S - 1]
                if S > 1:
                        loglike = np.logaddexp(loglike, alpha[T - 1, S - 2])

                loss_val = -loglike
                if zero_infinity and not np.isfinite(loss_val):
                        losses[b] = 0.0
                        continue

                losses[b] = loss_val

                for t in range(T):
                        for s in range(S):
                                prob = alpha[t, s] + beta[t, s] - loglike
                                grad[t, b, extended[s]] -= np.exp(prob)

        return losses, grad


def _backward_ctc(grad_output, log_probs_ref, **metadata):
        grads = []
        input_tensor = log_probs_ref() if log_probs_ref else None

        if input_tensor and input_tensor._requires_grad:
                backend = input_tensor._backend
                grad_template = metadata.get('grad')
                losses_shape = metadata.get('losses_shape', ())

                scale = grad_output.data if hasattr(grad_output, 'data') else grad_output
                scale_arr = backend.asarray(scale)

                if losses_shape and scale_arr.shape == losses_shape:
                        scale_arr = backend.reshape(scale_arr, (1, losses_shape[0], 1))

                grad_scaled = backend.multiply(grad_template, scale_arr)

                grad = input_tensor._new_like(grad_scaled, requires_grad=False)
                grads.append((input_tensor, grad))
        else:
                grads.append(None)

        return grads


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
                from ..tensor import Tensor

                backend = log_probs._backend
                time, batch_size, num_classes = log_probs.shape

                targets_np = targets.numpy() if isinstance(targets, Tensor) else targets
                targets_np = np.asarray(targets_np)

                if targets_np.ndim != 2:
                        raise ValueError(
                                f"CTCLoss expects targets to have shape (batch, max_target_length); got {targets_np.ndim}D"
                        )
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

                losses_data, grad_data = _compute_ctc_loss_and_grad(
                        log_probs.data,
                        targets_np,
                        input_lens,
                        target_lens,
                        self.blank,
                        self.zero_infinity,
                )

                losses = Tensor.__new__(Tensor)
                losses._backend = backend
                losses._dtype = log_probs._dtype
                losses._requires_grad = log_probs._requires_grad
                losses._grad = None
                losses.device = log_probs.device
                losses.active_device = log_probs.active_device
                losses.data = backend.asarray(losses_data)

                if is_grad_enabled() and losses._requires_grad:
                        losses._grad_fn = Function(
                                _backward_ctc,
                                [log_probs],
                                metadata={
                                        'grad': backend.asarray(grad_data),
                                        'losses_shape': losses_data.shape,
                                },
                        )

                return self._reduce(losses)


class FocalLoss(Loss):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__(reduction)
        self.alpha = alpha
        self.gamma = gamma
    
        def forward(self, input, target):
                from .. import engine

                import numpy as np
                # Focal Loss: -alpha * (1-p)^gamma * log(p)
                # where p is the probability of the true class

                original_shape = input.shape
                if len(original_shape) > 2:
                        batch_size = original_shape[0]
                        num_classes = original_shape[1]
                        input_2d = input.reshape((batch_size * np.prod(original_shape[2:]), num_classes))
                        target_flat = target.reshape((-1,))
                else:
                        input_2d = input
                        target_flat = target

                # Apply softmax to get probabilities
                probs = engine.softmax(input_2d, axis=-1)

                # Get probabilities for target classes
                backend = input._backend
                batch_size = input_2d.shape[0]
                probs_np = probs.numpy()
                target_np = target_flat.numpy()

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