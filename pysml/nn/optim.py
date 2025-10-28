"""
PySML Optimizers
v0.4.8-final
Optimized AdamW with optional AMP (GradScaler) integration.
"""

import math
from pysml.engine import amp_enabled, get_scaler, get_backend

# ---------------------------------------------------------------------
# Base Optimizer
# ---------------------------------------------------------------------
class Optimizer:
    """Base optimizer class."""

    def __init__(self, params, lr=1e-3, weight_decay=0.0):
        self.param_groups = [{"params": list(params), "lr": lr, "weight_decay": weight_decay}]
        self.state = {}

    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                if hasattr(p, "grad") and p.grad is not None:
                    if hasattr(p.grad, "fill"):
                        p.grad.fill(0)
                    else:
                        p.grad[:] = 0

    def step(self):
        raise NotImplementedError


# ---------------------------------------------------------------------
# AdamW Optimizer with AMP Support
# ---------------------------------------------------------------------
class AdamW(Optimizer):
    """
    AdamW optimizer (decoupled weight decay).
    Automatically integrates with PySML AMP (GradScaler).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr, weight_decay)
        self.betas = betas
        self.eps = eps
        self.backend = get_backend()
        self.scaler = get_scaler() if amp_enabled() else None
        self.t = 0

    def step(self):
        self.t += 1
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            b1, b2 = self.betas

            for p in group["params"]:
                if not hasattr(p, "grad") or p.grad is None:
                    continue

                grad = p.grad
                if self.scaler is not None:
                    # AMP-safe unscale
                    grad = grad / self.scaler.scale

                state = self.state.setdefault(id(p), {})
                if len(state) == 0:
                    state["exp_avg"] = self.backend.zeros_like(p)
                    state["exp_avg_sq"] = self.backend.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                # Update moving averages
                exp_avg[:] = b1 * exp_avg + (1 - b1) * grad
                exp_avg_sq[:] = b2 * exp_avg_sq + (1 - b2) * self.backend.power(grad, 2)

                # Bias correction
                bias_c1 = 1 - b1 ** self.t
                bias_c2 = 1 - b2 ** self.t
                step_size = lr * math.sqrt(bias_c2) / bias_c1

                # Decoupled weight decay
                if wd > 0:
                    p[:] = p - lr * wd * p

                # Parameter update
                denom = self.backend.sqrt(exp_avg_sq) + self.eps
                p[:] = p - step_size * exp_avg / denom


# ---------------------------------------------------------------------
# SGD (for completeness)
# ---------------------------------------------------------------------
class SGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum
        self.velocities = {}

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if not hasattr(p, "grad") or p.grad is None:
                    continue

                grad = p.grad
                if wd > 0:
                    grad = grad + wd * p

                if self.momentum > 0:
                    v = self.velocities.get(id(p))
                    if v is None:
                        v = grad
                    else:
                        v = self.momentum * v + grad
                    self.velocities[id(p)] = v
                    grad = v

                p[:] = p - lr * grad


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad.data
            if self.weight_decay:
                g = g + self.weight_decay * p.data

            state = self.state.get(p, None)
            if state is None:
                m = p.backend.zeros_like(g)
                v = p.backend.zeros_like(g)
            else:
                m, v = state

            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * (g * g)

            m_hat = m / (1 - b1 ** self.t)
            v_hat = v / (1 - b2 ** self.t)

            p.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)
            self.state[p] = (m, v)



__all__ = ["Optimizer", "Adam", "AdamW", "SGD"]
