"""
Automatic Gradient Scaling for PySML AMP
Compatible with both CUDA and XPU backends.
"""

class GradScaler:
    """
    Lightweight gradient scaler for mixed precision training.
    
    Example:
        >>> scaler = GradScaler()
        >>> with amp.autocast("float16"):
        ...     loss = model(x)
        >>> scaled_loss = scaler.scale(loss)
        >>> scaled_loss.backward()
        >>> scaler.step(optimizer)
        >>> scaler.update()
    """

    def __init__(self, init_scale=2.**16, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000, min_scale=1e-4, max_scale=2.**32):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.min_scale = min_scale
        self.max_scale = max_scale
        self._growth_tracker = 0
        self._found_inf = False

    # ---------------------------------------------------------------------
    # Scaling logic
    # ---------------------------------------------------------------------
    def scale(self, tensor):
        """Scale loss before backward pass."""
        return tensor * self.scale

    def unscale_(self, optimizer):
        """
        Unscale gradients in place.
        Should be called before checking for inf/nan.
        """
        for group in optimizer.param_groups:
            for p in group["params"]:
                if hasattr(p, "grad") and p.grad is not None:
                    p.grad /= self.scale

    def _check_inf_nan(self, optimizer):
        """Detect NaN or Inf gradients."""
        self._found_inf = False
        for group in optimizer.param_groups:
            for p in group["params"]:
                if hasattr(p, "grad") and p.grad is not None:
                    grad = p.grad
                    if hasattr(grad, "isnan"):
                        if grad.isnan().any() or grad.isinf().any():
                            self._found_inf = True
                            return

    # ---------------------------------------------------------------------
    # Step logic
    # ---------------------------------------------------------------------
    def step(self, optimizer):
        """
        Perform optimizer step with gradient scaling.
        Automatically skips step if overflow detected.
        """
        self.unscale_(optimizer)
        self._check_inf_nan(optimizer)

        if not self._found_inf:
            optimizer.step()
        else:
            # Skip update to prevent NaNs
            pass

    def update(self):
        """Adjust scale dynamically."""
        if self._found_inf:
            self.scale = max(self.scale * self.backoff_factor, self.min_scale)
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker % self.growth_interval == 0:
                self.scale = min(self.scale * self.growth_factor, self.max_scale)

    def state_dict(self):
        return {
            "scale": self.scale,
            "growth_tracker": self._growth_tracker,
        }

    def load_state_dict(self, state):
        self.scale = state.get("scale", self.scale)
        self._growth_tracker = state.get("growth_tracker", 0)

    def __repr__(self):
        return f"<GradScaler scale={self.scale:.2e} found_inf={self._found_inf}>"
