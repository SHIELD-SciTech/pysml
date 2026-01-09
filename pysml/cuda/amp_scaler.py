

class GradScaler:

    def __init__(self, init_scale=2.**16, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000, min_scale=1e-4, max_scale=2.**32):
        self._scale = init_scale
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

        return tensor * self._scale

    def get_scale(self):

        return self._scale

    def unscale_(self, optimizer):

        for group in optimizer.param_groups:
            for p in group["params"]:
                if hasattr(p, "grad") and p.grad is not None:
                    if hasattr(p.grad, 'data'):
                        p.grad.data = p.grad.data / self._scale
                    else:
                        p.grad = p.grad / self._scale

    def _check_inf_nan(self, optimizer):

        self._found_inf = False
        for group in optimizer.param_groups:
            for p in group["params"]:
                if hasattr(p, "grad") and p.grad is not None:
                    grad = p.grad
                    grad_data = grad.data if hasattr(grad, 'data') else grad
                    backend = getattr(grad, '_backend', None)
                    if backend is not None:
                        if backend.any(backend.isnan(grad_data)) or backend.any(backend.isinf(grad_data)):
                            self._found_inf = True
                            return
                    elif hasattr(grad_data, "isnan"):
                        if grad_data.isnan().any() or grad_data.isinf().any():
                            self._found_inf = True
                            return

    # ---------------------------------------------------------------------
    # Step logic
    # ---------------------------------------------------------------------
    def step(self, optimizer):

        self.unscale_(optimizer)
        self._check_inf_nan(optimizer)

        if not self._found_inf:
            optimizer.step()
        # Skip update to prevent NaNs if overflow detected

    def update(self):

        if self._found_inf:
            self._scale = max(self._scale * self.backoff_factor, self.min_scale)
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker % self.growth_interval == 0:
                self._scale = min(self._scale * self.growth_factor, self.max_scale)

    def state_dict(self):
        return {
            "scale": self._scale,
            "growth_tracker": self._growth_tracker,
        }

    def load_state_dict(self, state):
        self._scale = state.get("scale", self._scale)
        self._growth_tracker = state.get("growth_tracker", 0)

    def __repr__(self):
        return f"<GradScaler scale={self._scale:.2e} found_inf={self._found_inf}>"
