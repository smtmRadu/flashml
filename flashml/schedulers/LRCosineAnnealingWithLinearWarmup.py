import math


class LRConsineAnnealingWithLinearWarmup:
    """
        `step()` called after `optim.step()`\\
        `max_steps` = **(EPOCHS x DATA_SIZE) / (BATCH_SIZE x GRADIENT_ACCUM_STEPS)** \\
        `warmup_steps_ratio` = 3%-10% of max_steps (use `warmup_steps_ratio` instead of `warmup_steps` to set it)
    """

    def __init__(
        self,
        optimizer,
        max_steps: int,
        warmup_steps: int = 1000,
        min_lr: float = 0.0,
        *args,
        **kwargs,
    ):
        warmup_steps_ratio = kwargs.pop("warmup_steps_ratio", None)
        if warmup_steps_ratio is not None:
            assert 0.0 < warmup_steps_ratio < 1.0, (
                "warmup_steps_ratio must be between 0.0 and 1.0"
            )
            warmup_steps = int(warmup_steps_ratio * max_steps)

        assert warmup_steps is not None, (
            "warmup_steps or warump_steps_ratio must be specified"
        )
        assert max_steps > warmup_steps, (
            f"The warmup steps ({warmup_steps}) must be less than total training steps T max ({max_steps})"
        )

        self.optim = optimizer
        self.t_max = max_steps
        self.warmup_steps = warmup_steps
        self.eta_min = min_lr
        self.t_cur = 0
        self.eta_max = optimizer.param_groups[0]["lr"]
        self.step()  # init lr

    def step(self):
        """
        Sets the optimizer's learning rates for each param_group. Must be called before optim.step()"""
        self.t_cur += 1
        eta_t = self.current_lr
        for param_group in self.optim.param_groups:
            param_group["lr"] = eta_t

    @property
    def current_lr(self):
        """
        Get the current lr of the scheduler.
        """
        if self.t_cur < self.warmup_steps:
            warm_lr = self.t_cur * self.eta_max / self.warmup_steps
            return warm_lr
        else:
            eta_t = self.eta_min + (self.eta_max - self.eta_min) / 2.0 * (
                1.0
                + math.cos(
                    (self.t_cur - self.warmup_steps)
                    / (self.t_max - self.warmup_steps)
                    * math.pi
                )
            )
            return eta_t

    def state_dict(self):
        return {
            "t_cur": self.t_cur,
            "eta_max": self.eta_max,
            "eta_min": self.eta_min,
            "warmup_steps": self.warmup_steps,
            "t_max": self.t_max,
        }

    def load_state_dict(self, state_dict):
        self.t_cur = state_dict["t_cur"]
        self.eta_max = state_dict["eta_max"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.t_max = state_dict["t_max"]
        self.eta_min = state_dict["eta_min"]
