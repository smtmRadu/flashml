import math
from typing import Literal

class LRScheduler:
    """
        `step()` called after `optim.step()`\\
        `max_steps` = **(EPOCHS x DATA_SIZE) / (BATCH_SIZE x GRADIENT_ACCUM_STEPS)** \\
        `warmup_steps_ratio` = 3%-10% of max_steps (use `warmup_steps_ratio` instead of `warmup_steps` to set it)
        'constant_steps_ratio' = ~70% of max_steps (use `constant_steps_ratio` instead of `constant_steps` to set it)
    """
    def __init__(
        self,
        optimizer,
        max_steps: int,
        warmup_steps: int = 0,
        constant_steps: int = 0,
        annealing_type: Literal["linear", "cosine"] = "cosine",
        min_lr: float = 0.0,
        *args,
        **kwargs,
    ):
        # ratios override absolute values if given
        warmup_steps_ratio = kwargs.pop("warmup_steps_ratio", None)
        constant_steps_ratio = kwargs.pop("constant_steps_ratio", None)
        
        if warmup_steps_ratio is not None:
            assert 0.0 < warmup_steps_ratio < 1.0, (
                "warmup_steps_ratio must be between 0.0 and 1.0"
            )
            warmup_steps = int(warmup_steps_ratio * max_steps)
        
        if constant_steps_ratio is not None:
            assert 0.0 <= constant_steps_ratio < 1.0, (
                "constant_steps_ratio must be between 0.0 and 1.0"
            )
            constant_steps = int(constant_steps_ratio * max_steps)

        assert warmup_steps >= 0, "warmup_steps must be >= 0"
        assert max_steps > warmup_steps + constant_steps, (
            f"warmup_steps + constant_steps ({warmup_steps + constant_steps}) must be less than max_steps ({max_steps})"
        )

        self.optimizer = optimizer
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.constant_steps = constant_steps
        self.annealing_type = annealing_type
        self.min_lr = min_lr
        self.cur_step = 0
        self.max_lr = optimizer.param_groups[0]["lr"]
        self.step()  # init lr

    def step(self):
        """
        Updates optimizer LR (call after optimizer.step()).
        """
        # Don't increment cur_step on initialization call (see __init__)
        if self.cur_step > 0:
            self.cur_step += 1
        else:
            self.cur_step = 1

        lr = self.current_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @property
    def current_lr(self):
        if self.cur_step <= self.warmup_steps and self.warmup_steps > 0:
            # Linear warmup from 0 to max_lr
            return self.max_lr * self.cur_step / self.warmup_steps
        elif self.cur_step <= self.warmup_steps + self.constant_steps:
            # Constant phase
            return self.max_lr
        else:
            # Annealing phase
            anneal_steps = self.max_steps - self.warmup_steps - self.constant_steps
            anneal_step = self.cur_step - self.warmup_steps - self.constant_steps
            if anneal_step >= anneal_steps:
                return self.min_lr
            if self.annealing_type == "cosine":
                cosine_decay = 0.5 * (1 + math.cos(math.pi * anneal_step / anneal_steps))
                return self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
            elif self.annealing_type == "linear":
                linear_decay = 1 - anneal_step / anneal_steps
                return self.min_lr + (self.max_lr - self.min_lr) * linear_decay
            else:
                raise ValueError(f"Unknown annealing_type: {self.annealing_type}")

    def state_dict(self):
        return {
            "cur_step": self.cur_step,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "constant_steps": self.constant_steps,
            "max_steps": self.max_steps,
            "annealing_type": self.annealing_type,
        }

    def load_state_dict(self, state_dict):
        self.cur_step = state_dict["cur_step"]
        self.max_lr = state_dict["max_lr"]
        self.min_lr = state_dict["min_lr"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.constant_steps = state_dict["constant_steps"]
        self.max_steps = state_dict["max_steps"]
        self.annealing_type = state_dict["annealing_type"]
