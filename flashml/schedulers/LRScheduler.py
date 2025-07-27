from typing import Literal
import math
### This is a general LRScheduler class that can be used to make almost any annealing you want.
class LRScheduler:
    """
        `step()` called after `optim.step()`\\
        `max_steps` = **(EPOCHS x DATA_SIZE) / (BATCH_SIZE x GRADIENT_ACCUM_STEPS)** \\
        `warmup_steps_ratio` = 3%-10% of max_steps (use warmup_steps_ratio instead of warmup_steps to set it)\\
        `constant_steps_ratio` = ~10% of max_steps (use constant_steps_ratio instead of constant_steps to set it)\\
        `cycles` = number of cycles to anneal over (default 1)\\
        `max_lr_decay_factor` = factor to decay max_lr per cycle (default 1=no decay)\\
        `gamma` = (if selected) gamma for exponential/logarithmic decay (default 0.95)\
    """
    def __init__(
        self,
        optimizer,
        max_steps: int,
        warmup_steps: int = 0,
        constant_steps: int = 0,
        annealing_type: Literal["linear", "cosine", "exponential", "logarithmic"] = "cosine",
        num_cycles: int = 1,
        max_lr_decay_factor: float = 1.0, 
        min_lr: float = 1e-8,
        gamma: float = 0.95,
        *args,
        **kwargs,
    ):
        warmup_steps_ratio = kwargs.pop("warmup_steps_ratio", None)
        constant_steps_ratio = kwargs.pop("constant_steps_ratio", None)
        
        if warmup_steps_ratio is not None:
            assert 0.0 < warmup_steps_ratio < 1.0
            warmup_steps = int(warmup_steps_ratio * max_steps)
        if constant_steps_ratio is not None:
            assert 0.0 <= constant_steps_ratio < 1.0
            constant_steps = int(constant_steps_ratio * max_steps)
        assert warmup_steps >= 0
        assert max_steps > warmup_steps + constant_steps

        self.optimizer = optimizer
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.constant_steps = constant_steps
        self.annealing_type = annealing_type
        self.min_lr = min_lr
        self.num_cycles = num_cycles
        self.max_lr_decay_factor = max_lr_decay_factor
        self.cur_step = 0
        self.init_max_lr = optimizer.param_groups[0]["lr"]
        self.gamma = gamma
        self.step() 

    def step(self):
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
            return self.init_max_lr * self.cur_step / self.warmup_steps
        elif self.cur_step <= self.warmup_steps + self.constant_steps:
            return self.init_max_lr
        else:
            anneal_steps = self.max_steps - self.warmup_steps - self.constant_steps
            anneal_step = self.cur_step - self.warmup_steps - self.constant_steps - 1  

            if self.cur_step > self.max_steps:
                return self.min_lr
                    

            cycle_counts = [anneal_steps // self.num_cycles] * self.num_cycles
            for i in range(anneal_steps % self.num_cycles):
                cycle_counts[i] += 1

            cycle_idx = 0
            cycle_step = anneal_step
            for steps_in_this_cycle in cycle_counts:
                if cycle_step < steps_in_this_cycle:
                    break
                cycle_step -= steps_in_this_cycle
                cycle_idx += 1

            cur_max_lr = self.init_max_lr * (self.max_lr_decay_factor ** cycle_idx)

            if self.annealing_type == "cosine":
                if steps_in_this_cycle > 1:
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * cycle_step / (steps_in_this_cycle - 1)))
                else:
                    cosine_decay = 0.0
                lr = self.min_lr + (cur_max_lr - self.min_lr) * cosine_decay
            elif self.annealing_type == "linear":
                if steps_in_this_cycle > 1:
                    linear_decay = 1 - (cycle_step / (steps_in_this_cycle - 1))
                else:
                    linear_decay = 0.0
                lr = self.min_lr + (cur_max_lr - self.min_lr) * linear_decay
            elif self.annealing_type == "exponential":
                if steps_in_this_cycle > 1:
                    effective_gamma = (self.min_lr / cur_max_lr) ** (1 / (steps_in_this_cycle - 1))
                else:
                    effective_gamma = 1.0
                exp_decay = effective_gamma ** cycle_step
                lr = cur_max_lr * exp_decay
            elif self.annealing_type == "logarithmic":
                if steps_in_this_cycle > 1:
                    numer = math.log(1 + cycle_step)
                    denom = math.log(1 + steps_in_this_cycle - 1)
                    factor = numer / denom if denom != 0 else 0.0
                else:
                    factor = 1.0
                lr = self.min_lr + (cur_max_lr - self.min_lr) * (1 - factor)
            else:
                raise ValueError(f"Unknown annealing_type: {self.annealing_type}")

            return lr

    def state_dict(self):
        return {
            "cur_step": self.cur_step,
            "init_max_lr": self.init_max_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "constant_steps": self.constant_steps,
            "max_steps": self.max_steps,
            "annealing_type": self.annealing_type,
            "num_cycles": self.num_cycles,
            "max_lr_decay_factor": self.max_lr_decay_factor,
            "gamma": self.gamma
        }

    def load_state_dict(self, state_dict):
        self.cur_step = state_dict["cur_step"]
        self.init_max_lr = state_dict["init_max_lr"]
        self.min_lr = state_dict["min_lr"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.constant_steps = state_dict["constant_steps"]
        self.max_steps = state_dict["max_steps"]
        self.annealing_type = state_dict["annealing_type"]
        self.num_cycles= state_dict["num_cycles"]
        self.max_lr_decay_factor = state_dict["max_lr_decay_factor"]
        self.gamma = state_dict.get("gamma", 0.95)
