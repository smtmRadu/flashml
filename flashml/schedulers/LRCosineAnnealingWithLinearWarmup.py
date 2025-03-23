import torch
import math

class LRConsineAnnealingWithLinearWarmup():
    def __init__(self, 
                 optimizer:torch.optim, 
                 max_steps:int, 
                 warmup_steps:int=1000, 
                 min_lr:float=0.0,
                 *args,
                 **kwargs):
        '''
        @step() called after optim.step() \\
        max_steps = EPOCHS * BATCH_NUM \\
        warmup_steps = 3%-10% of max_steps
        '''      
        warmup_steps_ratio = kwargs.pop("warmup_steps_ratio", None)
        if warmup_steps_ratio is not None:
            assert 0.0 < warmup_steps_ratio < 1.0, "warmup_steps_ratio must be between 0.0 and 1.0"
            warmup_steps = int(warmup_steps_ratio * max_steps)

        assert warmup_steps is not None, "warmup_steps or warump_steps_ratio must be specified"
        assert max_steps > warmup_steps, f"The warmup steps ({warmup_steps}) must be less than total training steps T max ({max_steps})"

        self.optim = optimizer
        self.t_max = max_steps
        self.warmup_steps = warmup_steps
        self.eta_min = min_lr
        self.t_cur = 0
        self.eta_max = optimizer.param_groups[0]['lr']
        self.step() # init lr

    def step(self):
        '''
        Sets the optimizer's learning rates for each param_group. Must be called before optim.step()'''
        self.t_cur += 1
        eta_t = self.current_lr
        for param_group in self.optim.param_groups:
            param_group['lr'] = eta_t

    @property
    def current_lr(self):
        '''
        Get the current lr of the scheduler.
        '''
        if self.t_cur < self.warmup_steps:
            warm_lr = self.t_cur * self.eta_max / self.warmup_steps
            return warm_lr
        else:
            eta_t = self.eta_min + (self.eta_max - self.eta_min) / 2.0 * (1.0 + math.cos((self.t_cur - self.warmup_steps) / (self.t_max - self.warmup_steps) * math.pi))
            return eta_t
         