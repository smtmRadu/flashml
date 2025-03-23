import copy

class EMA():
    '''
    Exponential Moving Average (EMA) for NNs
    '''
    def __init__(self, model, polyak:float = 0.99, update_after:int=0):

        self._polyak = polyak
        self._warmup_steps = update_after
        self._step:int = 0
        self.ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    @property
    def EMA_Model(self):
        return self.ema_model

    def _update(self,model):
        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            old = ema_param.data
            new = param.data
            ema_param.data = old * self._polyak + (1 - self._polyak) * new 


    def step(self, model):
        if self._step < self._warmup_steps:
            # nothing happens
            pass
        elif self._step == self._warmup_steps:
            self.ema_model.load_state_dict(model.state_dict())
        else:
            self._update(model)
        self.step += 1


