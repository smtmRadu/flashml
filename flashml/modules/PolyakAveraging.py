from copy import deepcopy

class PolyakAveraging():
    """
    ### Usage:
    polyak_model = PolyakAveraging(model)

    for epoch in epochs:
        for step in steps:
            forward()
            zero_grad()
            backward()
            optim_step()
            polyak_model.step()  # Update EMA weights

    model = polyak_model.EMA_Model
    """
    def __init__(self, model, polyak:float = 0.99, update_after:int=0):
        '''
        Args:
            model: nn.Module
                The model to be used for EMA
            polyak: float
                The polyak averaging factor
            update_after: int
                The number of steps to wait before starting to update the EMA model
        '''
        self._polyak = polyak
        self._warmup_steps = update_after
        self._step:int = 0
        self._ema_model = deepcopy(model).eval().requires_grad_(False)
        self._model_ref = model
    @property
    def EMA_Model(self):
        '''
        Returns the current state of the EMA model'''
        return self._ema_model

    def step(self):
        if self._step < self._warmup_steps:
            # nothing happens
            pass
        else:
            for ema_param, param in zip(self._ema_model.parameters(), self._model_ref.parameters()):
                ema_param.data.mul_(self._polyak).add_(param.detach(), alpha=1 - self._polyak)

        self._step += 1


