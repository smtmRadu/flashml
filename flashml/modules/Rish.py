import torch
import torch.nn as nn


class Rish(nn.Module):
    """
    I'm experimenting with this one, seems to be too powerfull.
    = (x - 1) * exp(x) / (1 + exp(x))

    """

    def __init__(self):
        super(Rish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        exp_x = x.exp()
        return (x - 1) * exp_x / (1 + exp_x)
