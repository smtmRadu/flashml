import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim:int, p:int = None, eps=1e-8, affine=True):
        super(RMSNorm, self).__init__()
        assert p > 0. and p < 1. if p is not None else True, "p must be between 0 and 1"
        self.dim = dim
        self.eps = eps
        self.p = p
        self.affine = affine

        if affine:
            self.scale = nn.Parameter(torch.ones(dim))
            self.register_parameter("scale", self.scale)
            self.bias = nn.Parameter(torch.zeros(dim))
            self.register_parameter("shift", self.bias)
            
    def forward(self, x):
        if self.p is None:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.dim
        else:
            partial_size = int(self.dim * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.dim - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_hat = x / (rms_x + self.eps)

        if self.affine:
            return x_hat * self.scale + self.bias

        return x_hat