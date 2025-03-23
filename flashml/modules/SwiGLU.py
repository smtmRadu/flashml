import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    '''
    Swish-Gated Linear Unit (SwiGLU)
    '''
    def __init__(self, input_size, hidden_size, output_size, bias=False):
        super().__init__()
        self.input_size = input_size
    
        self.fc1 = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc2 = nn.Linear(input_size, hidden_size, bias=bias)
        self.ofc = nn.Linear(hidden_size, output_size, bias=bias)

        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.kaiming_normal_(self.ofc.weight)

        if bias:
            torch.nn.init.zeros_(self.fc1.bias)
            torch.nn.init.zeros_(self.fc2.bias)
            torch.nn.init.zeros_(self.ofc.bias)
        
    def forward(self, x:torch.Tensor):
        assert x.size(dim=-1) == self.input_size, f"Input size should be equal to input_size (x: {x.shape}, input_size: {self.input_size})"
        h = F.silu(self.fc1(x)) * self.fc2(x)
        return self.ofc(h) 