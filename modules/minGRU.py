import torch
import torch.nn as nn
import torch.nn.functional as F

class MinGRU(nn.Module):
    def __init__(self, embedding_dim, bias: bool = True):
        '''
        
        parallel_scan: bool = False, use parallel scan algorithm for faster computation.
        log_space: bool = False, use log space for computation and parallel_scan_log for stability.
        '''
        super(MinGRU, self).__init__()
        self.d = embedding_dim 
        self.fc_z = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.fc_n = nn.Linear(embedding_dim, embedding_dim, bias=bias)
       
        torch.nn.init.kaiming_normal_(self.fc_z.weight)
        torch.nn.init.kaiming_normal_(self.fc_n.weight)
        if bias:
            torch.nn.init.zeros_(self.fc_n.bias)
            torch.nn.init.zeros_(self.fc_z.bias)
    
    def forward(self, x: torch.Tensor):     
        assert x.dim() == 3, "MinGRU - Input should be a 3D tensor (B, L, D)"

        # if not self.training:
        #     return self._forward_sequentially_log_space(x)
        # else:
        return self._forward_parallel_log_space(x)

    
    def _forward_sequentially_log_space(self, x: torch.Tensor): 
        B, L, D = x.shape
        h:list[torch.Tensor] = [torch.zeros(size=(B, 1, D), device=x.device)]     

        for i in range(L):
            z = F.sigmoid(self.fc_z(x[:, i, :]))
            n = g(self.fc_n(x[:, i, :]))
            h_t = (1 - z) * h[-1] + z * n
            h.append(h_t)

        return torch.cat(h[1:], dim=-2)
    
    def _forward_parallel_log_space(self, x: torch.Tensor):
        B, _, D = x.shape
        h0 = torch.zeros((B,1,D)).to(x.device)
          
        k = self.fc_z(x)
        log_z = -F.softplus(-k)
        log_coeffs = -F.softplus(k)
        log_h0 = log_g(h0)
        log_h_ = log_g(self.fc_n(x))
        log_values = torch.cat([log_h0, log_z + log_h_], dim = -2)
        h = parallel_scan_log(log_coeffs=log_coeffs, log_values=log_values)
        return h
    
def parallel_scan_log(log_coeffs, log_values):
    # log_coeffs: (batch_size, seq_len, input_size)
    # log_values: (batch_size, seq_len + 1, input_size)
    assert log_coeffs.dim() == log_values.dim() == 3, "parallel_scan_log - log_coeffs and log_values should have 3 dimensions"
    a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
    log_h0_plus_b_star = torch.logcumsumexp(
    log_values - a_star, dim=-2)
    log_h = a_star + log_h0_plus_b_star
    return torch.exp(log_h)[:, 1:]

def g(x):
     return torch.where(x >= 0, x+0.5, F.sigmoid(x))


def log_g(x):
    return torch.where(x >= 0, (F.relu(x)+0.5).log(),-F.softplus(-x))
