import torch
import torch.nn as nn
import torch.nn.functional as F

class MinLSTM(nn.Module):
    def __init__(self, embedding_dim, bias: bool = True):
        '''
        parallel_scan: bool = False, use parallel scan algorithm for faster computation.
        log_space: bool = False, use log space for computation and parallel_scan_log for stability.
        '''
        super(MinLSTM, self).__init__()
        self.d = embedding_dim 
        self.fc_f = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.fc_i = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.fc_h = nn.Linear(embedding_dim, embedding_dim, bias=bias)
       
        torch.nn.init.kaiming_normal_(self.fc_f .weight)
        torch.nn.init.kaiming_normal_(self.fc_i.weight)
        torch.nn.init.kaiming_normal_(self.fc_h.weight)
        if bias:
            torch.nn.init.zeros_(self.fc_f.bias)
            torch.nn.init.zeros_(self.fc_i.bias)
            torch.nn.init.zeros_(self.fc_h.bias)

    def forward(self, x: torch.Tensor):     
        assert x.dim() == 3,f"MinLSTM - Input should be a 3D tensor (B, L, D), received ({x.shape})"
        assert x.size(-1) == self.d, f"MinLSTM - Input should have the same dimension as the embedding_dim (received {x.size(-1)}, expected {self.d})"
        # if not self.training:
        #     return self._forward_sequentially_log_space(x)
        
        return self._forward_parallel_log_space(x)

    
    def _forward_sequentially_log_space(self, x: torch.Tensor): 
        B, L, D = x.shape
        h:list[torch.Tensor] = [torch.zeros(size=(B, 1, D), device=x.device)]     

        for i in range(L):
            f_t = F.sigmoid(self.fc_f(x[:, i, :]))
            i_t = F.sigmoid(self.fc_i(x[:, i, :]))
            tilde_h_t = g(self.fc_h(x[:, i, :]))
            f_prime_t = f_t / (f_t + i_t)
            i_prime_t = i_t / (f_t + i_t)
            h_t = f_prime_t * h[-1] + i_prime_t * tilde_h_t
            h.append(h_t)

        return torch.cat(h[1:], dim=-2)
    
    def _forward_parallel_log_space(self, x: torch.Tensor):
        B, _, D = x.shape
        h0 = torch.zeros((B,1,D)).to(x.device)
          
        diff = F.softplus(-self.fc_f(x)) - F.softplus(-self.fc_i(x))
        log_f = -F.softplus(diff)
        log_i = -F.softplus(-diff)
        log_h0 = torch.log(h0)
        log_tilde_h = log_g(self.fc_h(x))
        h = parallel_scan_log(log_coeffs=log_f, log_values=torch.cat([log_h0, log_i + log_tilde_h], dim=-2))
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
     return torch.where(x >= 0, x+0.5, torch.sigmoid(x))
def log_g(x):
    return torch.where(x >= 0, (F.relu(x)+0.5).log(),-F.softplus(-x))
