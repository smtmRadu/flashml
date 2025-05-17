import torch
import torch.nn as nn
import torch.nn.functional as F
from FFN import FFN

class DeepSeekMoE(nn.Module):
    def __init__(self, embedding_dim:int, routed_experts:int=15, shared_experts:int=1, active_experts:int=3, hidden_expansion:int=1, router_temperature:float=1) -> None:
        assert "not finished yet"
        assert routed_experts % active_experts == 0, "Number of top k experts must be divisible by the number of experts"
        assert hidden_expansion >= 1
        assert routed_experts > 1
        '''
        A mixture of experts with FFNs
        Args:
            `k` = number of active experts / token
            `hidden_expansion` = hidden dimension of each expert multiplied by the embedding dim
            `router_T` = router temperature
        '''
        super().__init__()
        self.k = active_experts
        self.T = router_temperature
        self.router = nn.Linear(embedding_dim, routed_experts)
        self.shared_experts = nn.ModuleList([FFN(embedding_dim, embedding_dim, embedding_dim * hidden_expansion) for _ in range(shared_experts)])
        self.routed_experts = nn.ModuleList([FFN(embedding_dim, embedding_dim, embedding_dim * hidden_expansion) for _ in range(routed_experts)])

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        B, L, D = x.shape
        #shared pass
        shared_outputs = []
        for s_e in self.shared_experts:
            s_o = s_e(x)
            shared_outputs.append(s_o)


        # routed pass
        routed_outputs = []
        routs_logits = self.router(x)
        routs_probs = torch.exp(routs_probs / self.T) / torch.sum(torch.exp(routs_probs / self.T), keepdim=True)  # (B, L, D) -> (B, L, exp_num)
        top_k_probs, top_k_indices = torch.topk(routs_probs.view(B*L, D), self.k, dim=-1)
        #x = (B, L, D) - id_exp = (B * L, k)
        for idx, exp in enumerate(self.routed_experts):
            mask = top_k_indices == idx
            selected_indices = mask.any(dim=-1).nonzero(as_tuple=True)[0]
            batch_for_exp_i = x.view(B * L, -1)[selected_indices]
            if batch_for_exp_i.shape[0] > 0: 
                expert_output = exp(batch_for_exp_i)
                routed_outputs.append((selected_indices, expert_output))

        return sum(shared_outputs.extend(routed_outputs))

