import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings

class MHA(nn.Module):
    '''
    Multihead Attention Layer with Rotary Positional Embeddings
    '''
    def __init__(self, embedding_dim: int, heads_num: int, is_causal: bool = True, max_seq_len: int = 256):
        super().__init__()
        assert embedding_dim % heads_num == 0, "embedding_dim must be divisible by heads_num"
        self.head_dim = embedding_dim // heads_num
        self.heads_num = heads_num
        self.is_causal = is_causal
        self.w_qkv = nn.Linear(embedding_dim, embedding_dim * 3, bias=False) 
        self.w_o = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.rope = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = torch.chunk(self.w_qkv(x), 3, dim=-1)
        q = q.view(B, L, self.heads_num, self.head_dim)
        k = k.view(B, L, self.heads_num, self.head_dim)
        v = v.view(B, L, self.heads_num, self.head_dim)
        q, k = self.rope(q), self.rope(k) # rope gets input (B, L, heads_num, head_dim)
        q, k, v = q.transpose(-2, -3), k.transpose(-2, -3), v.transpose(-2, -3)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal) #sdpa gets input (B, heads_num, L, head_dim)
        y = y.transpose(-2, -3).contiguous().view(B, L, D)
        return self.w_o(y)