import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings

class GQA(nn.Module):
    '''
    Grouped Query Attention Layer with Rotary Positional Embeddings.
    '''
    def __init__(self, embedding_dim: int, q_heads_num:int, kv_heads_num:int, use_rope:bool=True, max_seq_len: int = 4096, is_causal: bool = True, dropout=0.0):
        super().__init__()
        assert q_heads_num % kv_heads_num == 0, "Query heads num must divide kv heads num"
        # kv heads num < q heads num
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // q_heads_num
        self.Hq = q_heads_num
        self.Hkv = kv_heads_num
        self.max_seq_len = max_seq_len
        self.is_causal = is_causal
        self.w_qkv = nn.Linear(embedding_dim, (self.Hq + self.Hkv * 2) * self.head_dim, bias=False) 
        self.w_o = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.rope = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=max_seq_len) if use_rope else None
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, "Input tensor must have 3 dimensions (B, L, D)"
        B, L, D = x.shape
        assert D == self.embedding_dim, f"Input embedding dim {D} does not match the expected dim {self.embedding_dim}"
        assert L <= self.max_seq_len, f"Input sequence length {L} exceeds the maximum allowed length {self.max_seq_len}"

        qkv = self.w_qkv(x)
        q = qkv[:, :, :self.Hq * self.head_dim].view(B, L, self.Hq, self.head_dim)
        k, v = qkv[:, :, self.Hq * self.head_dim:].chunk(2, dim=-1)
        k = k.view(B, L, self.Hkv, self.head_dim)
        v = v.view(B, L, self.Hkv, self.head_dim)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k) # rope gets input (B, L, heads_num, head_dim)
        q, k, v = q.transpose(-2, -3), k.transpose(-2, -3), v.transpose(-2, -3)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal, dropout_p=self.dropout, enable_gqa=True) #sdpa gets input (B, heads_num, L, head_dim)
        y = y.transpose(-2, -3).contiguous().view(B, L, D)
        return self.w_o(y)
