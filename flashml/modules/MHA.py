import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings

class MHA(nn.Module):
    '''
    Multihead Attention Layer with Rotary Positional Embeddings.
    '''
    def __init__(self, embedding_dim: int, heads_num: int, is_causal: bool = True, max_seq_len: int = 256, dropout=0.0):
        super().__init__()
        assert embedding_dim % heads_num == 0, "embedding_dim must be divisible by heads_num"
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // heads_num
        self.heads_num = heads_num
        self.max_seq_len = max_seq_len
        self.is_causal = is_causal
        self.w_qkv = nn.Linear(embedding_dim, embedding_dim * 3, bias=False) 
        self.w_o = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.rope = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=max_seq_len)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, "Input tensor must have 3 dimensions (B, L, D)"
        B, L, D = x.shape
        assert D == self.embedding_dim, f"Input embedding dim {D} does not match the expected dim {self.embedding_dim}"
        assert L <= self.max_seq_len, f"Input sequence length {L} exceeds the maximum allowed length {self.max_seq_len}"

        q, k, v = self.w_qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.heads_num, self.head_dim)
        k = k.view(B, L, self.heads_num, self.head_dim)
        v = v.view(B, L, self.heads_num, self.head_dim)
        q, k = self.rope(q), self.rope(k) # rope gets input (B, L, heads_num, head_dim)
        q, k, v = q.transpose(-2, -3), k.transpose(-2, -3), v.transpose(-2, -3)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal, dropout_p=self.dropout) #sdpa gets input (B, heads_num, L, head_dim)
        y = y.transpose(-2, -3).contiguous().view(B, L, D)
        return self.w_o(y)
        
        # autoregressive with kv caching
        if not self.training and self.is_causal:
            if self.kv_cache is not None:
                assert L-1 == self.kv_cache[0].shape[-2], "Add only 1 token at a time for autoregressive decoding"

            q = F.linear(x, weight=self.w_qkv.weight[:D], bias=None)
            k, v = F.linear(x[:, -1, :], weight=self.w_qkv.weight[D:], bias=None).chunk(2, -1)
            
            if not self.training and self.is_causal:
                if self.kv_cache is None:
                    self.kv_cache = k.detach().clone(), v.detach().clone()
                else:
                    k_cached, v_cached = self.kv_cache
                    q, k, v = q, torch.cat((k_cached, k), dim=1), torch.cat((v_cached, v), dim=1) 
                    self.kv_cache = k.detach().clone(), v.detach().clone()

                q, k, v = q.view(B, L, self.heads_num, self.head_dim), k.view(B, L, self.heads_num, self.head_dim), v.view(B, L, self.heads_num, self.head_dim)
                q, k = self.rope(q), self.rope(k) # rope gets input (B, L, heads_num, head_dim)
                q, k, v = q.transpose(-2, -3), k.transpose(-2, -3), v.transpose(-2, -3)
                y = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal) #sdpa gets input (B, heads_num, L, head_dim)
                y = y.transpose(-2, -3).contiguous().view(B, L, D)
                return self.w_o(y)
