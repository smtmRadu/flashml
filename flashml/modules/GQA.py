import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings

class GQA(nn.Module):
    """
    Grouped Query Attention with Rotary Positional Embeddings.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads_q: int,
        num_heads_kv: int = None,
        expansion_factor: float = 1,
        is_causal: bool = False,
        dropout=0.0,
        qk_norm: bool = False,
        qk_norm_eps:float = 1e-6,
        use_rope: bool = False,
        rope_max_seq_len: int = 4096,
        rope_theta=10000,
    ):
        """Grouped Query Attention with Rotary Positional Embeddings. The scale is default: 1/sqrt(dim(k)).
        Args:
            embedding_dim (int): The embedding dimension of the model.
            heads_num_q (int): number of attention heads.
            num_heads_kv (int): If None, it defaults to `heads_num_q` resulting in standard Multihead Attention. If 1, it is Multiquery Attention, otherwise (1 < nh_kv < nh_q, nh_q | nh_kv) it is Grouped Query Attention.
            is_causal (bool, optional): Applies causal masking to the attention weights. Defaults to False.
            dropout (float, optional): Dropout over attention weights. Defaults to 0.0.
            use_rope (bool, optional): Rotates the queries and keys to encode positions. Defaults to True.
            rope_max_seq_len (int, optional): Max sequence length (necessary for RoPE). Note that it can be extended into the future with furter finetuning on larger datasets or with YaRN. Defaults to 4096.
            rope_theta (int, optional): Rotation base frequency. Defaults to 10000.
        """
        super().__init__()
        assert embedding_dim % num_heads_q == 0, (
            "embedding_dim must be divisible by heads_num"
        )

        if num_heads_kv is None:
            num_heads_kv = num_heads_q
        assert (
            num_heads_q % num_heads_kv == 0
            and num_heads_kv <= num_heads_q
            and num_heads_kv >= 1
        ), "group_kv must be in range [1, num_heads] and must divide num_heads"

        self.embedding_dim = embedding_dim
        self.inner_embedding_dim = int(embedding_dim * expansion_factor)
        self.head_dim = self.inner_embedding_dim // num_heads_q
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.max_seq_len = rope_max_seq_len
        self.q_norm = nn.RMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None # 182 https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
        self.k_norm = nn.RMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.is_causal = is_causal
        self.w_qkv = nn.Linear(
            in_features=embedding_dim,
            out_features=self.inner_embedding_dim
            + self.inner_embedding_dim * num_heads_kv // num_heads_q * 2,
            bias=False,
        )
        self.w_o = nn.Linear(self.inner_embedding_dim, embedding_dim, bias=False)
        self.rope = (
            RotaryPositionalEmbeddings(
                dim=self.head_dim, max_seq_len=rope_max_seq_len, base=rope_theta
            )
            if use_rope
            else None
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        assert x.dim() == 3, "Input tensor must have 3 dimensions (B, L, D)"
        B, L, D = x.shape
        assert D == self.embedding_dim, (
            f"Input embedding dim {D} does not match the expected dim {self.embedding_dim}"
        )
        assert L <= self.max_seq_len, (
            f"Input sequence length {L} exceeds the maximum allowed length {self.max_seq_len}"
        )
        qkv_splits = [
            self.inner_embedding_dim,
            self.inner_embedding_dim * self.num_heads_kv // self.num_heads_q,
            self.inner_embedding_dim * self.num_heads_kv // self.num_heads_q,
        ]
        q, k, v = torch.split(self.w_qkv(x), split_size_or_sections=qkv_splits, dim=-1)
        


        q = q.view(B, L, self.num_heads_q, self.head_dim)
        k = k.view(B, L, self.num_heads_kv, self.head_dim)
        v = v.view(B, L, self.num_heads_kv, self.head_dim)
        
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
            
        if self.rope is not None:
            q, k = (
                self.rope(q),
                self.rope(k),
            )  # rope gets input (B, L, heads_num, head_dim)
        q, k, v = q.transpose(-2, -3), k.transpose(-2, -3), v.transpose(-2, -3)

        attn_mask_expanded = None
        if attn_mask is not None:
            # mask: (B, L) with 1 = real token, 0 = pad
            # We want (B, 1, L) → (B, 1, 1, L) then broadcast
            attn_mask_expanded = ~attn_mask.to(torch.bool).unsqueeze(1).unsqueeze(
                2
            )  # shape: (B, 1, 1, L)
            attn_mask_expanded = attn_mask_expanded.expand(
                -1, self.num_heads_q, L, -1
            )  # (B, H, L, L)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=self.is_causal,
            dropout_p=self.dropout,
            scale=None,  # default 1/(sqrt(dim(k))
            attn_mask=attn_mask_expanded,
            enable_gqa=False if self.num_heads_q == self.num_heads_kv else True,
        )  # sdpa gets input (B, heads_num, L, head_dim)
        y = y.transpose(-2, -3).contiguous().view(B, L, self.inner_embedding_dim)
        return self.w_o(y)
