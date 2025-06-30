import torch
import torch.nn as nn
from torchtune.modules import RotaryPositionalEmbeddings


class MLA(nn.Module):
    """
    Multihead Latent Attention Layer with Rotary Positional Embeddings.
    """

    def __init__(
        self,
        embedding_dim: int,
        heads_num: int,
        max_seq_len: int = 4096,
        is_causal: bool = False,
        dropout=0.0,
    ):
        super().__init__()
        assert embedding_dim % heads_num == 0, (
            "embedding_dim must be divisible by heads_num"
        )
        self.max_seq_len = max_seq_len
        self.is_causal = is_causal

        self.d = embedding_dim
        self.dh = embedding_dim // heads_num
        self.nh = heads_num

        # idk what value they use for the compression ratio
        kv_compression_dim = self.d // 3
        q_compression_dim = self.d // 3
        assert kv_compression_dim < self.dh * self.nh, (
            "latent_dim must be way smaller than the total number of heads"
        )
        assert q_compression_dim < self.dh * self.nh, (
            "latent_dim must be way smaller than the total number of heads"
        )

        self.dc = kv_compression_dim
        self.dc_prime = q_compression_dim
        self.dhR = None

        self.W_DKV = nn.Linear(embedding_dim, self.dc, bias=False)  # W_DKV in R_dcxd
        self.W_UK = nn.Linear(
            self.dc, self.dh * self.nh, bias=False
        )  # W_UK & W_UV in R_dhnh*dc
        self.W_UV = nn.Linear(self.dc, self.dh * self.nh, bias=False)
        self.W_KR = nn.Linear(self.d, self.dhR, bias=False)  # W_KR in R_dRh x d

        self.W_DQ = nn.Linear(
            embedding_dim, self.dc_prime, bias=False
        )  # W_DQ in R_dc_prime x d
        self.W_UQ = nn.Linear(self.dh * self.nh, self.dc_prime, bias=False)
        self.W_QR = nn.Linear(self.dc_prime, self.dhR * self.nh, bias=False)

        self.W_O = nn.Linear(self.dh * self.nh, self.d, bias=False)

        self.rope = RotaryPositionalEmbeddings(dim=self.dhR, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, "Input tensor must have 3 dimensions (B, L, D)"
        B, L, D = x.shape
        assert D == self.d, (
            f"Input embedding dim {D} does not match the expected dim {self.d}"
        )
        assert L <= self.max_seq_len, (
            f"Input sequence length {L} exceeds the maximum allowed length {self.max_seq_len}"
        )

        # input (B, L, d)
        # c_kv = (B, L, dc)
        c_KV = self.W_DKV(x)  # TO BE CACHED-----------------------------------
        k_C = self.W_UK(c_KV)  # (B, L, nh*dh)

        k_R = self.rope(
            self.W_KR(x)
        )  # (B, L, nh*dh) # TO BE CACHED-----------------------------------

        k_i = torch.cat([k_C, k_R], dim=-1)
        v_C = self.W_UV(c_KV)

        c_Q = self.W_DQ(x)
        q_C = self.W_UQ(c_Q)
        q_R = self.rope(self.W_QR(c_Q))
        q_o = torch.cat([q_C, q_R], dim=-1)
