from typing import Optional

import torch
import torch.nn as nn


class pRoPE(nn.Module):
    def __init__(
        self, head_dim: int, max_wavelength: int = 10_000, rope_percentage: float = 0.75
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be an even number, but got {head_dim}")
        if not (0.0 <= rope_percentage <= 1.0):
            raise ValueError(
                f"rope_percentage must be between 0.0 and 1.0, but got {rope_percentage}"
            )

        self.head_dim = head_dim
        self.max_wavelength = max_wavelength
        self.rope_percentage = rope_percentage

        rope_angles = int(round(rope_percentage * self.head_dim / 2))
        nope_angles = self.head_dim // 2 - rope_angles

        if rope_angles > 0:
            fraction = 2.0 * torch.arange(0, rope_angles) / self.head_dim
            timescale_rope = max_wavelength**fraction
        else:
            timescale_rope = torch.empty((0,))

        if nope_angles > 0:
            timescale_nope = torch.full((nope_angles,), float("inf"))
        else:
            timescale_nope = torch.empty((0,))

        timescale = torch.cat((timescale_rope, timescale_nope), dim=0)
        self.register_buffer("timescale", timescale, persistent=True)

    def forward(
        self, x: torch.Tensor, positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Applies p-RoPE to the input tensor.

        Args:
            x (torch.Tensor): Input tensor, e.g., query or key, with shape [B, L, num_heads, head_dim].
                              Rotation is applied to the last dimension.
            positions (Optional[torch.Tensor]): Optional tensor containing the position ids
                                                of each token. Shape [B, L].
                                                If None, assumes sequential positions [0, 1, ..., L-1]
                                                for all sequences in the batch. Defaults to None.

        Returns:
            torch.Tensor: Output tensor with p-RoPE applied, same shape as input x ([B, L, nH, hD]).
        """
        B, L, _, head_dim = x.shape

        if head_dim != self.head_dim:
            raise ValueError(
                f"Input tensor's head_dim (last dim={head_dim}) does not match "
                f"initialized head_dim {self.head_dim}"
            )

        if positions is None:
            positions = torch.arange(L, device=x.device, dtype=torch.long).unsqueeze(0)
        else:
            if positions.shape != (B, L):
                raise ValueError(
                    f"Provided positions shape {positions.shape} does not match "
                    f"expected shape [B, L] derived from x: [{B}, {L}]"
                )

        positions = positions.long()

        positions_expanded = positions.float().unsqueeze(-1).unsqueeze(-1)
        timescale_expanded = self.timescale.view(1, 1, 1, -1)  # type: ignore

        sinusoid_inp = positions_expanded / timescale_expanded
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)

        first_half, second_half = torch.chunk(x, 2, dim=-1)

        first_part_rotated = first_half * cos - second_half * sin
        second_part_rotated = second_half * cos + first_half * sin
        out = torch.cat([first_part_rotated, second_part_rotated], dim=-1)
        return out.to(x.dtype)
