import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for diffusion timesteps.

    Encodes integer timestep t into a fixed-dimensional vector using
    sine/cosine functions at different frequencies, enabling the
    transformer to distinguish different noise levels.

    Paper: Section IV-C-2 "Time step encoding"
    """

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch,) integer timesteps in [0, T-1]
        Returns:
            (batch, dim) sinusoidal embeddings
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) *
            torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (batch, half)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding