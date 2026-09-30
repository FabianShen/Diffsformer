import torch
import torch.nn as nn

class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization with conditioning injection.

    Regresses scale (γ) and shift (β) from the sum of time embedding
    and condition embedding, then applies:
        output = layer_norm(x) * (1 + γ) + β

    This is the key mechanism for injecting diffusion-step and condition
    information into the transformer backbone.

    Paper: Section IV-C-2 "Adaptive layer norm (ALN)"
    Reference: Peebles & Xie (2022) "Scalable Diffusion Models with Transformers"
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        # Scale and shift are regressed from the conditioning vector
        self.scale_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.shift_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self, x: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            conditioning: (batch, hidden_dim) — time + label + industry embedding
        Returns:
            (batch, seq_len, hidden_dim)
        """
        scale = self.scale_mlp(conditioning).unsqueeze(1)  # (batch, 1, hidden_dim)
        shift = self.shift_mlp(conditioning).unsqueeze(1)  # (batch, 1, hidden_dim)
        return self.norm(x) * (1.0 + scale) + shift