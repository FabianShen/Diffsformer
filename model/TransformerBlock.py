import torch
import torch.nn as nn
from AdaptiveLayerNorm import AdaptiveLayerNorm

class TransformerBlock(nn.Module):
    """
    Single transformer block with Adaptive Layer Norm conditioning.

    Structure:
        x → ALN(x, c) → Multi-Head Self-Attention → Residual
          → ALN(x, c) → FFN (MLP) → Residual

    Zero-initialized: the final linear layer of each component is initialized
    to zero, making the block an identity function at training start.
    This stabilizes early training.

    Paper: Section IV-C-2 "Zero initialization"
    """

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Self-attention branch
        self.attn_norm = AdaptiveLayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        # Zero-init for attention output projection
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)

        # FFN branch
        self.ffn_norm = AdaptiveLayerNorm(hidden_dim)
        ffn_hidden = int(hidden_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_dim),
            nn.Dropout(dropout),
        )
        # Zero-init for FFN output
        nn.init.zeros_(self.ffn[-2].weight)
        nn.init.zeros_(self.ffn[-2].bias)

    def forward(
        self, x: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            conditioning: (batch, hidden_dim)
        Returns:
            (batch, seq_len, hidden_dim)
        """
        # Self-attention with ALN
        normed = self.attn_norm(x, conditioning)
        attn_out, _ = self.attention(normed, normed, normed)
        x = x + attn_out

        # FFN with ALN
        normed = self.ffn_norm(x, conditioning)
        ffn_out = self.ffn(normed)
        x = x + ffn_out

        return x