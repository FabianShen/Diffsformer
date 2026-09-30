import torch.nn as nn
import torch

class ConditionEmbedding(nn.Module):
    """
    Embed label (continuous return ratio) and industry (categorical sector code)
    into a combined condition vector for predictor-free guidance.

    Paper: Section IV-B "Conditional Diffusion Augmentation"
           Eq.(11) — predictor-free guidance

    The label is a scalar (5-day forward return ratio).
    The industry is an integer code (e.g., 0-27 for 28 Shenwan industries).
    During training the combined condition is dropped per sample and replaced
    with learnable null embeddings. This supports predictor-free guidance.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_industries: int = 28,
        label_embed_dim: int = 64,
        industry_embed_dim: int = 32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Label embedding:
        self.label_proj = nn.Sequential(
            nn.Linear(1, label_embed_dim),
            nn.SiLU(),
            nn.Linear(label_embed_dim, label_embed_dim),
        )

        # Industry embedding:
        self.industry_embed = nn.Embedding(num_industries, industry_embed_dim)

        # Null embeddings
        self.null_label_embed = nn.Parameter(torch.randn(1, label_embed_dim))
        self.null_industry_embed = nn.Parameter(torch.randn(1, industry_embed_dim))

        # Combine and project to hidden_dim
        combined_dim = label_embed_dim + industry_embed_dim
        self.output_proj = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        labels: torch.Tensor,             # (batch,) or (batch, 1)
        industries: torch.Tensor,         # (batch,) integer codes
        drop_condition: bool | torch.Tensor = False,
    ) -> torch.Tensor:
        """
        Args:
            labels: (batch,) return ratio values
            industries: (batch,) industry codes (int)
            drop_condition: scalar or per-sample mask selecting null conditions
        Returns:
            (batch, hidden_dim) combined condition embedding
        """
        batch = labels.shape[0]

        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        label_emb = self.label_proj(labels.float())
        industry_emb = self.industry_embed(industries.long())

        mask = torch.as_tensor(drop_condition, device=labels.device, dtype=torch.bool)
        if mask.ndim == 0:
            mask = mask.expand(batch)
        if mask.shape != (batch,):
            raise ValueError(f"drop_condition must be scalar or shape ({batch},)")
        label_emb = torch.where(
            mask[:, None], self.null_label_embed.expand(batch, -1), label_emb
        )
        industry_emb = torch.where(
            mask[:, None], self.null_industry_embed.expand(batch, -1), industry_emb
        )

        # Combine
        combined = torch.cat([label_emb, industry_emb], dim=-1)
        return self.output_proj(combined)
