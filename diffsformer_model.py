"""
DiffsFormer: A Diffusion Transformer for Stock Factor Augmentation.

Paper: DiffsFormer: A Diffusion Transformer on Stock Factor Augmentation
       (Gao et al., arXiv:2402.06656v1, 2024)

Architecture:
  - DDPM (Denoising Diffusion Probabilistic Model) with variance-preserving schedule
  - Transformer backbone (replaces UNet, following DiT design)
  - Adaptive Layer Norm (ALN) for conditioning injection
  - Predictor-free guidance (label + industry conditions)
  - DDIM accelerated sampling
  - Transfer-learning editing (pre-train on source domain, edit target domain)
  - Loss-guided noise addition

Input shape:  (batch, 8, 158)  — 8-day lookback of Alpha158 factors
Output shape: (batch, 8, 158)  — predicted noise ε (same shape as input)
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


# ============================================================================
# 1. Time Step Embedding (Sinusoidal Positional Encoding)
# ============================================================================

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


# ============================================================================
# 2. Condition Embedding (Label + Industry)
# ============================================================================

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

        # Label embedding: scalar → vector (via learned projection)
        self.label_proj = nn.Sequential(
            nn.Linear(1, label_embed_dim),
            nn.SiLU(),
            nn.Linear(label_embed_dim, label_embed_dim),
        )

        # Industry embedding: categorical → vector
        self.industry_embed = nn.Embedding(num_industries, industry_embed_dim)

        # Null embeddings (learnable "no condition" tokens)
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


# ============================================================================
# 3. Adaptive Layer Norm (ALN)
# ============================================================================

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


# ============================================================================
# 4. Transformer Block with ALN
# ============================================================================

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


# ============================================================================
# 5. DiffsFormer: Full Diffusion Transformer
# ============================================================================

@dataclass
class DiffusionSchedule:
    """Pre-computed DDPM schedule parameters."""
    betas: torch.Tensor          # (T,) noise schedule
    alphas: torch.Tensor         # (T,) = 1 - betas
    alpha_bars: torch.Tensor     # (T,) cumulative product of alphas
    alpha_bar_prev: torch.Tensor # (T,) previous alpha_bar
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor
    sqrt_recip_alphas: torch.Tensor


def make_beta_schedule(
    T: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> torch.Tensor:
    """Linear beta schedule (variance-preserving DDPM)."""
    return torch.linspace(beta_start, beta_end, T)


class DiffsFormer(nn.Module):
    """
    DiffsFormer: Diffusion Transformer for Stock Factor Augmentation.

    Architecture:
      - Transformer backbone with ALN conditioning
      - Predictor-free guidance (label + industry)
      - DDIM accelerated sampling
      - Transfer-learning editing pipeline

    Input/Output:
      - x: (batch, seq_len=8, n_factors=158) stock factor sequences
      - t: (batch,) diffusion timesteps
      - labels: (batch,) return ratio (optional, for conditioning)
      - industries: (batch,) industry codes (optional, for conditioning)

    Usage:
        model = DiffsFormer(seq_len=8, n_factors=158)
        noise_pred = model(x_noisy, t, labels, industries)  # training
        edited = model.edit(x_target, labels, industries, T_prime=300)  # inference
        model.save("diffsformer.pt")
        model = DiffsFormer.load("diffsformer.pt")
    """

    def __init__(
        self,
        seq_len: int = 8,
        n_factors: int = 158,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        total_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        num_industries: int = 28,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_factors = n_factors
        self.hidden_dim = hidden_dim
        self.total_timesteps = total_timesteps
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_industries = num_industries

        # ── Input projection: (seq_len, n_factors) → (seq_len, hidden_dim) ──
        self.input_proj = nn.Linear(n_factors, hidden_dim)

        # ── Positional encoding for sequence positions (0..seq_len-1) ──
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)

        # ── Time embedding ──
        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── Condition embedding (label + industry) ──
        self.condition_embed = ConditionEmbedding(
            hidden_dim=hidden_dim,
            num_industries=num_industries,
        )

        # ── Transformer blocks ──
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # ── Output projection: hidden_dim → n_factors ──
        self.output_proj = nn.Sequential(
            AdaptiveLayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_factors),
        )
        # Zero-init output projection
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

        # ── Build diffusion schedule ──
        self.register_buffer("betas", make_beta_schedule(total_timesteps, beta_start, beta_end))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("alpha_bar_prev", F.pad(self.alpha_bars[:-1], (1, 0), value=1.0))
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(self.alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - self.alpha_bars))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / self.alphas))

    # ── Forward diffusion ────────────────────────────────────────────────

    def diffuse(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Forward diffusion: x0 → xt by adding noise.
        q(xt|x0) = N(xt; sqrt(ᾱ_t)·x0, (1-ᾱ_t)·I)

        Eq.(4) in paper: xt = sqrt(ᾱ_t)·x0 + sqrt(1-ᾱ_t)·ε

        Args:
            x0: (batch, seq_len, n_factors) clean data
            t: (batch,) timesteps
            noise: optional pre-sampled noise
        Returns:
            xt: (batch, seq_len, n_factors) noisy data
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)

        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise, noise

    # ── Noise prediction (transformer forward pass) ──────────────────────

    def predict_noise(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        industries: Optional[torch.Tensor] = None,
        drop_condition: bool | torch.Tensor = False,
    ) -> torch.Tensor:
        """
        Predict the noise ε added to xt at timestep t.

        This is ε_θ(xt, t, c) in the paper — the trainable denoising network.

        Args:
            xt: (batch, seq_len, n_factors) noisy input
            t: (batch,) timesteps
            labels: (batch,) optional return ratio conditioning
            industries: (batch,) optional industry conditioning
            drop_condition: scalar or per-sample mask for the null condition
        Returns:
            (batch, seq_len, n_factors) predicted noise ε
        """
        batch, seq_len, _ = xt.shape

        # Project input
        h = self.input_proj(xt)  # (batch, seq_len, hidden_dim)

        # Add positional encoding
        h = h + self.pos_embed[:, :seq_len, :]

        # Time embedding
        t_emb = self.time_embed(t)        # (batch, hidden_dim)
        t_emb = self.time_proj(t_emb)     # (batch, hidden_dim)

        # Condition embedding
        if labels is not None and industries is not None:
            c_emb = self.condition_embed(
                labels, industries, drop_condition=drop_condition
            )
            conditioning = t_emb + c_emb
        else:
            conditioning = t_emb

        # Transformer blocks with conditioning
        for block in self.blocks:
            h = block(h, conditioning)

        # Output projection
        h = self.output_proj[0](h, conditioning)  # ALN
        eps_pred = self.output_proj[1](h)          # Linear

        return eps_pred

    def guided_noise(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
        industries: torch.Tensor,
        guidance_strength: float,
    ) -> torch.Tensor:
        """Predict conditional and null noise in one batched forward pass."""
        if guidance_strength == 1.0:
            return self.predict_noise(xt, t, labels, industries)
        batch = xt.shape[0]
        both = self.predict_noise(
            torch.cat([xt, xt], dim=0),
            torch.cat([t, t], dim=0),
            torch.cat([labels, labels], dim=0),
            torch.cat([industries, industries], dim=0),
            drop_condition=torch.cat([
                torch.ones(batch, dtype=torch.bool, device=xt.device),
                torch.zeros(batch, dtype=torch.bool, device=xt.device),
            ]),
        )
        eps_uncond, eps_cond = both.chunk(2, dim=0)
        return eps_uncond + guidance_strength * (eps_cond - eps_uncond)

    # ── Training loss ────────────────────────────────────────────────────

    def training_loss(
        self,
        x0: torch.Tensor,
        labels: torch.Tensor,
        industries: torch.Tensor,
        editing_step: int = 300,
        uncondition_prob: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute the DDPM training loss L_da (Eq.6).

        L_da = E[||ε - ε_θ(xt, t)||²]

        Training steps t are sampled from Uniform{0, ..., editing_step - 1}
        instead of Uniform{1, ..., T} for time efficiency.
        (Paper Section IV-A: "Time efficiency improvements")

        Randomly drops conditions with probability uncondition_prob
        to enable predictor-free guidance at inference time.
        """
        batch = x0.shape[0]
        device = x0.device

        if not 1 <= editing_step <= self.total_timesteps:
            raise ValueError("editing_step must be within the diffusion schedule")
        t = torch.randint(0, editing_step, (batch,), device=device)

        # Sample noise
        noise = torch.randn_like(x0)

        # Forward diffuse
        xt, _ = self.diffuse(x0, t, noise)

        # Predictor-free guidance drops the complete condition independently
        # for each sample, not once for the entire mini-batch.
        drop_condition = torch.rand(batch, device=device) < uncondition_prob
        eps_pred = self.predict_noise(
            xt, t, labels, industries, drop_condition=drop_condition
        )

        # MSE loss
        loss = F.mse_loss(eps_pred, noise)
        return loss

    # ── DDIM Sampling ────────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(
        self,
        shape: Tuple[int, int, int],
        labels: torch.Tensor,
        industries: torch.Tensor,
        sampling_steps: int = 50,
        guidance_strength: float = 2.0,
    ) -> torch.Tensor:
        """
        Generate samples from pure noise using DDIM accelerated sampling.

        This is used for UNCONDITIONAL generation (from scratch).
        For the editing pipeline, use self.edit() instead.

        DDIM deterministic sampling (η=0):
        x_{t-1} = sqrt(ᾱ_{t-1})·x̂_0 + sqrt(1-ᾱ_{t-1})·ε_θ

        Where x̂_0 = (x_t - sqrt(1-ᾱ_t)·ε_θ) / sqrt(ᾱ_t)

        Paper: Eq.(7), Section IV-A "Inference acceleration"

        Args:
            shape: (batch, seq_len, n_factors)
            labels: (batch,) for conditioning
            industries: (batch,) for conditioning
            sampling_steps: number of DDIM steps (l << T)
            guidance_strength: ω_free in Eq.(11), typically 2.0
        Returns:
            Generated samples from learned distribution
        """
        batch = shape[0]
        device = self.sqrt_alpha_bars.device

        # Create DDIM step schedule (uniformly spaced)
        step_indices = torch.linspace(
            self.total_timesteps - 1, 0, sampling_steps, device=device
        ).long()

        # Start from pure noise
        x = torch.randn(shape, device=device)

        for i in range(sampling_steps):
            t = step_indices[i]
            t_batch = torch.full((batch,), t, device=device)

            # Predictor-free guidance (Eq.11)
            # ε̂ = ε_θ(xt, ∅) + ω · (ε_θ(xt, c) - ε_θ(xt, ∅))
            eps = self.guided_noise(
                x, t_batch, labels, industries, guidance_strength
            )

            # Predict x0 from noise
            alpha_bar_t = self.alpha_bars[t]
            sqrt_alpha_bar_t = self.sqrt_alpha_bars[t]
            sqrt_one_minus_t = self.sqrt_one_minus_alpha_bars[t]
            x0_pred = (x - sqrt_one_minus_t * eps) / sqrt_alpha_bar_t

            # Clamp to reasonable range (factor values after robust z-score)
            x0_pred = torch.clamp(x0_pred, -10.0, 10.0)

            # Get next alpha_bar
            if i < sampling_steps - 1:
                t_next = step_indices[i + 1]
                alpha_bar_next = self.alpha_bars[t_next]
                sqrt_alpha_bar_next = torch.sqrt(alpha_bar_next)
                sqrt_one_minus_next = torch.sqrt(1.0 - alpha_bar_next)

                # DDIM deterministic update
                x = sqrt_alpha_bar_next * x0_pred + sqrt_one_minus_next * eps
            else:
                x = x0_pred

        return x

    # ── Transfer-Learning Editing Pipeline (Key Innovation) ──────────────

    @torch.no_grad()
    def edit(
        self,
        x_target: torch.Tensor,
        labels: torch.Tensor,
        industries: torch.Tensor,
        editing_step: int = 300,
        guidance_strength: float = 2.0,
        ddim_steps: int = 50,
        noise_scale: Optional[torch.Tensor] = None,
        clip_value: float = 3.0,
    ) -> torch.Tensor:
        """
        Edit target-domain samples by corrupting for T' steps and denoising.

        This is THE KEY INNOVATION of DiffsFormer (Section IV-A):
          1. Start from real target-domain data x0_target
          2. Forward-diffuse for T' steps (T' << T): x0 → x_{T'}
          3. Reverse-denoise back to x̂0 using DDIM

        The edited sample x̂0 retains fidelity to the target domain
        while incorporating knowledge from the source domain (learned
        during pre-training).

        T' controls editing strength:
          - Larger T' → more source-domain influence, less target fidelity
          - Smaller T' → less change, closer to original

        Optimal T' = 300 per Table VI.

        Args:
            x_target: (batch, seq_len, n_factors) real target-domain samples
            labels: (batch,) return ratio labels (unchanged through editing)
            industries: (batch,) industry codes
            editing_step: T' — number of forward steps (default 300)
            guidance_strength: ω_free for predictor-free guidance
            ddim_steps: number of DDIM steps for reverse process
        Returns:
            (batch, seq_len, n_factors) edited samples with same labels
        """
        if not 1 <= editing_step <= self.total_timesteps:
            raise ValueError("editing_step must be within the diffusion schedule")
        if not 1 <= ddim_steps <= editing_step:
            raise ValueError("ddim_steps must be between 1 and editing_step")
        batch = x_target.shape[0]
        device = x_target.device

        # Step 1: Forward-diffuse for T' steps to get seed point
        max_t = editing_step - 1
        t_prime = torch.full((batch,), max_t, device=device, dtype=torch.long)
        noise = torch.randn_like(x_target)
        if noise_scale is not None:
            scale = noise_scale.to(device=device, dtype=x_target.dtype).reshape(batch, 1, 1)
            noise = noise * scale
        seed, _ = self.diffuse(x_target, t_prime, noise)

        # Step 2: Create DDIM step schedule from T' down to 0
        step_indices = torch.linspace(max_t, 0, ddim_steps, device=device).round().long()

        x = seed
        for i, t_val in enumerate(step_indices):
            t = torch.full((batch,), t_val.item(), device=device)

            # Predictor-free guidance
            eps = self.guided_noise(
                x, t, labels, industries, guidance_strength
            )

            # Predict x0
            sqrt_alpha_bar_t = self.sqrt_alpha_bars[t_val]
            sqrt_one_minus_t = self.sqrt_one_minus_alpha_bars[t_val]
            x0_pred = (x - sqrt_one_minus_t * eps) / sqrt_alpha_bar_t
            x0_pred = torch.clamp(x0_pred, -clip_value, clip_value)

            # Get next step
            if i < len(step_indices) - 1:
                t_next = step_indices[i + 1]
                alpha_bar_next = self.alpha_bars[t_next]
                sqrt_alpha_bar_next = torch.sqrt(alpha_bar_next)
                sqrt_one_minus_next = torch.sqrt(1.0 - alpha_bar_next)
                x = sqrt_alpha_bar_next * x0_pred + sqrt_one_minus_next * eps
            else:
                x = x0_pred  # final step: return x̂0

        return x

    # ── Loss-Guided Noise ────────────────────────────────────────────────

    def loss_guided_diffuse(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        per_sample_loss: torch.Tensor,
        noise_boost_factor: float = 1.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add stronger noise to samples with lower training loss.

        Paper Section V-G-1: "Loss-guided Noise Addition"
        Easily-fitted samples (low loss) receive stronger noise,
        preventing overfitting and reducing model volatility.

        Args:
            x0: (batch, seq_len, n_factors)
            t: (batch,) base timesteps
            per_sample_loss: (batch,) training loss per sample
            noise_boost_factor: multiplier for low-loss samples
        Returns:
            xt, noise
        """
        noise = torch.randn_like(x0)

        # Compute noise boost: low loss → boost > 1, high loss → boost ≈ 1
        loss_median = per_sample_loss.median()
        loss_ratio = loss_median / (per_sample_loss + 1e-8)  # >1 for low loss
        boost = 1.0 + (noise_boost_factor - 1.0) * torch.tanh(loss_ratio - 1.0)
        # Clamp boost to [1.0, noise_boost_factor]
        boost = boost.clamp(1.0, noise_boost_factor).view(-1, 1, 1)

        boosted_noise = noise * boost

        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)

        xt = sqrt_alpha_bar * x0 + sqrt_one_minus * boosted_noise
        return xt, boosted_noise

    # ── Model Save / Load ────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save model weights and config to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": {
                "seq_len": self.seq_len,
                "n_factors": self.n_factors,
                "hidden_dim": self.hidden_dim,
                "num_layers": len(self.blocks),
                "num_heads": self.blocks[0].attention.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "dropout": self.dropout,
                "total_timesteps": self.total_timesteps,
                "beta_start": self.beta_start,
                "beta_end": self.beta_end,
                "num_industries": self.num_industries,
            },
        }
        torch.save(checkpoint, path)
        print(f"[DiffsFormer] Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "DiffsFormer":
        """Load model weights and config from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        config = checkpoint["config"]
        model = cls(
            seq_len=config["seq_len"],
            n_factors=config["n_factors"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            mlp_ratio=config.get("mlp_ratio", 4.0),
            dropout=config.get("dropout", 0.1),
            total_timesteps=config["total_timesteps"],
            beta_start=config.get("beta_start", 1e-4),
            beta_end=config.get("beta_end", 0.02),
            num_industries=config.get("num_industries", 28),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[DiffsFormer] Model loaded from {path}")
        return model
