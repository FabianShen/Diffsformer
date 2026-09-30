"""
Training pipeline for DiffsFormer factor augmentation (pure PyTorch).

This module handles three phases:
  1. SOURCE PRE-TRAINING: Train DiffsFormer on the full A-share market (source domain)
  2. TARGET EDITING: Use trained DiffsFormer to edit/augment CSI300/CSI800 training data
  3. DOWNSTREAM TRAINING: Train downstream models on augmented data

Integration pattern:
    Alpha158 factors (numpy) → DiffsFormer tensors → edited tensors → downstream training

Paper: DiffsFormer (Gao et al., arXiv:2402.06656v1, 2024)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
from dataclasses import asdict, dataclass
import time

from diffsformer_model import DiffsFormer


# ============================================================================
# 1. Factor Batch
# ============================================================================

@dataclass
class FactorBatch:
    """
    A batch of factor data in DiffsFormer-compatible format.

    All tensors are on the specified device.
    """
    features: torch.Tensor        # (batch, 8, 158) factor sequences
    labels: torch.Tensor          # (batch,) 5-day forward return
    industries: torch.Tensor      # (batch,) industry codes, or zeros if unavailable

    def to(self, device: str) -> "FactorBatch":
        return FactorBatch(
            features=self.features.to(device),
            labels=self.labels.to(device),
            industries=self.industries.to(device),
        )


# ============================================================================
# 2. DiffsFormer Trainer
# ============================================================================

@dataclass
class TrainConfig:
    """Configuration for DiffsFormer training."""
    editing_step: int = 300           # T' — training steps sampled from [1, T']
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 100
    uncondition_prob: float = 0.1     # probability of dropping conditioning
    grad_clip: float = 1.0
    use_loss_guided_noise: bool = True
    noise_boost_factor: float = 1.5
    log_interval: int = 10            # epochs between loss logging
    save_interval: int = 20           # epochs between checkpoint saves
    stop_loss: Optional[float] = 0.965
    min_epochs: int = 5
    amp: bool = True
    data_fingerprint: str = ""


class DiffsFormerTrainer:
    """
    Trainer for DiffsFormer pre-training on the source domain.

    Handles:
      - Training loop with DDPM objective (Eq.6)
      - Predictor-free guidance training (random condition dropping)
      - Loss-guided noise addition (Section V-G-1)
      - Model checkpointing
      - Loss tracking and logging
    """

    def __init__(
        self,
        model: DiffsFormer,
        data: FactorBatch,
        config: TrainConfig,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.data = data
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Cosine annealing LR schedule
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=1e-6
        )

        self.loss_history: list[float] = []
        self.per_sample_losses: Optional[torch.Tensor] = None
        self.amp_enabled = bool(config.amp and str(device).startswith("cuda"))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

    def _loader(self) -> DataLoader:
        ds = TensorDataset(
            self.data.features, self.data.labels, self.data.industries
        )
        return DataLoader(
            ds, batch_size=self.config.batch_size, shuffle=True, drop_last=True,
            pin_memory=str(self.device).startswith("cuda"), num_workers=0,
        )

    def _save_checkpoint(self, path: Path, epoch: int) -> None:
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "loss_history": self.loss_history,
            "train_config": asdict(self.config),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(payload, path)

    def _resume(self, path: str | Path) -> int:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        saved_config = checkpoint.get("train_config", {})
        saved_fingerprint = saved_config.get("data_fingerprint", "")
        if (self.config.data_fingerprint and saved_fingerprint
                and saved_fingerprint != self.config.data_fingerprint):
            raise ValueError("Checkpoint data fingerprint does not match this run")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if "scaler_state_dict" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.loss_history = list(checkpoint.get("loss_history", []))
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        start_epoch = int(checkpoint.get("epoch", 0))
        print(f"[Trainer] Resumed {path} at epoch {start_epoch}")
        return start_epoch

    def train(self, resume_from: Optional[str] = None) -> DiffsFormer:
        """
        Pre-train DiffsFormer on source domain data.

        Args:
            resume_from: optional checkpoint path to resume training
        Returns:
            Trained DiffsFormer model
        """
        start_epoch = self._resume(resume_from) if resume_from else 0
        loader = self._loader()
        self.model.train()

        print(f"[Trainer] Starting pre-training on source domain "
              f"({len(loader.dataset)} samples, {self.config.epochs} epochs)")
        t0 = time.time()

        for epoch in range(start_epoch, self.config.epochs):
            loss_sum, sample_count = 0.0, 0

            for batch_idx, (x0, labels, industries) in enumerate(loader):
                x0 = x0.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                industries = industries.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)

                # Get per-sample losses from previous epoch (for loss-guided noise)
                # On first epoch, use uniform noise
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16,
                    enabled=self.amp_enabled,
                ):
                    loss = self.model.training_loss(
                        x0, labels, industries,
                        editing_step=self.config.editing_step,
                        uncondition_prob=self.config.uncondition_prob,
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite diffusion loss at epoch {epoch + 1}, batch {batch_idx + 1}"
                    )
                self.scaler.scale(loss).backward()
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                loss_sum += loss.item() * len(x0)
                sample_count += len(x0)

            self.scheduler.step()
            avg_loss = loss_sum / sample_count
            self.loss_history.append(avg_loss)

            if (epoch + 1) % self.config.log_interval == 0 or epoch == 0:
                elapsed = time.time() - t0
                print(f"[Trainer] Epoch {epoch + 1:4d}/{self.config.epochs} | "
                      f"Loss: {avg_loss:.6f} | LR: {self.scheduler.get_last_lr()[0]:.2e} | "
                      f"Time: {elapsed:.0f}s")

            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                ckpt_path = self.checkpoint_dir / f"diffsformer_epoch{epoch + 1:04d}.pt"
                self._save_checkpoint(ckpt_path, epoch + 1)
            self._save_checkpoint(self.checkpoint_dir / "diffsformer_latest.pt", epoch + 1)

            if (self.config.stop_loss is not None
                    and epoch + 1 >= self.config.min_epochs
                    and avg_loss <= self.config.stop_loss):
                print(f"[Trainer] Stop-loss threshold reached at epoch {epoch + 1}")
                break

        # Final save
        final_path = self.checkpoint_dir / "diffsformer_final.pt"
        self.model.save(str(final_path))
        print(f"[Trainer] Pre-training complete. Final model: {final_path}")
        return self.model

    def plot_loss(self):
        """Quick loss curve plot (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 4))
            plt.plot(self.loss_history)
            plt.xlabel("Epoch")
            plt.ylabel("L_da (MSE)")
            plt.title("DiffsFormer Training Loss")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("[Trainer] matplotlib not available, skipping loss plot")


# ============================================================================
# 3. Factor Augmentor (Editing Pipeline)
# ============================================================================

class FactorAugmentor:
    """
    Run DiffsFormer editing on target-domain training data.

    For each downstream training epoch, generates a new set of augmented
    factor data while preserving the original labels.

    Key insight from the paper (Section V-D):
      "before the start of training of the predictor in each epoch,
       we generate a new set of stock data. Therefore, the total amount
       of data utilized is n× the original where n equals the total
       number of epochs."

    ``augment_tensor`` is used by the downstream trainer to create a fresh
    edited version of every mini-batch in every epoch, as described in the
    paper, without materializing multi-gigabyte epoch copies.
    """

    def __init__(
        self,
        model: DiffsFormer,
        editing_step: int = 300,
        guidance_strength: float = 2.0,
        ddim_steps: int = 50,
        batch_size: int = 256,
        device: str = "cpu",
        amp: bool = True,
        clip_value: float = 3.0,
    ):
        self.model = model.to(device)
        self.editing_step = editing_step
        self.guidance_strength = guidance_strength
        self.ddim_steps = ddim_steps
        self.batch_size = batch_size
        self.device = device
        self.amp_enabled = bool(amp and str(device).startswith("cuda"))
        self.clip_value = clip_value
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def augment_tensor(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        industries: torch.Tensor,
        noise_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Edit a tensor batch and return float32 features on ``device``."""
        outputs = []
        for start in range(0, len(features), self.batch_size):
            end = min(start + self.batch_size, len(features))
            batch_x = features[start:end].to(self.device, non_blocking=True)
            batch_l = labels[start:end].to(self.device, non_blocking=True)
            batch_i = industries[start:end].to(self.device, non_blocking=True)
            batch_s = None if noise_scale is None else noise_scale[start:end]
            with torch.autocast(
                device_type="cuda", dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                edited = self.model.edit(
                    batch_x, batch_l, batch_i,
                    editing_step=self.editing_step,
                    guidance_strength=self.guidance_strength,
                    ddim_steps=self.ddim_steps,
                    noise_scale=batch_s,
                    clip_value=self.clip_value,
                )
            outputs.append(edited.float())
        return torch.cat(outputs, dim=0)

    def augment(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        industries: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Edit/augment a batch of factor data.

        Paper Section IV-A: "Factor editing with transfer learning"

        Args:
            features: (N, 8, 158) original factor sequences
            labels: (N,) return ratio labels (unchanged)
            industries: (N,) industry codes
        Returns:
            augmented_features: (N, 8, 158) numpy array
            labels: (N,) numpy array (unchanged from input)
        """
        n = features.shape[0]
        all_augmented = []

        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                edited = self.augment_tensor(
                    features[start:end], labels[start:end], industries[start:end]
                )
                all_augmented.append(edited.cpu())

        augmented = torch.cat(all_augmented, dim=0).numpy()
        print(f"[Augmentor] Augmented {n} samples "
              f"(editing_step={self.editing_step}, guidance={self.guidance_strength})")
        return augmented, labels.detach().cpu().numpy()

    def augment_epochs(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        industries: torch.Tensor,
        n_epochs: int,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate a new augmented dataset for each epoch (as paper does).

        WARNING: This is computationally expensive. For quick experiments,
        use augment() once and train downstream on single augmented set.
        """
        results = {}
        for ep in range(n_epochs):
            aug_f, aug_l = self.augment(features, labels, industries)
            results[ep] = (aug_f, aug_l)
            print(f"[Augmentor] Epoch {ep + 1}/{n_epochs} complete")
        return results


# ============================================================================
# 4. Downstream Training
# ============================================================================

class DownstreamTrainer:
    """
    Train a downstream model on (augmented or original) factor data.

    Provides direct PyTorch training of MLP/LSTM/GRU/Transformer
    regressors on original or DiffsFormer-augmented data.
    """

    # Whitelist of directly-implemented downstream models
    SUPPORTED_MODELS = {"mlp", "lstm", "gru", "transformer"}

    def __init__(self, model_name: str = "transformer"):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {sorted(self.SUPPORTED_MODELS)}"
            )
        self.model_name = model_name
        self.model = None

    def _build_direct_model(
        self, n_factors: int, seq_len: int, hidden_dim: int,
        num_layers: int, num_heads: int,
    ) -> nn.Module:
        if self.model_name == "mlp":
            return _MLPRegressor(n_factors * seq_len, hidden_dim)
        if self.model_name in ("lstm", "gru"):
            return _RNNRegressor(
                n_factors, hidden_dim, num_layers, rnn_type=self.model_name
            )
        if self.model_name == "transformer":
            return _TransformerRegressor(
                n_factors, seq_len, hidden_dim, num_layers, num_heads
            )
        print(f"[Downstream] Model '{self.model_name}' is unavailable; using MLP")
        return _MLPRegressor(n_factors * seq_len, hidden_dim)


    def train_direct(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: np.ndarray,
        val_labels: np.ndarray,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        device: str = "cpu",
    ) -> "DownstreamTrainer":
        """
        Direct PyTorch training of downstream models.

        Trains an MLP/LSTM/GRU/Transformer regressor with early stopping.
        """
        train_features = np.asarray(train_features, dtype=np.float32)
        val_features = np.asarray(val_features, dtype=np.float32)
        train_labels = np.asarray(train_labels, dtype=np.float32).reshape(-1)
        val_labels = np.asarray(val_labels, dtype=np.float32).reshape(-1)

        for name, x, y in (
            ("train", train_features, train_labels),
            ("validation", val_features, val_labels),
        ):
            if x.ndim != 3 or len(x) != len(y) or len(x) == 0:
                raise ValueError(f"Invalid {name} data shape: X={x.shape}, y={y.shape}")
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                bad_x = int((~np.isfinite(x)).sum())
                bad_y = int((~np.isfinite(y)).sum())
                raise ValueError(f"Non-finite {name} data: X={bad_x}, y={bad_y}")
        if val_features.shape[1:] != train_features.shape[1:]:
            raise ValueError(
                f"Train/validation feature shapes differ: "
                f"{train_features.shape[1:]} vs {val_features.shape[1:]}"
            )

        # Build model
        n_factors = train_features.shape[2]
        seq_len = train_features.shape[1]
        model = self._build_direct_model(
            n_factors, seq_len, hidden_dim, num_layers, num_heads
        ).to(device)

        # Prepare data
        train_ds = TensorDataset(
            torch.from_numpy(train_features).float(),
            torch.from_numpy(train_labels).float().unsqueeze(1),
        )
        val_ds = TensorDataset(
            torch.from_numpy(val_features).float(),
            torch.from_numpy(val_labels).float().unsqueeze(1),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2)

        # Training
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                     weight_decay=weight_decay)
        criterion = nn.MSELoss()
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        print(f"[Downstream] Training {self.model_name} ({epochs} epochs)...")
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for batch_idx, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite training loss at epoch {epoch + 1}, batch {batch_idx + 1}"
                    )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    batch_loss = criterion(pred, yb)
                    if not torch.isfinite(batch_loss):
                        raise FloatingPointError(f"Non-finite validation loss at epoch {epoch + 1}")
                    val_loss += batch_loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"[Downstream] Early stopping at epoch {epoch + 1}")
                break

        if best_state:
            model.load_state_dict(best_state)

        self.model = model
        print(f"[Downstream] Training complete. Best val loss: {best_val_loss:.6f}")
        return self

    def train_augmented(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        train_industries: np.ndarray,
        val_features: np.ndarray,
        val_labels: np.ndarray,
        augmentor: FactorAugmentor,
        noise_scale: Optional[np.ndarray] = None,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        device: str = "cpu",
        checkpoint_path: Optional[str | Path] = None,
        resume: bool = True,
        data_fingerprint: str = "",
    ) -> "DownstreamTrainer":
        """Train on a newly edited, augmented-only sample in every epoch."""
        train_features = np.asarray(train_features, dtype=np.float32)
        train_labels = np.asarray(train_labels, dtype=np.float32).reshape(-1)
        train_industries = np.asarray(train_industries, dtype=np.int64).reshape(-1)
        val_features = np.asarray(val_features, dtype=np.float32)
        val_labels = np.asarray(val_labels, dtype=np.float32).reshape(-1)
        if not (len(train_features) == len(train_labels) == len(train_industries)):
            raise ValueError("Augmented training arrays have inconsistent lengths")
        if train_features.shape[1:] != val_features.shape[1:]:
            raise ValueError("Train/validation feature shapes differ")
        if noise_scale is None:
            noise_scale = np.ones(len(train_labels), dtype=np.float32)
        noise_scale = np.asarray(noise_scale, dtype=np.float32).reshape(-1)
        if len(noise_scale) != len(train_labels) or not np.isfinite(noise_scale).all():
            raise ValueError("Invalid loss-guided noise scale")

        model = self._build_direct_model(
            train_features.shape[2], train_features.shape[1],
            hidden_dim, num_layers, num_heads,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        criterion = nn.MSELoss()
        train_ds = TensorDataset(
            torch.from_numpy(train_features),
            torch.from_numpy(train_labels).unsqueeze(1),
            torch.from_numpy(train_industries),
            torch.from_numpy(noise_scale),
        )
        val_ds = TensorDataset(
            torch.from_numpy(val_features), torch.from_numpy(val_labels).unsqueeze(1)
        )
        loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            pin_memory=str(device).startswith("cuda"), num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size * 2,
            pin_memory=str(device).startswith("cuda"), num_workers=0,
        )

        start_epoch, best_val_loss, patience_counter = 0, float("inf"), 0
        best_state = None
        checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        if checkpoint_path and resume and checkpoint_path.exists():
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            saved_fingerprint = checkpoint.get("data_fingerprint", "")
            if data_fingerprint and saved_fingerprint and saved_fingerprint != data_fingerprint:
                raise ValueError("Augmented checkpoint data fingerprint mismatch")
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
            best_val_loss = checkpoint["best_val_loss"]
            patience_counter = checkpoint["patience_counter"]
            best_state = checkpoint.get("best_state")
            torch.set_rng_state(checkpoint["torch_rng_state"])
            if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
            print(f"[Downstream] Resumed augmented training at epoch {start_epoch}")

        print(f"[Downstream] Training augmented-only {self.model_name} ({epochs} epochs)...")
        for epoch in range(start_epoch, epochs):
            model.train()
            train_loss = 0.0
            for batch_idx, (xb, yb, ib, sb) in enumerate(loader):
                edited = augmentor.augment_tensor(xb, yb[:, 0], ib, sb)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                pred = model(edited)
                loss = criterion(pred, yb)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite augmented loss at epoch {epoch + 1}, batch {batch_idx + 1}"
                    )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    batch_loss = criterion(model(xb), yb)
                    if not torch.isfinite(batch_loss):
                        raise FloatingPointError("Non-finite augmented validation loss")
                    val_loss += batch_loss.item()
            train_loss /= len(loader)
            val_loss /= len(val_loader)
            print(
                f"[Downstream] Epoch {epoch + 1:3d}/{epochs} | "
                f"train={train_loss:.6f} | val={val_loss:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if checkpoint_path:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_state": best_state,
                    "best_val_loss": best_val_loss,
                    "patience_counter": patience_counter,
                    "data_fingerprint": data_fingerprint,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                }, checkpoint_path)
            if patience_counter >= early_stopping_patience:
                print(f"[Downstream] Early stopping at epoch {epoch + 1}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model = model
        print(f"[Downstream] Augmented training complete. Best val loss: {best_val_loss:.6f}")
        return self

    def predict(
        self, features: np.ndarray, device: str = "cpu", batch_size: int = 2048
    ) -> np.ndarray:
        """Predict return ratios for input factor sequences."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train_direct() first.")
        self.model.eval()
        self.model = self.model.to(device)
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 3 or not np.isfinite(features).all():
            raise ValueError("Prediction features must be a finite 3D array")
        predictions = []
        with torch.no_grad():
            for start in range(0, len(features), batch_size):
                x = torch.from_numpy(features[start:start + batch_size]).to(device)
                predictions.append(self.model(x).cpu().numpy().reshape(-1))
        pred = np.concatenate(predictions) if predictions else np.array([], dtype=np.float32)
        if not np.isfinite(pred).all():
            raise FloatingPointError("Model produced non-finite predictions")
        return pred

    def save_direct_checkpoint(
        self, path: str | Path, model_config: dict[str, Any],
        data_fingerprint: str = "",
    ) -> None:
        if self.model is None:
            raise RuntimeError("No downstream model to save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_name": self.model_name,
            "model_config": model_config,
            "data_fingerprint": data_fingerprint,
            "model_state_dict": self.model.state_dict(),
        }, path)

    def load_direct_checkpoint(
        self, path: str | Path, device: str = "cpu",
        expected_fingerprint: str = "",
    ) -> "DownstreamTrainer":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if checkpoint["model_name"] != self.model_name:
            raise ValueError("Downstream checkpoint model type mismatch")
        saved_fingerprint = checkpoint.get("data_fingerprint", "")
        if expected_fingerprint and saved_fingerprint != expected_fingerprint:
            raise ValueError("Downstream checkpoint data fingerprint mismatch")
        config = checkpoint["model_config"]
        self.model = self._build_direct_model(
            config["n_factors"], config["seq_len"], config["hidden_dim"],
            config["num_layers"], config["num_heads"],
        ).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return self


# ============================================================================
# 5. Downstream model implementations
# ============================================================================

class _MLPRegressor(nn.Module):
    """Simple MLP for regression on flattened factor sequences."""
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class _RNNRegressor(nn.Module):
    """LSTM or GRU for factor sequence regression."""
    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 num_layers: int = 2, rnn_type: str = "lstm"):
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(input_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)  # (batch, seq, hidden)
        return self.head(out[:, -1, :])  # use last timestep


class _TransformerRegressor(nn.Module):
    """Transformer encoder for factor sequences."""
    def __init__(self, input_dim: int, seq_len: int, hidden_dim: int = 256,
                 num_layers: int = 2, num_heads: int = 8):
        super().__init__()
        if hidden_dim % 2 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be even and divisible by num_heads")
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            * (-np.log(10000.0) / hidden_dim)
        )
        positional = torch.zeros(1, seq_len, hidden_dim)
        positional[0, :, 0::2] = torch.sin(position * scale)
        positional[0, :, 1::2] = torch.cos(position * scale)
        self.register_buffer("positional", positional)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_proj(x) + self.positional[:, :x.shape[1]]
        x = self.encoder(x)
        return self.head(x[:, -1])
