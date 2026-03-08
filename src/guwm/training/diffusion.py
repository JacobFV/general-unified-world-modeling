"""Diffusion training for the Grand Unified World Model.

The world model uses a denoising diffusion objective on the canvas:
- Input: partially observed canvas (some fields have real data)
- Target: fill in the unobserved fields + predict future values
- Noise: applied only to output fields (is_output=True)
- Loss: masked by presence × loss_weight

This is NOT image diffusion. The "image" is a structured canvas where
each position has semantic meaning. The diffusion process learns to
generate coherent latent states that are consistent with the observed
data and the causal structure encoded in the topology.

Multi-frequency diffusion:
- Fields with different periods get noise at different rates
- Fast fields (period=1, markets) get re-noised every step
- Slow fields (period=2304, demographics) hold constant for long stretches
- The model learns the natural timescales of each modality
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineNoiseSchedule:
    """Cosine noise schedule for diffusion."""

    def __init__(self, n_steps: int = 1000, s: float = 0.008):
        self.n_steps = n_steps
        steps = torch.arange(n_steps + 1, dtype=torch.float64)
        f = torch.cos((steps / n_steps + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = f / f[0]
        self.alphas_cumprod = alphas_cumprod.float()
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).float()

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add noise to x_start at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].to(x_start.device)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].to(x_start.device)

        # Reshape for broadcasting: (B,) -> (B, 1, 1)
        while sqrt_alpha.dim() < x_start.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        noisy = sqrt_alpha * x_start + sqrt_one_minus * noise
        return noisy, noise


class MultiFrequencyNoise:
    """Apply noise at different rates to different field frequencies.

    Fields with long periods (slow-changing quantities like demographics)
    should get less noise per training step, because their dynamics are
    inherently smoother. This prevents the model from learning to predict
    noise for slow fields, which would waste capacity.

    Strategy: scale noise magnitude inversely with period.
    """

    def __init__(self, bound_schema, base_noise: float = 1.0):
        self.bound = bound_schema
        self.n_positions = bound_schema.layout.num_positions

        # Build per-position noise scale based on field period
        self.noise_scale = torch.ones(self.n_positions)

        for field_name in bound_schema.field_names:
            bf = bound_schema[field_name]
            spec = bf.spec
            period = getattr(spec, 'period', 1)
            # Scale noise inversely with log of period
            # period=1 → scale=1.0, period=192 → scale~0.2, period=4608 → scale~0.1
            scale = base_noise / (1 + math.log(max(period, 1)))
            indices = bf.indices()
            for idx in indices:
                if idx < self.n_positions:
                    self.noise_scale[idx] = scale

    def apply(self, noise: torch.Tensor) -> torch.Tensor:
        """Scale noise per-position based on field frequencies.

        Args:
            noise: (B, N, d_model) standard normal noise.

        Returns:
            Scaled noise tensor.
        """
        scale = self.noise_scale.to(noise.device)
        return noise * scale.unsqueeze(0).unsqueeze(-1)


class DiffusionWorldModelTrainer:
    """Denoising diffusion trainer for the world model.

    The training objective: given a partially-observed canvas with some
    noise added to output fields, predict the noise (epsilon prediction)
    or the clean canvas (x0 prediction).

    Missing modalities are handled by:
    1. Not adding noise to missing positions (they stay at zero)
    2. Not computing loss on missing positions (presence mask)
    3. The model still generates predictions for missing positions —
       these are the world model's counterfactual estimates

    Args:
        bound_schema: Compiled canvas schema.
        backbone: Transformer backbone.
        schedule: Noise schedule.
        prediction_type: "epsilon" (predict noise) or "x0" (predict clean).
        device: Training device.
    """

    def __init__(
        self,
        bound_schema,
        backbone: nn.Module,
        schedule: CosineNoiseSchedule | None = None,
        prediction_type: str = "epsilon",
        device: str = "cuda",
    ):
        self.bound = bound_schema
        self.backbone = backbone.to(device)
        self.schedule = schedule or CosineNoiseSchedule()
        self.prediction_type = prediction_type
        self.device = device

        # Build masks
        self.loss_weight_mask = bound_schema.layout.loss_weight_mask(device)
        self.multi_freq = MultiFrequencyNoise(bound_schema)

        # Attention mask
        if bound_schema.topology is not None:
            self.attn_mask = bound_schema.topology.to_additive_mask(
                bound_schema.layout, device=device
            )
        else:
            self.attn_mask = None

        # Output mask: which positions participate in diffusion
        output_indices = set(bound_schema.layout.output_mask())
        self.output_mask = torch.zeros(bound_schema.layout.num_positions, device=device)
        for idx in output_indices:
            self.output_mask[idx] = 1.0

    def train_step(
        self,
        x_clean: torch.Tensor,
        presence_mask: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        """Single diffusion training step.

        Args:
            x_clean: (B, N, d_model) clean canvas data.
            presence_mask: (B, N) binary mask — 1 where data exists.
            optimizer: Optimizer.

        Returns:
            Dict with loss and metrics.
        """
        x_clean = x_clean.to(self.device)
        presence_mask = presence_mask.to(self.device)
        B = x_clean.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.schedule.n_steps, (B,), device=self.device)

        # Generate and scale noise
        noise = torch.randn_like(x_clean)
        noise = self.multi_freq.apply(noise)

        # Only add noise to output positions that have data
        noise_mask = presence_mask * self.output_mask.unsqueeze(0)  # (B, N)
        noise = noise * noise_mask.unsqueeze(-1)  # zero noise on missing/input positions

        # Forward diffusion
        x_noisy, _ = self.schedule.q_sample(x_clean, t, noise)

        # Predict
        if self.attn_mask is not None:
            prediction = self.backbone(x_noisy, mask=self.attn_mask)
        else:
            prediction = self.backbone(x_noisy)

        # Compute target
        if self.prediction_type == "epsilon":
            target = noise
        else:  # x0
            target = x_clean

        # Per-position MSE loss
        per_pos_loss = F.mse_loss(prediction, target, reduction="none")
        per_pos_loss = per_pos_loss.mean(dim=-1)  # (B, N)

        # Mask: only compute loss where we have data AND it's an output position
        loss_mask = noise_mask * self.loss_weight_mask.unsqueeze(0)

        n_active = loss_mask.sum()
        if n_active > 0:
            loss = (per_pos_loss * loss_mask).sum() / n_active
        else:
            loss = per_pos_loss.sum() * 0.0

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), max_norm=1.0)
        optimizer.step()

        return {
            "loss": loss.item(),
            "n_active": int(n_active.item()),
            "coverage": float(n_active.item() / max(noise_mask.numel(), 1)),
            "t_mean": t.float().mean().item(),
        }

    @torch.no_grad()
    def sample(
        self,
        x_cond: torch.Tensor,
        cond_mask: torch.Tensor,
        n_steps: int = 50,
    ) -> torch.Tensor:
        """Generate world model predictions via reverse diffusion.

        Args:
            x_cond: (B, N, d_model) conditioning data (observed fields).
            cond_mask: (B, N) binary mask — 1 where conditioning data exists.
            n_steps: Number of denoising steps.

        Returns:
            (B, N, d_model) denoised canvas prediction.
        """
        x_cond = x_cond.to(self.device)
        cond_mask = cond_mask.to(self.device)

        B, N, D = x_cond.shape

        # Start from pure noise for unconditioned positions
        x = torch.randn(B, N, D, device=self.device)
        # Inject conditioning data
        x = x * (1 - cond_mask.unsqueeze(-1)) + x_cond * cond_mask.unsqueeze(-1)

        step_size = self.schedule.n_steps // n_steps

        for i in range(n_steps - 1, -1, -1):
            t = torch.full((B,), i * step_size, device=self.device, dtype=torch.long)

            if self.attn_mask is not None:
                pred = self.backbone(x, mask=self.attn_mask)
            else:
                pred = self.backbone(x)

            if self.prediction_type == "epsilon":
                # DDIM-style update
                alpha = self.schedule.alphas_cumprod[t[0]]
                alpha_prev = self.schedule.alphas_cumprod[max(t[0] - step_size, 0)]
                x0_pred = (x - (1 - alpha).sqrt() * pred) / alpha.sqrt()
                x = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * pred
            else:
                # x0 prediction: direct interpolation
                alpha = self.schedule.alphas_cumprod[t[0]]
                x = pred

            # Re-inject conditioning at every step
            x = x * (1 - cond_mask.unsqueeze(-1)) + x_cond * cond_mask.unsqueeze(-1)

        return x
