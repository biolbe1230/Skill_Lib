"""
Diffusion Policy action head (DDPM + Transformer).

Reference: Chi et al. 2023, "Diffusion Policy: Visuomotor Policy Learning
via Action Diffusion"

Architecture:
  - Noise prediction network: Transformer decoder
    - Input: noisy action chunk (B, T_pred, action_dim) + sinusoidal timestep emb
    - Observation tokens injected via cross-attention
    - Output: predicted noise epsilon (B, T_pred, action_dim)
  - DDPM with cosine beta schedule, 100 training steps
  - DDIM sampling with 10 steps at inference
  - EMA on network weights (decay=0.999)
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --------------------------------------------------------------------------- #
#  Cosine beta schedule (improved DDPM)
# --------------------------------------------------------------------------- #

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal 2021."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, 0.0001, 0.9999).float()


# --------------------------------------------------------------------------- #
#  Sinusoidal timestep embedding
# --------------------------------------------------------------------------- #

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,) integer timesteps -> (B, dim)."""
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device, dtype=torch.float32) * -emb)
        emb = x.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# --------------------------------------------------------------------------- #
#  Transformer-based noise prediction network
# --------------------------------------------------------------------------- #

class TransformerNoiseNet(nn.Module):
    """Predicts noise using a Transformer decoder with cross-attention to obs."""

    def __init__(
        self,
        action_dim: int = 7,
        T_pred: int = 16,
        d_model: int = 512,
        nhead: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.T_pred = T_pred
        self.d_model = d_model

        # Action embedding
        self.action_proj = nn.Linear(action_dim, d_model)

        # Timestep embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Learnable positional embedding for action sequence
        self.pos_emb = nn.Parameter(torch.randn(1, T_pred, d_model) * 0.02)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection back to action space
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, action_dim),
        )

    def forward(
        self,
        noisy_actions: torch.Tensor,   # (B, T_pred, action_dim)
        timestep: torch.Tensor,        # (B,) int
        obs_tokens: torch.Tensor,      # (B, N_obs, d_model)
    ) -> torch.Tensor:
        """Predict noise epsilon. Returns (B, T_pred, action_dim)."""
        B = noisy_actions.shape[0]

        # Embed noisy actions
        x = self.action_proj(noisy_actions)  # (B, T_pred, d_model)
        x = x + self.pos_emb[:, :self.T_pred, :]

        # Add timestep embedding (broadcast to all tokens)
        t_emb = self.time_emb(timestep)  # (B, d_model)
        x = x + t_emb.unsqueeze(1)

        # Transformer decode with cross-attention to obs
        x = self.decoder(x, obs_tokens)  # (B, T_pred, d_model)

        return self.out_proj(x)  # (B, T_pred, action_dim)


# --------------------------------------------------------------------------- #
#  EMA helper
# --------------------------------------------------------------------------- #

class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply(self, model: nn.Module):
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


# --------------------------------------------------------------------------- #
#  Diffusion Policy Head
# --------------------------------------------------------------------------- #

class DiffusionHead(nn.Module):
    """DDPM-based action head with Transformer noise prediction."""

    def __init__(
        self,
        action_dim: int = 7,
        T_pred: int = 16,
        d_model: int = 512,
        nhead: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
        num_train_steps: int = 100,
        num_infer_steps: int = 10,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.T_pred = T_pred
        self.num_train_steps = num_train_steps
        self.num_infer_steps = num_infer_steps

        self.noise_net = TransformerNoiseNet(
            action_dim=action_dim,
            T_pred=T_pred,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Pre-compute diffusion schedule
        betas = cosine_beta_schedule(num_train_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # For DDPM reverse sampling
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    def compute_loss(
        self,
        actions: torch.Tensor,        # (B, T_pred, action_dim) ground-truth
        obs_tokens: torch.Tensor,      # (B, N_obs, d_model) from obs_encoder
    ) -> torch.Tensor:
        """Compute DDPM training loss (MSE on predicted noise)."""
        B = actions.shape[0]
        device = actions.device

        # Sample random timesteps
        t = torch.randint(0, self.num_train_steps, (B,), device=device, dtype=torch.long)

        # Sample noise
        noise = torch.randn_like(actions)

        # Forward diffusion: q(x_t | x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(B, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)
        noisy_actions = sqrt_alpha * actions + sqrt_one_minus_alpha * noise

        # Predict noise
        noise_pred = self.noise_net(noisy_actions, t, obs_tokens)

        return F.mse_loss(noise_pred, noise)

    @torch.inference_mode()
    def predict(
        self,
        obs_tokens: torch.Tensor,    # (B, N_obs, d_model)
        use_ddim: bool = True,
    ) -> torch.Tensor:
        """Generate action chunk via reverse diffusion.

        Returns: (B, T_pred, action_dim)
        """
        B = obs_tokens.shape[0]
        device = obs_tokens.device

        # Start from pure noise
        x = torch.randn(B, self.T_pred, self.action_dim, device=device)

        if use_ddim:
            x = self._ddim_sample(x, obs_tokens)
        else:
            x = self._ddpm_sample(x, obs_tokens)

        return x

    def _ddpm_sample(self, x: torch.Tensor, obs_tokens: torch.Tensor) -> torch.Tensor:
        """Full DDPM reverse sampling (num_train_steps iterations)."""
        for i in reversed(range(self.num_train_steps)):
            t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
            noise_pred = self.noise_net(x, t, obs_tokens)

            mean = (
                self.posterior_mean_coef1[i] * x
                + self.posterior_mean_coef2[i] * noise_pred
            )
            # Correct DDPM: x_0 = (x_t - sqrt(1-alpha_bar)*eps) / sqrt(alpha_bar)
            # Then posterior mean = coef1 * x_0 + coef2 * x_t
            # Actually, let's use the standard formulation:
            alpha_bar_t = self.alphas_cumprod[i]
            alpha_t = self.alphas[i]
            beta_t = self.betas[i]

            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            x0_pred = torch.clamp(x0_pred, -1, 1)

            if i > 0:
                alpha_bar_prev = self.alphas_cumprod[i - 1]
                posterior_mean = (
                    torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t) * x0_pred
                    + torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t) * x
                )
                posterior_var = self.posterior_variance[i]
                noise = torch.randn_like(x)
                x = posterior_mean + torch.sqrt(posterior_var) * noise
            else:
                x = x0_pred

        return x

    def _ddim_sample(self, x: torch.Tensor, obs_tokens: torch.Tensor) -> torch.Tensor:
        """DDIM accelerated sampling (num_infer_steps iterations)."""
        # Uniformly spaced timesteps
        step_size = self.num_train_steps // self.num_infer_steps
        timesteps = list(range(0, self.num_train_steps, step_size))
        timesteps = list(reversed(timesteps))

        for i, t_cur in enumerate(timesteps):
            t = torch.full((x.shape[0],), t_cur, device=x.device, dtype=torch.long)
            noise_pred = self.noise_net(x, t, obs_tokens)

            alpha_bar_t = self.alphas_cumprod[t_cur]

            # Predict x_0
            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            x0_pred = torch.clamp(x0_pred, -1, 1)

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_bar_next = self.alphas_cumprod[t_next]
            else:
                alpha_bar_next = torch.tensor(1.0, device=x.device)

            # DDIM deterministic step
            x = (
                torch.sqrt(alpha_bar_next) * x0_pred
                + torch.sqrt(1 - alpha_bar_next) * noise_pred
            )

        return x
