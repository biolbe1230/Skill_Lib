"""
ACT (Action Chunking with Transformers) action head.

Reference: Zhao et al. 2023, "Learning Fine-Grained Bimanual Manipulation
with Large-Scale Demonstrations"

Architecture:
  - CVAE encoder (training only):
      action_chunk + obs_token -> Transformer encoder -> mu, logvar -> z (latent_dim)
  - CVAE decoder:
      z + T_pred learnable position queries + obs_tokens (cross-attn) -> Transformer decoder -> action_chunk
  - Loss: L1 reconstruction + beta * KL divergence
  - Inference: z ~ N(0, I), optional temporal ensemble for overlapping chunks
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ACTHead(nn.Module):
    """Action Chunking Transformer with CVAE."""

    def __init__(
        self,
        action_dim: int = 7,
        T_pred: int = 16,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.T_pred = T_pred
        self.d_model = d_model
        self.latent_dim = latent_dim

        # ---------- CVAE Encoder (train only) ----------
        # Projects action chunk to d_model tokens
        self.enc_action_proj = nn.Linear(action_dim, d_model)
        self.enc_action_pos = nn.Parameter(torch.randn(1, T_pred, d_model) * 0.02)

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder for CVAE
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cvae_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        # Mu and logvar projections from CLS token
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

        # ---------- CVAE Decoder ----------
        # Latent z -> d_model token
        self.latent_proj = nn.Linear(latent_dim, d_model)

        # Learnable position queries for decoding
        self.query_pos = nn.Parameter(torch.randn(1, T_pred, d_model) * 0.02)

        # Transformer decoder: queries attend to obs tokens via cross-attention
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        # Output projection
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, action_dim),
        )

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = mu + sigma * epsilon."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(
        self,
        actions: torch.Tensor,      # (B, T_pred, action_dim)
        obs_token: torch.Tensor,    # (B, d_model)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CVAE encoder: encode actions + obs -> mu, logvar.

        Returns: (mu, logvar) each (B, latent_dim)
        """
        B = actions.shape[0]

        # Project actions to d_model tokens
        act_tokens = self.enc_action_proj(actions) + self.enc_action_pos  # (B, T, d_model)

        # Prepend CLS token and obs token
        cls = self.cls_token.expand(B, -1, -1)                 # (B, 1, d_model)
        obs = obs_token.unsqueeze(1)                            # (B, 1, d_model)
        seq = torch.cat([cls, obs, act_tokens], dim=1)          # (B, 2+T, d_model)

        # Encode
        encoded = self.cvae_encoder(seq)  # (B, 2+T, d_model)
        cls_out = encoded[:, 0]           # (B, d_model)

        return self.fc_mu(cls_out), self.fc_logvar(cls_out)

    def decode(
        self,
        z: torch.Tensor,            # (B, latent_dim)
        obs_tokens: torch.Tensor,   # (B, N_obs, d_model)
    ) -> torch.Tensor:
        """CVAE decoder: z + obs -> action chunk.

        Returns: (B, T_pred, action_dim)
        """
        B = z.shape[0]

        # Project z to d_model and prepend to obs tokens as memory
        z_token = self.latent_proj(z).unsqueeze(1)  # (B, 1, d_model)
        memory = torch.cat([z_token, obs_tokens], dim=1)  # (B, 1+N_obs, d_model)

        # Learnable queries
        queries = self.query_pos.expand(B, -1, -1)  # (B, T_pred, d_model)

        # Decode
        decoded = self.decoder(queries, memory)  # (B, T_pred, d_model)

        return self.out_proj(decoded)  # (B, T_pred, action_dim)

    def compute_loss(
        self,
        actions: torch.Tensor,        # (B, T_pred, action_dim) ground-truth
        obs_tokens: torch.Tensor,      # (B, N_obs, d_model) from obs_encoder
        beta: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Compute ACT training loss = L1 recon + beta * KL.

        Args:
            beta: KL divergence weight (use warm-up schedule externally).

        Returns dict with keys: loss, recon_loss, kl_loss
        """
        # Pool obs tokens to single token for encoder
        obs_mean = obs_tokens.mean(dim=1)  # (B, d_model)

        # Encode
        mu, logvar = self.encode(actions, obs_mean)

        # Reparameterize
        z = self._reparameterize(mu, logvar)

        # Decode
        actions_pred = self.decode(z, obs_tokens)

        # Reconstruction loss (L1)
        recon_loss = F.l1_loss(actions_pred, actions)

        # KL divergence: D_KL(q(z|x,c) || N(0,I))
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = recon_loss + beta * kl_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    @torch.inference_mode()
    def predict(
        self,
        obs_tokens: torch.Tensor,    # (B, N_obs, d_model)
    ) -> torch.Tensor:
        """Generate action chunk by sampling z ~ N(0, I).

        Returns: (B, T_pred, action_dim)
        """
        B = obs_tokens.shape[0]
        device = obs_tokens.device

        z = torch.randn(B, self.latent_dim, device=device)
        return self.decode(z, obs_tokens)


def kl_warmup_beta(step: int, warmup_steps: int = 1000, max_beta: float = 10.0) -> float:
    """Linear warm-up for KL beta coefficient."""
    return min(max_beta, max_beta * step / warmup_steps)
