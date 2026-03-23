"""
Shared observation encoder for action heads.

Each modality becomes an independent token for cross-attention.
d_model = 512 to match ResNet-18 output, avoiding lossy compression.

  - agentview  -> ResNet-18 (frozen) -> avgpool -> 512              (1 token, as-is)
  - wrist      -> ResNet-18 (frozen) -> avgpool -> 512              (1 token, as-is)
  - proprio(8) -> Linear(8 -> 512)                                  (1 token)

Per frame: 3 tokens.  With T_obs frames: 3 * T_obs tokens total.
Learnable modality embeddings distinguish agentview / wrist / proprio.
Learnable temporal embeddings distinguish frames 0..T_obs-1.

Only proprio projection and embeddings are trainable; ResNet is frozen.
Images pass through without any projection layer.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# ResNet-18 avgpool output dim; this IS our d_model.
RESNET_DIM = 512


class ObsEncoder(nn.Module):
    """Encodes multi-view images + proprioception into per-modality tokens."""

    N_MODALITIES = 3  # agentview, wrist, proprio
    d_model = RESNET_DIM  # 512, fixed to match ResNet output

    def __init__(
        self,
        proprio_dim: int = 8,
        T_obs: int = 2,
        freeze_resnet: bool = True,
    ):
        super().__init__()
        self.T_obs = T_obs
        self.proprio_dim = proprio_dim

        # ---------- Visual backbone (frozen) ----------
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.visual_backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            resnet.avgpool,
            nn.Flatten(1),  # -> (B, 512)
        )
        if freeze_resnet:
            for p in self.visual_backbone.parameters():
                p.requires_grad_(False)
            self.visual_backbone.eval()
        self.freeze_resnet = freeze_resnet

        # ---------- Proprio projection (trainable) ----------
        # 8 -> 512.  Single linear to keep original semantics interpretable.
        # No projection needed for images — ResNet already outputs 512.
        self.proprio_proj = nn.Linear(proprio_dim, self.d_model)

        # ---------- Learnable embeddings ----------
        self.modality_emb = nn.Embedding(self.N_MODALITIES, self.d_model)
        self.temporal_emb = nn.Embedding(T_obs, self.d_model)

    def train(self, mode: bool = True):
        """Override to keep ResNet in eval mode when frozen."""
        super().train(mode)
        if self.freeze_resnet:
            self.visual_backbone.eval()
        return self

    @property
    def num_tokens(self) -> int:
        """Total number of obs tokens produced: 3 * T_obs."""
        return self.N_MODALITIES * self.T_obs

    def forward_tokens(
        self,
        agentview: torch.Tensor,  # (B, T_obs, 3, H, W)
        wrist: torch.Tensor,      # (B, T_obs, 3, H, W)
        proprio: torch.Tensor,    # (B, T_obs, proprio_dim)
    ) -> torch.Tensor:
        """Produce per-modality, per-frame tokens for cross-attention.

        Returns: (B, 3 * T_obs, d_model)
            Token order: [av_t0, wr_t0, pr_t0, av_t1, wr_t1, pr_t1, ...]
        """
        B, T = agentview.shape[:2]

        # --- Visual features (frozen ResNet, already 512-d = d_model) ---
        av_flat = agentview.reshape(B * T, *agentview.shape[2:])
        wr_flat = wrist.reshape(B * T, *wrist.shape[2:])

        with torch.set_grad_enabled(not self.freeze_resnet):
            av_feat = self.visual_backbone(av_flat)  # (B*T, 512)
            wr_feat = self.visual_backbone(wr_flat)  # (B*T, 512)

        # No projection — ResNet output is already d_model (512)
        av_tok = av_feat.reshape(B, T, self.d_model)  # (B, T, 512)
        wr_tok = wr_feat.reshape(B, T, self.d_model)  # (B, T, 512)

        return self._add_embeddings(av_tok, wr_tok, proprio)

    def forward_tokens_from_feat(
        self,
        agentview_feat: torch.Tensor,  # (B, T_obs, 512) pre-extracted
        wrist_feat: torch.Tensor,      # (B, T_obs, 512) pre-extracted
        proprio: torch.Tensor,         # (B, T_obs, proprio_dim)
    ) -> torch.Tensor:
        """Same as forward_tokens but using pre-extracted ResNet features.

        Skips ResNet completely — 3-5x faster for training.
        Returns: (B, 3 * T_obs, d_model)
        """
        return self._add_embeddings(agentview_feat, wrist_feat, proprio)

    def _add_embeddings(
        self,
        av_tok: torch.Tensor,   # (B, T, 512)
        wr_tok: torch.Tensor,   # (B, T, 512)
        proprio: torch.Tensor,  # (B, T, proprio_dim)
    ) -> torch.Tensor:
        B, T = av_tok.shape[:2]

        # Proprio: 8 -> 512
        pr_tok = self.proprio_proj(proprio)      # (B, T, 512)

        # --- Build device indices for embeddings ---
        device = av_tok.device
        t_idx = torch.arange(T, device=device)                    # (T,)
        mod_av = torch.zeros(T, dtype=torch.long, device=device)  # 0
        mod_wr = torch.ones(T, dtype=torch.long, device=device)   # 1
        mod_pr = torch.full((T,), 2, dtype=torch.long, device=device)  # 2

        t_emb = self.temporal_emb(t_idx)          # (T, d_model)
        m_av_emb = self.modality_emb(mod_av)      # (T, d_model)
        m_wr_emb = self.modality_emb(mod_wr)      # (T, d_model)
        m_pr_emb = self.modality_emb(mod_pr)      # (T, d_model)

        # Add modality + temporal embeddings
        av_tok = av_tok + m_av_emb + t_emb   # (B, T, d_model)
        wr_tok = wr_tok + m_wr_emb + t_emb
        pr_tok = pr_tok + m_pr_emb + t_emb

        # --- Interleave: [av_t0, wr_t0, pr_t0, av_t1, wr_t1, pr_t1, ...] ---
        # Stack modalities: (B, T, 3, d_model) then reshape to (B, T*3, d_model)
        tokens = torch.stack([av_tok, wr_tok, pr_tok], dim=2)  # (B, T, 3, d)
        tokens = tokens.reshape(B, T * self.N_MODALITIES, self.d_model)

        return tokens  # (B, 3*T_obs, d_model)

    def forward(
        self,
        agentview: torch.Tensor,
        wrist: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        """Single obs vector (mean-pool over all tokens).

        Returns: (B, d_model)
        """
        return self.forward_tokens(agentview, wrist, proprio).mean(dim=1)
