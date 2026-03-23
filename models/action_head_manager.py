"""
ActionHeadManager: dispatches observations to the correct per-L1-class action head.

At inference time, the manager:
  1. Receives l1_class from VerbNet parsing of the current subtask.
  2. Looks up the corresponding action head (DiffusionHead or ACTHead).
  3. Encodes obs via the shared ObsEncoder.
  4. Calls head.predict(obs_tokens) → action_chunk.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn

from .obs_encoder import ObsEncoder
from .diffusion_head import DiffusionHead, EMAModel
from .act_head import ACTHead


class ActionHeadManager(nn.Module):
    """Manages per-L1-class action heads with shared obs encoder."""

    def __init__(
        self,
        head_type: str = "diffusion",
        l1_classes: list[str] | None = None,
        d_model: int = 512,
        T_obs: int = 2,
        T_pred: int = 16,
        action_dim: int = 7,
        proprio_dim: int = 8,
        **head_kwargs,
    ):
        super().__init__()
        self.head_type = head_type
        self.d_model = d_model
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.action_dim = action_dim

        if l1_classes is None:
            l1_classes = [
                "get-13.5.1", "hold-15.1-1", "other_cos-45.4",
                "push-12-1", "put-9.1-2", "roll-51.3.1", "turn-26.6.1-1",
            ]
        self.l1_classes = l1_classes

        # Shared observation encoder (ResNet frozen + trainable projections)
        self.obs_encoder = ObsEncoder(
            proprio_dim=proprio_dim,
            T_obs=T_obs,
        )

        # Per-L1-class action heads
        self.heads = nn.ModuleDict()
        for cls_name in l1_classes:
            # ModuleDict keys cannot contain "." so replace
            key = cls_name.replace(".", "_")
            if head_type == "diffusion":
                self.heads[key] = DiffusionHead(
                    action_dim=action_dim,
                    T_pred=T_pred,
                    d_model=d_model,
                    **head_kwargs,
                )
            elif head_type == "act":
                self.heads[key] = ACTHead(
                    action_dim=action_dim,
                    T_pred=T_pred,
                    d_model=d_model,
                    **head_kwargs,
                )
            else:
                raise ValueError(f"Unknown head_type: {head_type}")

    @staticmethod
    def _l1_to_key(l1_class: str) -> str:
        return l1_class.replace(".", "_")

    def get_head(self, l1_class: str) -> nn.Module:
        key = self._l1_to_key(l1_class)
        if key not in self.heads:
            raise KeyError(f"No action head for L1 class '{l1_class}' (key='{key}')")
        return self.heads[key]

    def encode_obs(
        self,
        agentview: torch.Tensor,
        wrist: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        """Encode observations to tokens for cross-attention.

        Returns: (B, T_obs, d_model)
        """
        return self.obs_encoder.forward_tokens(agentview, wrist, proprio)

    def predict(
        self,
        agentview: torch.Tensor,   # (B, T_obs, 3, H, W)
        wrist: torch.Tensor,       # (B, T_obs, 3, H, W)
        proprio: torch.Tensor,     # (B, T_obs, proprio_dim)
        l1_class: str,
    ) -> torch.Tensor:
        """Predict action chunk for given L1 class.

        Returns: (B, T_pred, action_dim)
        """
        obs_tokens = self.encode_obs(agentview, wrist, proprio)
        head = self.get_head(l1_class)
        return head.predict(obs_tokens)

    def compute_loss(
        self,
        agentview: torch.Tensor,
        wrist: torch.Tensor,
        proprio: torch.Tensor,
        actions: torch.Tensor,
        l1_class: str,
        **kwargs,
    ):
        """Compute loss for a specific L1 class head.

        Returns: loss tensor (diffusion) or dict with loss/recon/kl (ACT)
        """
        obs_tokens = self.encode_obs(agentview, wrist, proprio)
        head = self.get_head(l1_class)

        if isinstance(head, DiffusionHead):
            return head.compute_loss(actions, obs_tokens)
        elif isinstance(head, ACTHead):
            return head.compute_loss(actions, obs_tokens, **kwargs)
        else:
            raise TypeError(f"Unknown head type: {type(head)}")

    def save_head(self, l1_class: str, save_dir: str, ema: EMAModel | None = None):
        """Save a specific head + obs encoder checkpoint."""
        os.makedirs(save_dir, exist_ok=True)
        head = self.get_head(l1_class)
        ckpt = {
            "obs_encoder": self.obs_encoder.state_dict(),
            "head": head.state_dict(),
            "l1_class": l1_class,
            "head_type": self.head_type,
        }
        if ema is not None:
            ckpt["ema_head"] = ema.state_dict()
        torch.save(ckpt, os.path.join(save_dir, f"{self._l1_to_key(l1_class)}_best.pt"))

    def load_head(self, l1_class: str, ckpt_path: str, use_ema: bool = True):
        """Load a specific head checkpoint."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.obs_encoder.load_state_dict(ckpt["obs_encoder"])
        head = self.get_head(l1_class)
        if use_ema and "ema_head" in ckpt:
            head.load_state_dict(ckpt["ema_head"])
        else:
            head.load_state_dict(ckpt["head"])

    def load_all(self, ckpt_dir: str, use_ema: bool = True):
        """Load all head checkpoints from a directory.

        Supports two layouts:
          1. Training layout: {ckpt_dir}/{key}/{head_type}/best.pt
          2. Flat layout:     {ckpt_dir}/{key}_best.pt
        """
        for cls_name in self.l1_classes:
            key = self._l1_to_key(cls_name)
            # Try training layout first
            nested = os.path.join(ckpt_dir, key, self.head_type, "best.pt")
            flat = os.path.join(ckpt_dir, f"{key}_best.pt")
            if os.path.exists(nested):
                self.load_head(cls_name, nested, use_ema=use_ema)
            elif os.path.exists(flat):
                self.load_head(cls_name, flat, use_ema=use_ema)
            else:
                print(f"Warning: checkpoint not found for {cls_name}"
                      f" (tried {nested} and {flat})")
