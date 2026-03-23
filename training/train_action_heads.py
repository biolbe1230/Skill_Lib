"""
Unified training script for per-L1-class action heads.

Trains one L1 class at a time (or loops over all). Supports both
Diffusion Policy and ACT head types.

Usage:
    # Train one class
    python -m training.train_action_heads \
        --l1_class put-9.1-2 \
        --head_type diffusion \
        --segmented_json segmented_demos.json

    # Train all classes sequentially
    python -m training.train_action_heads \
        --l1_class all \
        --head_type act
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.skill_dataset import SkillSegmentDataset, compute_action_stats
from models.obs_encoder import ObsEncoder
from models.diffusion_head import DiffusionHead, EMAModel
from models.act_head import ACTHead, kl_warmup_beta

logger = logging.getLogger(__name__)

ALL_L1_CLASSES = [
    "get-13.5.1", "hold-15.1-1", "other_cos-45.4",
    "push-12-1", "push-12-1-1", "put-9.1-1", "put-9.1-2",
    "put_direction-9.4", "roll-51.3.1", "turn-26.6.1-1",
]


# Minimum number of total windows required to train a class.
# Classes below this threshold are skipped (use a fallback head at inference).
MIN_TRAIN_WINDOWS = 256


def build_dataloader(args, l1_class: str):
    """Build train/val dataloaders for a specific L1 class.

    Returns (train_loader, val_loader, action_stats) or None if the class
    has fewer than MIN_TRAIN_WINDOWS samples.
    """
    # Compute action normalization stats
    action_stats = compute_action_stats(
        args.segmented_json, args.demo_dir, l1_class=l1_class,
    )
    logger.info("Action stats for %s: min=%s, max=%s",
                l1_class, action_stats["min"], action_stats["max"])

    dataset = SkillSegmentDataset(
        segmented_json=args.segmented_json,
        demo_dir=args.demo_dir,
        l1_class=l1_class,
        T_obs=args.T_obs,
        T_pred=args.T_pred,
        action_stats=action_stats,
        feat_dir=getattr(args, 'feat_dir', None),
    )

    n_total = len(dataset)
    if n_total < MIN_TRAIN_WINDOWS:
        logger.warning(
            "L1=%s has only %d windows (< %d), skipping training. "
            "This class will fall back to a generic head at inference.",
            l1_class, n_total, MIN_TRAIN_WINDOWS,
        )
        return None

    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val

    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    logger.info("L1=%s: %d total windows, %d train, %d val",
                l1_class, n_total, n_train, n_val)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    return train_loader, val_loader, action_stats


def train_one_class(args, l1_class: str):
    """Train action head for a single L1 class."""
    device = torch.device(args.device)
    logger.info("=" * 60)
    logger.info("Training %s head for L1 class: %s", args.head_type, l1_class)
    logger.info("=" * 60)

    # Data
    result = build_dataloader(args, l1_class)
    if result is None:
        return
    train_loader, val_loader, action_stats = result

    use_cached_feat = getattr(args, 'feat_dir', None) is not None
    if use_cached_feat:
        logger.info("Using cached ResNet features from %s", args.feat_dir)

    # Model
    obs_encoder = ObsEncoder(
        proprio_dim=8,
        T_obs=args.T_obs,
    ).to(device)

    if args.head_type == "diffusion":
        head = DiffusionHead(
            action_dim=7,
            T_pred=args.T_pred,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            num_train_steps=args.diffusion_steps,
            num_infer_steps=args.diffusion_infer_steps,
        ).to(device)
    elif args.head_type == "act":
        head = ACTHead(
            action_dim=7,
            T_pred=args.T_pred,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_layers,
            num_decoder_layers=args.num_layers,
            latent_dim=args.latent_dim,
        ).to(device)
    else:
        raise ValueError(f"Unknown head type: {args.head_type}")

    # Only train: obs_encoder trainable parts (projections + embeddings) + head
    # ResNet is frozen; trainable = agentview_proj, wrist_proj, proprio_proj,
    # modality_emb, temporal_emb, plus the entire action head.
    trainable_params = [
        p for p in obs_encoder.parameters() if p.requires_grad
    ] + list(head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader),
    )

    # Mixed precision scaler
    use_amp = args.amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # torch.compile (PyTorch 2.x): speeds up repeated graph patterns
    if args.compile:
        logger.info("torch.compile enabled, compiling obs_encoder + head ...")
        obs_encoder = torch.compile(obs_encoder)
        head = torch.compile(head)

    # EMA for diffusion
    ema = None
    if args.head_type == "diffusion":
        ema = EMAModel(head, decay=0.999)

    # Checkpoint dir
    ckpt_dir = os.path.join(args.ckpt_dir, l1_class.replace(".", "_"), args.head_type)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save action stats
    stats_path = os.path.join(ckpt_dir, "action_stats.json")
    with open(stats_path, "w") as f:
        json.dump({k: v.tolist() for k, v in action_stats.items()}, f)

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        obs_encoder.train()
        head.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            proprio = batch["proprio"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                # Encode observations
                if use_cached_feat:
                    av_feat = batch["agentview_feat"].to(device, non_blocking=True)
                    wr_feat = batch["wrist_feat"].to(device, non_blocking=True)
                    obs_tokens = obs_encoder.forward_tokens_from_feat(av_feat, wr_feat, proprio)
                else:
                    agentview = batch["agentview"].to(device, non_blocking=True)
                    wrist = batch["wrist"].to(device, non_blocking=True)
                    obs_tokens = obs_encoder.forward_tokens(agentview, wrist, proprio)

                # Compute loss
                if args.head_type == "diffusion":
                    loss = head.compute_loss(actions, obs_tokens)
                else:
                    beta = kl_warmup_beta(global_step, args.kl_warmup_steps, args.kl_max_beta)
                    loss_dict = head.compute_loss(actions, obs_tokens, beta=beta)
                    loss = loss_dict["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if ema is not None:
                ema.update(head)

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        obs_encoder.eval()
        head.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                proprio = batch["proprio"].to(device, non_blocking=True)
                actions = batch["actions"].to(device, non_blocking=True)

                with autocast(enabled=use_amp):
                    if use_cached_feat:
                        av_feat = batch["agentview_feat"].to(device, non_blocking=True)
                        wr_feat = batch["wrist_feat"].to(device, non_blocking=True)
                        obs_tokens = obs_encoder.forward_tokens_from_feat(av_feat, wr_feat, proprio)
                    else:
                        agentview = batch["agentview"].to(device, non_blocking=True)
                        wrist = batch["wrist"].to(device, non_blocking=True)
                        obs_tokens = obs_encoder.forward_tokens(agentview, wrist, proprio)

                    if args.head_type == "diffusion":
                        loss = head.compute_loss(actions, obs_tokens)
                    else:
                        loss_dict = head.compute_loss(actions, obs_tokens, beta=1.0)
                        loss = loss_dict["loss"]

                val_loss += loss.item()
                n_val += 1

        avg_val_loss = val_loss / max(n_val, 1)

        logger.info(
            "Epoch %d/%d | train_loss=%.6f | val_loss=%.6f | lr=%.2e | step=%d",
            epoch + 1, args.epochs, avg_train_loss, avg_val_loss,
            scheduler.get_last_lr()[0], global_step,
        )

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt = {
                "obs_encoder": obs_encoder.state_dict(),
                "head": head.state_dict(),
                "l1_class": l1_class,
                "head_type": args.head_type,
                "epoch": epoch,
                "val_loss": best_val_loss,
                "action_stats": {k: v.tolist() for k, v in action_stats.items()},
            }
            if ema is not None:
                ckpt["ema_head"] = ema.state_dict()
            torch.save(ckpt, os.path.join(ckpt_dir, "best.pt"))
            logger.info("  -> Saved best checkpoint (val_loss=%.6f)", best_val_loss)

    logger.info("Finished training %s for %s. Best val_loss=%.6f",
                args.head_type, l1_class, best_val_loss)


def main():
    parser = argparse.ArgumentParser(description="Train per-L1-class action heads")

    # Data
    parser.add_argument("--segmented_json", default="segmented_demos.json")
    parser.add_argument("--demo_dir", default="/data/datasets/liyuxuan/datasets/libero_90/")
    parser.add_argument("--feat_dir", default=None,
                        help="Pre-extracted ResNet feature dir (skip ResNet at train time)")
    parser.add_argument("--val_ratio", type=float, default=0.1)

    # Model
    parser.add_argument("--head_type", choices=["diffusion", "act"], default="diffusion")
    parser.add_argument("--l1_class", default="all",
                        help="L1 class to train, or 'all' for all classes")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--T_obs", type=int, default=2)
    parser.add_argument("--T_pred", type=int, default=16)
    parser.add_argument("--latent_dim", type=int, default=32, help="ACT latent dim")

    # Diffusion specific
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--diffusion_infer_steps", type=int, default=10)

    # ACT specific
    parser.add_argument("--kl_warmup_steps", type=int, default=1000)
    parser.add_argument("--kl_max_beta", type=float, default=10.0)

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ckpt_dir", default="checkpoints")

    # Efficiency
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use automatic mixed precision (AMP, default: on)")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable AMP")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="Use torch.compile (PyTorch 2.x, ~20%% faster after warmup)")
    parser.add_argument("--parallel", action="store_true", default=False,
                        help="Train all L1 classes in parallel subprocesses (one per class)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.no_amp:
        args.amp = False

    if args.l1_class == "all":
        classes = ALL_L1_CLASSES
        if args.parallel:
            # Spawn one process per L1 class (each uses same GPU unless overridden)
            logger.info("Parallel mode: spawning %d subprocesses", len(classes))
            procs = []
            for cls in classes:
                p = mp.Process(target=train_one_class, args=(args, cls))
                p.start()
                procs.append(p)
            for p in procs:
                p.join()
        else:
            for cls in classes:
                train_one_class(args, cls)
    else:
        train_one_class(args, args.l1_class)


if __name__ == "__main__":
    main()
