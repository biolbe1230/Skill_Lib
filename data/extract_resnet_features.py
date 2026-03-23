"""
Pre-extract frozen ResNet-18 features for all LIBERO-90 HDF5 files.

Saves per-frame 512-d features for agentview and wrist cameras
to HDF5 files in the output directory, matching the original file structure:

    output_dir/
      <original_filename>.hdf5
        data/
          demo_0/
            agentview_feat  (T, 512) float16
            wrist_feat      (T, 512) float16
          demo_1/
            ...

Usage:
    python -m data.extract_resnet_features \
        --demo_dir /data/datasets/liyuxuan/datasets/libero_90/ \
        --output_dir /data/datasets/liyuxuan/datasets/libero_90_feat \
        --batch_size 256 \
        --device cuda
"""

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

logger = logging.getLogger(__name__)


def build_resnet_backbone(device: str = "cuda") -> nn.Module:
    """Build frozen ResNet-18 feature extractor (same as ObsEncoder)."""
    resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        resnet.avgpool,
        nn.Flatten(1),  # -> (B, 512)
    )
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone.to(device)


@torch.inference_mode()
def extract_features_for_file(
    backbone: nn.Module,
    src_path: str,
    dst_path: str,
    batch_size: int = 256,
    device: str = "cuda",
    save_fp16: bool = True,
):
    """Extract ResNet features for all demos in one HDF5 file."""
    src = h5py.File(src_path, "r")
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    dst = h5py.File(dst_path, "w")

    demo_keys = sorted(src["data"].keys())
    dst.create_group("data")

    for dk in demo_keys:
        obs = src["data"][dk]["obs"]
        agentview_imgs = obs["agentview_rgb"]     # (T, 128, 128, 3) uint8
        wrist_imgs = obs["eye_in_hand_rgb"]       # (T, 128, 128, 3) uint8
        T = agentview_imgs.shape[0]

        av_feats_list = []
        wr_feats_list = []

        # Process in batches
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)

            # Load and preprocess: HWC uint8 -> CHW float32 [0,1]
            av_batch = agentview_imgs[start:end]  # (B, H, W, 3)
            wr_batch = wrist_imgs[start:end]

            av_t = torch.from_numpy(
                av_batch.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            ).to(device)
            wr_t = torch.from_numpy(
                wr_batch.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            ).to(device)

            av_feat = backbone(av_t).cpu().numpy()  # (B, 512)
            wr_feat = backbone(wr_t).cpu().numpy()  # (B, 512)

            av_feats_list.append(av_feat)
            wr_feats_list.append(wr_feat)

        av_feats = np.concatenate(av_feats_list, axis=0)  # (T, 512)
        wr_feats = np.concatenate(wr_feats_list, axis=0)  # (T, 512)

        dtype = np.float16 if save_fp16 else np.float32
        grp = dst.create_group(f"data/{dk}")
        grp.create_dataset("agentview_feat", data=av_feats.astype(dtype))
        grp.create_dataset("wrist_feat", data=wr_feats.astype(dtype))

    src.close()
    dst.close()


def main():
    parser = argparse.ArgumentParser(
        description="Pre-extract ResNet-18 features for LIBERO-90"
    )
    parser.add_argument(
        "--demo_dir",
        default="/data/datasets/liyuxuan/datasets/libero_90/",
        help="Source directory with LIBERO HDF5 files",
    )
    parser.add_argument(
        "--output_dir",
        default="/data/datasets/liyuxuan/datasets/libero_90_feat",
        help="Output directory for cached features",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--fp32", action="store_true",
        help="Save as float32 instead of float16 (double disk usage)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Find all HDF5 files
    src_files = sorted(glob.glob(os.path.join(args.demo_dir, "*.hdf5")))
    if not src_files:
        logger.error("No HDF5 files found in %s", args.demo_dir)
        sys.exit(1)
    logger.info("Found %d HDF5 files in %s", len(src_files), args.demo_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    # Build backbone
    backbone = build_resnet_backbone(args.device)
    logger.info("ResNet-18 backbone loaded on %s", args.device)

    # Extract
    for i, src_path in enumerate(src_files):
        fname = os.path.basename(src_path)
        dst_path = os.path.join(args.output_dir, fname)

        if os.path.exists(dst_path):
            logger.info("[%d/%d] SKIP (exists): %s", i + 1, len(src_files), fname)
            continue

        logger.info("[%d/%d] Extracting: %s", i + 1, len(src_files), fname)
        extract_features_for_file(
            backbone, src_path, dst_path,
            batch_size=args.batch_size,
            device=args.device,
            save_fp16=not args.fp32,
        )

    logger.info("Done! Features saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
