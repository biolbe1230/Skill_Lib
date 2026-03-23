"""
PyTorch Dataset for skill-segmented LIBERO trajectories.

Loads segments filtered by L1 VerbNet class. Each sample provides:
  - agentview image(s)   (T_obs, 3, 128, 128)
  - wrist image(s)       (T_obs, 3, 128, 128)
  - proprio state        (T_obs, proprio_dim)
  - action chunk         (T_pred, 7)

Usage:
    dataset = SkillSegmentDataset(
        segmented_json="segmented_demos.json",
        demo_dir="/data/datasets/liyuxuan/datasets/libero_90/",
        l1_class="put-9.1-2",
        T_obs=2, T_pred=16,
    )
"""

import json
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class SkillSegmentDataset(Dataset):
    """Dataset of action-chunked samples filtered by L1 VerbNet class."""

    # proprio = ee_pos(3) + ee_ori(3) + gripper_states(2) = 8
    PROPRIO_DIM = 8

    def __init__(
        self,
        segmented_json: str,
        demo_dir: str,
        l1_class: str | None = None,
        T_obs: int = 2,
        T_pred: int = 16,
        img_size: int = 128,
        action_stats: dict | None = None,
        feat_dir: str | None = None,
    ):
        """
        Args:
            segmented_json: Path to segmented_demos.json from segment_trajectories.
            demo_dir: Root dir of LIBERO HDF5 files.
            l1_class: Filter by L1 VerbNet class. None = use all classes.
            T_obs: Number of observation history frames.
            T_pred: Action chunk prediction horizon.
            img_size: Expected image size (for validation).
            action_stats: Optional dict {"min": (7,), "max": (7,)} for normalization.
            feat_dir: Directory with pre-extracted ResNet features. If set,
                      returns 512-d features instead of raw images.
        """
        self.demo_dir = demo_dir
        self.feat_dir = feat_dir
        self.use_cached_feat = feat_dir is not None
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.img_size = img_size

        with open(segmented_json) as f:
            all_demos = json.load(f)

        # Build flat list of (hdf5_file, demo_key, seg_start, seg_end, l1_class)
        self.samples: list[dict] = []
        for item in all_demos:
            for seg in item["segments"]:
                if l1_class is not None and seg["l1_class"] != l1_class:
                    continue
                seg_len = seg["end"] - seg["start"] + 1
                if seg_len < T_obs + T_pred:
                    continue  # segment too short

                self.samples.append({
                    "hdf5_file": item["hdf5_file"],
                    "demo_key": item["demo_key"],
                    "seg_start": seg["start"],
                    "seg_end": seg["end"],
                    "l1_class": seg["l1_class"],
                    "subtask": seg["subtask"],
                })

        # Pre-compute sliding windows: each sample is a (file, demo, frame_idx)
        self.windows: list[tuple[int, int]] = []  # (sample_idx, offset_in_seg)
        for si, s in enumerate(self.samples):
            seg_len = s["seg_end"] - s["seg_start"] + 1
            max_start = seg_len - T_pred  # earliest obs frame needs T_obs-1 history
            for offset in range(T_obs - 1, max_start + 1):
                self.windows.append((si, offset))

        # Action normalization stats
        self._action_min = None
        self._action_max = None
        if action_stats is not None:
            self._action_min = np.array(action_stats["min"], dtype=np.float32)
            self._action_max = np.array(action_stats["max"], dtype=np.float32)

        # Cache for open HDF5 handles (closed in __del__)
        self._hdf5_cache: dict[str, h5py.File] = {}
        self._feat_cache: dict[str, h5py.File] = {}

    def set_action_stats(self, action_min: np.ndarray, action_max: np.ndarray):
        """Set action normalization statistics (computed externally)."""
        self._action_min = action_min.astype(np.float32)
        self._action_max = action_max.astype(np.float32)

    def _get_hdf5(self, fname: str) -> h5py.File:
        if fname not in self._hdf5_cache:
            self._hdf5_cache[fname] = h5py.File(
                os.path.join(self.demo_dir, fname), "r"
            )
        return self._hdf5_cache[fname]

    def _get_feat_hdf5(self, fname: str) -> h5py.File:
        if fname not in self._feat_cache:
            self._feat_cache[fname] = h5py.File(
                os.path.join(self.feat_dir, fname), "r"
            )
        return self._feat_cache[fname]

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        if self._action_min is None:
            return action.astype(np.float32)
        # Min-max normalize to [-1, 1]
        range_ = np.maximum(self._action_max - self._action_min, 1e-6)
        return ((action - self._action_min) / range_ * 2 - 1).astype(np.float32)

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        if self._action_min is None:
            return action
        range_ = np.maximum(self._action_max - self._action_min, 1e-6)
        return ((action + 1) / 2 * range_ + self._action_min).astype(np.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample_idx, offset = self.windows[idx]
        s = self.samples[sample_idx]

        hf = self._get_hdf5(s["hdf5_file"])
        demo = hf["data"][s["demo_key"]]
        obs = demo["obs"]

        # Absolute frame indices
        abs_start = s["seg_start"] + offset - (self.T_obs - 1)
        obs_frames = list(range(abs_start, abs_start + self.T_obs))
        action_start = s["seg_start"] + offset
        action_end = action_start + self.T_pred

        # Read proprio: ee_pos(3) + ee_ori(3) + gripper(2) = 8
        ee_pos = np.stack([obs["ee_pos"][t] for t in obs_frames])
        ee_ori = np.stack([obs["ee_ori"][t] for t in obs_frames])
        gripper = np.stack([obs["gripper_states"][t] for t in obs_frames])
        proprio = np.concatenate([ee_pos, ee_ori, gripper], axis=-1)  # (T_obs, 8)

        # Read action chunk: (T_pred, 7)
        actions = demo["actions"][action_start:action_end]

        # Action normalization
        actions = self._normalize_action(actions)
        proprio = proprio.astype(np.float32)

        if self.use_cached_feat:
            # Load pre-extracted ResNet features (512-d)
            feat_hf = self._get_feat_hdf5(s["hdf5_file"])
            feat_demo = feat_hf["data"][s["demo_key"]]
            av_feat = np.stack([feat_demo["agentview_feat"][t] for t in obs_frames]).astype(np.float32)
            wr_feat = np.stack([feat_demo["wrist_feat"][t] for t in obs_frames]).astype(np.float32)
            return {
                "agentview_feat": torch.from_numpy(av_feat),   # (T_obs, 512)
                "wrist_feat": torch.from_numpy(wr_feat),       # (T_obs, 512)
                "proprio": torch.from_numpy(proprio),          # (T_obs, 8)
                "actions": torch.from_numpy(actions),          # (T_pred, 7)
            }
        else:
            # Load raw images
            agentview = np.stack([obs["agentview_rgb"][t] for t in obs_frames])
            wrist = np.stack([obs["eye_in_hand_rgb"][t] for t in obs_frames])
            agentview = agentview.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            wrist = wrist.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            return {
                "agentview": torch.from_numpy(agentview),       # (T_obs, 3, 128, 128)
                "wrist": torch.from_numpy(wrist),                # (T_obs, 3, 128, 128)
                "proprio": torch.from_numpy(proprio),            # (T_obs, 8)
                "actions": torch.from_numpy(actions),            # (T_pred, 7)
            }

    def close(self):
        for hf in self._hdf5_cache.values():
            hf.close()
        self._hdf5_cache.clear()
        for hf in self._feat_cache.values():
            hf.close()
        self._feat_cache.clear()

    def __del__(self):
        self.close()


def compute_action_stats(
    segmented_json: str,
    demo_dir: str,
    l1_class: str | None = None,
) -> dict[str, np.ndarray]:
    """Compute per-dim min/max action statistics over all segments.

    Returns:
        {"min": np.ndarray (7,), "max": np.ndarray (7,)}
    """
    with open(segmented_json) as f:
        all_demos = json.load(f)

    global_min = np.full(7, np.inf, dtype=np.float64)
    global_max = np.full(7, -np.inf, dtype=np.float64)

    seen_files: dict[str, h5py.File] = {}

    for item in all_demos:
        fname = item["hdf5_file"]
        for seg in item["segments"]:
            if l1_class is not None and seg["l1_class"] != l1_class:
                continue
            if fname not in seen_files:
                seen_files[fname] = h5py.File(os.path.join(demo_dir, fname), "r")
            hf = seen_files[fname]
            actions = hf["data"][item["demo_key"]]["actions"][seg["start"]:seg["end"] + 1]
            global_min = np.minimum(global_min, actions.min(axis=0))
            global_max = np.maximum(global_max, actions.max(axis=0))

    for hf in seen_files.values():
        hf.close()

    return {
        "min": global_min.astype(np.float32),
        "max": global_max.astype(np.float32),
    }
