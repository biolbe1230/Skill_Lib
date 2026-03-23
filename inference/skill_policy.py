"""
End-to-end inference loop for skill-conditioned action execution.

Pipeline:
  1. Look up subtask list from subtasks_by_task.json
  2. Initialize Qwen3-VL checker for online subtask completion detection
  3. Load ActionHeadManager with per-L1-class action heads
  4. For each subtask:
     a. Parse L1 class via VerbNet
     b. Select action head
     c. Predict & execute action chunks
     d. Periodically check subtask completion via Qwen3-VL
     e. On completion, advance to next subtask
  5. Task done when all subtasks completed or max steps reached.

Usage:
    # Standalone demo (requires LIBERO env):
    python -m inference.skill_policy \
        --task "close the top drawer of the cabinet" \
        --head_type diffusion \
        --ckpt_dir checkpoints
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verbnet_utils import _parse_subtask_verbnet
from QwenPlanner import QwenPlanner
from models.action_head_manager import ActionHeadManager

logger = logging.getLogger(__name__)


class SkillPolicy:
    """High-level policy: subtask sequencing + per-skill action generation."""

    def __init__(
        self,
        subtasks_json: str,
        ckpt_dir: str,
        head_type: str = "diffusion",
        model_path: str = "/export/ra/liyuxuan/VLA/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct",
        device: str = "cuda",
        check_interval: int = 8,
        T_obs: int = 2,
        T_pred: int = 16,
        d_model: int = 512,
        use_ema: bool = True,
    ):
        self.device = torch.device(device)
        self.check_interval = check_interval
        self.T_obs = T_obs
        self.T_pred = T_pred

        # Load subtask lookup
        with open(subtasks_json) as f:
            tasks = json.load(f)
        self.task_to_subtasks = {t["task_language"]: t["subtasks"] for t in tasks}

        # Load action head manager
        self.manager = ActionHeadManager(
            head_type=head_type,
            d_model=d_model,
            T_obs=T_obs,
            T_pred=T_pred,
        ).to(self.device)
        self.manager.load_all(ckpt_dir, use_ema=use_ema)
        self.manager.eval()
        logger.info("Loaded action heads from %s", ckpt_dir)

        # Load action stats per L1 class for denormalization
        self.action_stats: dict[str, dict] = {}
        for cls_name in self.manager.l1_classes:
            key = cls_name.replace(".", "_")
            stats_path = os.path.join(ckpt_dir, key, head_type, "action_stats.json")
            if os.path.exists(stats_path):
                with open(stats_path) as f:
                    self.action_stats[cls_name] = json.load(f)

        # Initialize Qwen planner for subtask checking
        self.planner = QwenPlanner(model_path=model_path)
        logger.info("Qwen planner loaded")

    def _denormalize_action(self, action: np.ndarray, l1_class: str) -> np.ndarray:
        """Denormalize action from [-1,1] to original scale."""
        if l1_class not in self.action_stats:
            return action
        stats = self.action_stats[l1_class]
        a_min = np.array(stats["min"], dtype=np.float32)
        a_max = np.array(stats["max"], dtype=np.float32)
        range_ = np.maximum(a_max - a_min, 1e-6)
        return ((action + 1) / 2 * range_ + a_min).astype(np.float32)

    def _preprocess_obs(
        self,
        agentview_history: list[np.ndarray],   # list of (H,W,3) uint8
        wrist_history: list[np.ndarray],        # list of (H,W,3) uint8
        proprio_history: list[np.ndarray],      # list of (proprio_dim,)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert raw obs to model input tensors."""
        # Stack and convert: (T_obs, H, W, 3) -> (1, T_obs, 3, H, W)
        av = np.stack(agentview_history[-self.T_obs:])
        wr = np.stack(wrist_history[-self.T_obs:])
        pr = np.stack(proprio_history[-self.T_obs:])

        av = torch.from_numpy(av.transpose(0, 3, 1, 2).astype(np.float32) / 255.0)
        wr = torch.from_numpy(wr.transpose(0, 3, 1, 2).astype(np.float32) / 255.0)
        pr = torch.from_numpy(pr.astype(np.float32))

        return (
            av.unsqueeze(0).to(self.device),
            wr.unsqueeze(0).to(self.device),
            pr.unsqueeze(0).to(self.device),
        )

    @torch.inference_mode()
    def run(
        self,
        task_language: str,
        env,
        max_steps: int = 600,
    ) -> dict:
        """Execute full task in environment.

        Args:
            task_language: High-level task description.
            env: LIBERO environment with step() and get_obs() methods.

        Returns:
            dict with keys: success, steps, subtasks_completed
        """
        if task_language not in self.task_to_subtasks:
            raise ValueError(f"Unknown task: {task_language}")

        subtasks = self.task_to_subtasks[task_language]
        logger.info("Task: %s | Subtasks: %s", task_language, subtasks)

        current_idx = 0
        finished_subtasks: list[str] = []
        total_steps = 0

        # Observation history buffers
        av_history: list[np.ndarray] = []
        wr_history: list[np.ndarray] = []
        pr_history: list[np.ndarray] = []

        # Get initial observation
        obs = env.get_obs()
        av_history.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        wr_history.append(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]))
        pr_history.append(np.concatenate([
            obs["robot0_eef_pos"], obs["robot0_eef_quat"][:3], obs["robot0_gripper_qpos"],
        ]).astype(np.float32))

        # Pad history if needed
        while len(av_history) < self.T_obs:
            av_history.insert(0, av_history[0].copy())
            wr_history.insert(0, wr_history[0].copy())
            pr_history.insert(0, pr_history[0].copy())

        steps_since_check = 0

        while current_idx < len(subtasks) and total_steps < max_steps:
            subtask = subtasks[current_idx]
            parsed = _parse_subtask_verbnet(subtask)
            l1_class = parsed["verbnet_class"]
            logger.info("Step %d | Subtask[%d]: %s (L1: %s)",
                        total_steps, current_idx, subtask, l1_class)

            # Predict action chunk
            av_t, wr_t, pr_t = self._preprocess_obs(av_history, wr_history, pr_history)
            action_chunk = self.manager.predict(av_t, wr_t, pr_t, l1_class)
            action_chunk = action_chunk[0].cpu().numpy()  # (T_pred, 7)

            # Denormalize
            action_chunk = self._denormalize_action(action_chunk, l1_class)

            # Execute action chunk
            for ai, act in enumerate(action_chunk):
                obs, reward, done, info = env.step(act)
                total_steps += 1
                steps_since_check += 1

                # Update history
                av_history.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                wr_history.append(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]))
                pr_history.append(np.concatenate([
                    obs["robot0_eef_pos"], obs["robot0_eef_quat"][:3], obs["robot0_gripper_qpos"],
                ]).astype(np.float32))

                # Keep history bounded
                if len(av_history) > self.T_obs * 2:
                    av_history = av_history[-self.T_obs * 2:]
                    wr_history = wr_history[-self.T_obs * 2:]
                    pr_history = pr_history[-self.T_obs * 2:]

                if done:
                    break

                # Check subtask completion periodically
                if steps_since_check >= self.check_interval:
                    completed = self.planner.check_subtask(
                        high_task=task_language,
                        image_list=[np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
                                    np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])],
                        current_subtask=subtask,
                        all_subtasks=subtasks,
                        finished_subtasks=finished_subtasks,
                    )
                    steps_since_check = 0

                    if completed:
                        logger.info("  Subtask '%s' COMPLETED at step %d", subtask, total_steps)
                        finished_subtasks.append(subtask)
                        current_idx += 1
                        break

                if total_steps >= max_steps:
                    break

            if done:
                break

        success = current_idx >= len(subtasks)
        logger.info("Task %s | steps=%d | subtasks_completed=%d/%d | success=%s",
                    "SUCCESS" if success else "FAIL",
                    total_steps, current_idx, len(subtasks), success)

        return {
            "success": success,
            "steps": total_steps,
            "subtasks_completed": current_idx,
            "total_subtasks": len(subtasks),
        }


def main():
    parser = argparse.ArgumentParser(description="Skill policy inference")
    parser.add_argument("--task", required=True, help="Task language description")
    parser.add_argument("--head_type", choices=["diffusion", "act"], default="diffusion")
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--subtasks_json",
                        default="skill_lib_results_clip_film_raw/subtasks_by_task.json")
    parser.add_argument("--model_path",
                        default="/export/ra/liyuxuan/VLA/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    policy = SkillPolicy(
        subtasks_json=args.subtasks_json,
        ckpt_dir=args.ckpt_dir,
        head_type=args.head_type,
        model_path=args.model_path,
        device=args.device,
    )

    # This requires a LIBERO environment to be available.
    # Example integration:
    #   import libero; env = libero.make(args.task)
    #   result = policy.run(args.task, env, max_steps=args.max_steps)
    logger.info("SkillPolicy initialized. Provide a LIBERO env to run inference.")
    logger.info("Example: result = policy.run('%s', env)", args.task)


if __name__ == "__main__":
    main()
