"""
Evaluation script for per-L1-class action heads on LIBERO-90.

For each task (or a subset):
  1. Create LIBERO OffScreenRenderEnv, set init state.
  2. Call QwenPlanner.get_subtasks() on the initial observation to get subtask list.
  3. For each subtask: parse L1 class via VerbNet → select action head → predict
     action chunk → execute → periodically call Qwen check_subtask() → advance.
  4. Record success (env returns done=True) and per-subtask completion.
  5. Report per-task and aggregate success rates.

Usage:
    python -m evaluation.evaluate \
        --head_type diffusion \
        --ckpt_dir checkpoints \
        --n_eval 20 \
        --max_steps 600
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from QwenPlanner import QwenPlanner
from models.obs_encoder import ObsEncoder
from models.action_head_manager import ActionHeadManager
from models.diffusion_head import DiffusionHead, EMAModel
from models.act_head import ACTHead
from verbnet_utils import _parse_subtask_verbnet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def build_env(bddl_file: str, img_size: int = 128) -> OffScreenRenderEnv:
    """Create a LIBERO OffScreenRenderEnv."""
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=img_size,
        camera_widths=img_size,
    )
    return env


def extract_obs(obs: dict) -> dict:
    """Extract and normalise observation from LIBERO env dict.

    LIBERO env keys differ from HDF5 keys:
      env: agentview_image, robot0_eye_in_hand_image, robot0_eef_pos,
           robot0_eef_quat, robot0_gripper_qpos
      HDF5: agentview_rgb, eye_in_hand_rgb, ee_pos, ee_ori, gripper_states

    Images are rotated 180° (LIBERO eval convention).
    """
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    ee_pos = obs["robot0_eef_pos"]          # (3,)
    ee_quat = obs["robot0_eef_quat"][:3]    # (3,) — take first 3 of quaternion to match training ee_ori
    gripper = obs["robot0_gripper_qpos"]     # (2,)
    proprio = np.concatenate([ee_pos, ee_quat, gripper]).astype(np.float32)
    return {
        "agentview": agentview,        # (H, W, 3) uint8
        "wrist": wrist,                # (H, W, 3) uint8
        "proprio": proprio,            # (8,) float32
    }


# ---------------------------------------------------------------------------
# Policy wrapper (simplified from skill_policy.py for evaluation)
# ---------------------------------------------------------------------------

class EvalPolicy:
    """Wraps action head manager + Qwen planner for evaluation rollouts."""

    def __init__(
        self,
        ckpt_dir: str,
        head_type: str = "diffusion",
        model_path: str = "/export/ra/liyuxuan/VLA/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct",
        device: str = "cuda",
        T_obs: int = 2,
        T_pred: int = 16,
        check_interval: int = 8,
        use_ema: bool = True,
        l1_class: str | None = None,
    ):
        self.device = torch.device(device)
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.check_interval = check_interval
        self.head_type = head_type
        self.force_l1_class = l1_class  # if set, override VerbNet parsing

        # ------- Qwen planner -------
        logger.info("Loading QwenPlanner from %s", model_path)
        self.planner = QwenPlanner(model_path=model_path, device=device)

        # ------- Action head manager -------
        logger.info("Loading ActionHeadManager from %s", ckpt_dir)
        # Discover L1 classes from checkpoint directory.
        # Training saves as: {ckpt_dir}/{key}/{head_type}/best.pt
        # where key = l1_class.replace(".", "_"), e.g. put-9_1-2, push-12-1
        # Recover original class names by reading the checkpoint metadata.
        l1_classes = []
        for entry in sorted(os.listdir(ckpt_dir)):
            ckpt_path = os.path.join(ckpt_dir, entry, head_type, "best.pt")
            if os.path.isfile(ckpt_path):
                # Read l1_class from checkpoint if stored
                meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if "l1_class" in meta:
                    l1_classes.append(meta["l1_class"])
                else:
                    # Fallback: directory name is the key; dots were replaced
                    l1_classes.append(entry)
                del meta
        if not l1_classes:
            raise RuntimeError(f"No checkpoints found under {ckpt_dir}/*/{head_type}/best.pt")
        logger.info("Found L1 classes: %s", l1_classes)

        self.manager = ActionHeadManager(
            head_type=head_type,
            l1_classes=l1_classes,
            T_obs=T_obs,
            T_pred=T_pred,
        ).to(self.device)
        self.manager.load_all(ckpt_dir, use_ema=use_ema)
        self.manager.eval()

        # Reliable key→class mapping for fallback
        self._available_classes = {cls.replace('.', '_'): cls for cls in l1_classes}
        self._fallback_class = l1_classes[0]

        # ------- Action stats for denormalization -------
        self.action_stats: dict[str, dict] = {}
        for cls_name in l1_classes:
            key = cls_name.replace(".", "_")
            stats_path = os.path.join(ckpt_dir, key, head_type, "action_stats.json")
            if os.path.isfile(stats_path):
                with open(stats_path) as f:
                    self.action_stats[cls_name] = json.load(f)

    def _denormalize(self, action: np.ndarray, l1_class: str) -> np.ndarray:
        if l1_class not in self.action_stats:
            return action
        stats = self.action_stats[l1_class]
        a_min = np.array(stats["min"], dtype=np.float32)
        a_max = np.array(stats["max"], dtype=np.float32)
        rng = np.maximum(a_max - a_min, 1e-6)
        return ((action + 1) / 2 * rng + a_min).astype(np.float32)

    def _obs_to_tensor(
        self,
        av_history: list[np.ndarray],
        wr_history: list[np.ndarray],
        pr_history: list[np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        av = np.stack(av_history[-self.T_obs:])            # (T, H, W, 3)
        wr = np.stack(wr_history[-self.T_obs:])
        pr = np.stack(pr_history[-self.T_obs:])            # (T, 8)

        av = torch.from_numpy(av.transpose(0, 3, 1, 2).astype(np.float32) / 255.0)
        wr = torch.from_numpy(wr.transpose(0, 3, 1, 2).astype(np.float32) / 255.0)
        pr = torch.from_numpy(pr.astype(np.float32))

        return (
            av.unsqueeze(0).to(self.device),   # (1, T, 3, H, W)
            wr.unsqueeze(0).to(self.device),
            pr.unsqueeze(0).to(self.device),   # (1, T, 8)
        )

    @staticmethod
    def _draw_text(img: np.ndarray, lines: list[str], font_scale: float = 0.35,
                   color=(255, 255, 255), bg_color=(0, 0, 0)) -> np.ndarray:
        """Draw multi-line text with background on an image (in-place)."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        y = 12
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
            cv2.rectangle(img, (0, y - th - 2), (tw + 4, y + 4), bg_color, -1)
            cv2.putText(img, line, (2, y), font, font_scale, color, thickness, cv2.LINE_AA)
            y += th + 6
        return img

    @torch.inference_mode()
    def rollout(
        self,
        env: OffScreenRenderEnv,
        task_language: str,
        init_state,
        max_steps: int = 600,
        video_path: str | None = None,
    ) -> dict:
        """Run one evaluation episode.

        Args:
            video_path: If set, save an mp4 video of the episode with subtask overlay.

        Returns dict with:
            success: bool (env done flag)
            steps: int
            subtasks_completed: int
            total_subtasks: int
            subtasks: list[str]
        """
        record_video = video_path is not None
        frames: list[np.ndarray] = []
        # Reset env
        env.reset()
        raw_obs = env.set_init_state(init_state)

        # Settle physics with dummy actions
        dummy = np.zeros(7)
        for _ in range(5):
            raw_obs, _, _, _ = env.step(dummy)

        obs = extract_obs(raw_obs)

        # ---------- Qwen: generate subtasks from initial image ----------
        subtasks = self.planner.get_subtasks(
            high_task=task_language,
            image_list=[obs["agentview"], obs["wrist"]],
        )
        if not subtasks:
            logger.warning("Qwen returned empty subtasks for '%s'", task_language)
            return {
                "success": False, "steps": 0,
                "subtasks_completed": 0, "total_subtasks": 0,
                "subtasks": [],
            }
        logger.info("  Subtasks: %s", subtasks)

        # ---------- Rollout ----------
        current_idx = 0
        finished_subtasks: list[str] = []
        total_steps = 0
        done = False

        av_hist = [obs["agentview"]]
        wr_hist = [obs["wrist"]]
        pr_hist = [obs["proprio"]]
        # Pad to T_obs
        while len(av_hist) < self.T_obs:
            av_hist.insert(0, av_hist[0].copy())
            wr_hist.insert(0, wr_hist[0].copy())
            pr_hist.insert(0, pr_hist[0].copy())

        steps_since_check = 0

        # Record initial frame
        if record_video:
            frame = cv2.resize(obs["agentview"], (256, 256))
            self._draw_text(frame, [
                f"Task: {task_language}",
                f"Subtask[0]: {subtasks[0] if subtasks else '?'}",
                "Step: 0",
            ])
            frames.append(frame)

        # Resolve L1 class for a subtask index (or use last head after all subtasks done)
        def _resolve_l1(idx: int) -> str:
            if self.force_l1_class:
                return self.force_l1_class
            if idx < len(subtasks):
                parsed = _parse_subtask_verbnet(subtasks[idx])
                cls = parsed["verbnet_class"]
            else:
                # All subtasks done — reuse last subtask's class
                parsed = _parse_subtask_verbnet(subtasks[-1])
                cls = parsed["verbnet_class"]
            key = cls.replace(".", "_")
            if key not in self._available_classes:
                logger.warning("No head for L1=%s, falling back to %s", cls, self._fallback_class)
                return self._fallback_class
            return cls

        all_subtasks_done = False

        while total_steps < max_steps and not done:
            l1_class = _resolve_l1(current_idx)

            # Predict action chunk
            av_t, wr_t, pr_t = self._obs_to_tensor(av_hist, wr_hist, pr_hist)
            action_chunk = self.manager.predict(av_t, wr_t, pr_t, l1_class)
            action_chunk = action_chunk[0].cpu().numpy()  # (T_pred, 7)
            action_chunk = self._denormalize(action_chunk, l1_class)

            # Execute chunk step by step
            for act in action_chunk:
                raw_obs, reward, done, info = env.step(act)
                total_steps += 1
                steps_since_check += 1

                obs = extract_obs(raw_obs)
                av_hist.append(obs["agentview"])
                wr_hist.append(obs["wrist"])
                pr_hist.append(obs["proprio"])

                # Record frame with annotation
                if record_video:
                    frame = cv2.resize(obs["agentview"], (256, 256))
                    if current_idx < len(subtasks):
                        sub_label = f"Subtask[{current_idx}]: {subtasks[current_idx]}"
                    else:
                        sub_label = "[All subtasks done, continuing...]"
                    self._draw_text(frame, [
                        f"Task: {task_language}",
                        sub_label,
                        f"Step: {total_steps}  L1: {l1_class}",
                    ])
                    frames.append(frame)

                # Bound history
                if len(av_hist) > self.T_obs * 4:
                    av_hist = av_hist[-self.T_obs * 2:]
                    wr_hist = wr_hist[-self.T_obs * 2:]
                    pr_hist = pr_hist[-self.T_obs * 2:]

                if done:
                    break

                # Periodic subtask completion check (only if subtasks remain)
                if not all_subtasks_done and steps_since_check >= self.check_interval:
                    completed = self.planner.check_subtask(
                        high_task=task_language,
                        image_list=[obs["agentview"], obs["wrist"]],
                        current_subtask=subtasks[current_idx],
                        all_subtasks=subtasks,
                        finished_subtasks=finished_subtasks,
                    )
                    steps_since_check = 0

                    if completed:
                        logger.info("    Subtask[%d] '%s' DONE at step %d",
                                    current_idx, subtasks[current_idx], total_steps)
                        finished_subtasks.append(subtasks[current_idx])
                        current_idx += 1
                        if current_idx >= len(subtasks):
                            all_subtasks_done = True
                            logger.info("    All subtasks marked done at step %d, "
                                        "continuing with last head until env done.",
                                        total_steps)
                        break

                if total_steps >= max_steps:
                    break

        # Final success: use env's done signal
        success = bool(done)

        # Save video
        if record_video and frames:
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            # Add SUCCESS/FAIL banner to last 10 frames
            tag = "SUCCESS" if success else "FAIL"
            tag_color = (0, 255, 0) if success else (0, 0, 255)
            for f in frames[-min(10, len(frames)):]:
                cv2.putText(f, tag, (80, 140), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, tag_color, 2, cv2.LINE_AA)
            writer = cv2.VideoWriter(
                video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                20, (frames[0].shape[1], frames[0].shape[0]),
            )
            for f in frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            logger.info("  Video saved: %s (%d frames)", video_path, len(frames))

        return {
            "success": success,
            "steps": total_steps,
            "subtasks_completed": len(finished_subtasks),
            "total_subtasks": len(subtasks),
            "subtasks": subtasks,
        }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_all(args):
    """Evaluate across LIBERO-90 tasks."""
    device = args.device

    # Build policy
    policy = EvalPolicy(
        ckpt_dir=args.ckpt_dir,
        head_type=args.head_type,
        model_path=args.model_path,
        device=device,
        T_obs=args.T_obs,
        T_pred=args.T_pred,
        check_interval=args.check_interval,
        use_ema=args.use_ema,
        l1_class=args.l1_class,
    )

    # LIBERO benchmark setup
    task_suite = benchmark.get_benchmark_dict()["libero_90"]()
    n_tasks = task_suite.n_tasks
    if args.max_tasks is not None:
        n_tasks = min(n_tasks, args.max_tasks)
    bddl_root = Path(get_libero_path("bddl_files"))

    # Optional: restrict to specific task IDs
    if args.task_ids:
        task_ids = [int(x) for x in args.task_ids.split(",")]
    else:
        task_ids = list(range(n_tasks))

    logger.info("Evaluating %d tasks, %d trials each, max %d steps",
                len(task_ids), args.n_eval, args.max_steps)

    # Results storage
    all_results = []
    task_success_rates = {}

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        task_language = task.language
        bddl_file = str(bddl_root / task.problem_folder / task.bddl_file)
        init_states = task_suite.get_task_init_states(task_id)

        logger.info("=" * 60)
        logger.info("Task %d/%d: %s", task_id, len(task_ids), task_language)
        logger.info("=" * 60)

        n_success = 0
        task_results = []

        for trial in range(args.n_eval):
            # Cycle through init states
            init_state = init_states[trial % len(init_states)]

            try:
                env = build_env(bddl_file, img_size=args.img_size)
                env.seed(args.seed + trial)

                t0 = time.time()
                # Video path
                vid_path = None
                if args.save_video:
                    vid_dir = os.path.join(args.video_dir, f"task_{task_id}")
                    vid_path = os.path.join(vid_dir, f"trial_{trial}.mp4")

                result = policy.rollout(
                    env=env,
                    task_language=task_language,
                    init_state=init_state,
                    max_steps=args.max_steps,
                    video_path=vid_path,
                )
                elapsed = time.time() - t0

                result["task_id"] = task_id
                result["task_language"] = task_language
                result["trial"] = trial
                result["time"] = round(elapsed, 1)
                task_results.append(result)

                if result["success"]:
                    n_success += 1

                logger.info(
                    "  Trial %d/%d: %s | steps=%d | subtasks=%d/%d | %.1fs",
                    trial + 1, args.n_eval,
                    "SUCCESS" if result["success"] else "FAIL",
                    result["steps"],
                    result["subtasks_completed"],
                    result["total_subtasks"],
                    elapsed,
                )

            except Exception as e:
                logger.error("  Trial %d EXCEPTION: %s", trial, e, exc_info=True)
                task_results.append({
                    "task_id": task_id,
                    "task_language": task_language,
                    "trial": trial,
                    "success": False,
                    "error": str(e),
                })
            finally:
                try:
                    env.close()
                except Exception:
                    pass
                del env
                gc.collect()

        sr = n_success / max(args.n_eval, 1)
        task_success_rates[task_id] = sr
        all_results.extend(task_results)

        logger.info("Task %d success rate: %.1f%% (%d/%d)",
                    task_id, sr * 100, n_success, args.n_eval)

    # Aggregate
    overall_sr = np.mean(list(task_success_rates.values())) if task_success_rates else 0.0
    logger.info("=" * 60)
    logger.info("OVERALL success rate: %.1f%% (mean over %d tasks)",
                overall_sr * 100, len(task_success_rates))
    logger.info("=" * 60)

    # Per-task summary table
    for tid, sr in sorted(task_success_rates.items()):
        tname = task_suite.get_task(tid).language
        logger.info("  Task %2d: %5.1f%% | %s", tid, sr * 100, tname)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, "eval_results.json")
    summary = {
        "head_type": args.head_type,
        "ckpt_dir": args.ckpt_dir,
        "n_eval": args.n_eval,
        "max_steps": args.max_steps,
        "overall_success_rate": round(overall_sr, 4),
        "task_success_rates": {str(k): round(v, 4) for k, v in task_success_rates.items()},
        "per_trial": all_results,
    }
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Results saved to %s", results_file)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate action heads on LIBERO-90")

    # Model
    parser.add_argument("--head_type", choices=["diffusion", "act"], default="diffusion")
    parser.add_argument("--ckpt_dir", default="checkpoints",
                        help="Root checkpoint dir containing per-L1-class subdirs")
    parser.add_argument("--model_path",
                        default="/export/ra/liyuxuan/VLA/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct",
                        help="Path to Qwen VLM for subtask gen/check")
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--l1_class", type=str, default=None,
                        help="Force all subtasks to use this L1 class head "
                             "(e.g. 'push-12-1'), bypassing VerbNet parsing")

    # Eval config
    parser.add_argument("--n_eval", type=int, default=20,
                        help="Number of evaluation trials per task")
    parser.add_argument("--max_steps", type=int, default=600,
                        help="Max env steps per trial")
    parser.add_argument("--max_tasks", type=int, default=None,
                        help="Limit number of tasks (default: all 90)")
    parser.add_argument("--task_ids", type=str, default=None,
                        help="Comma-separated task IDs to evaluate (default: all)")
    parser.add_argument("--check_interval", type=int, default=8,
                        help="Steps between Qwen subtask completion checks")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)

    # Policy config
    parser.add_argument("--T_obs", type=int, default=2)
    parser.add_argument("--T_pred", type=int, default=16)

    # Output
    parser.add_argument("--output_dir", default="eval_results")
    parser.add_argument("--device", default="cuda")

    # Video
    parser.add_argument("--save_video", action="store_true", default=False,
                        help="Save mp4 video for each trial with subtask overlay")
    parser.add_argument("--video_dir", default="eval_videos",
                        help="Directory to save evaluation videos")

    args = parser.parse_args()
    if args.no_ema:
        args.use_ema = False

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    evaluate_all(args)


if __name__ == "__main__":
    main()
