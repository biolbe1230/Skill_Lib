"""
Offline trajectory segmentation using Qwen3-VL check_subtask().

For each LIBERO-90 HDF5 demo file, this script:
  1. Extracts the task description from the filename.
  2. Matches it to subtasks_by_task.json to get the subtask list.
  3. For each demo, runs Qwen3-VL check_subtask() at regular intervals
     to determine when each subtask completes.
  4. Labels each segment with its VerbNet L1 class.
  5. Saves the result as segmented_demos.json.

Usage:
    python -m data.segment_trajectories \
        --demo_dir /data/datasets/liyuxuan/datasets/libero_90/ \
        --subtasks_json skill_lib_results_clip_film_raw/subtasks_by_task.json \
        --output segmented_demos.json \
        --check_interval 10 \
        --model_path /export/ra/liyuxuan/VLA/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np

# Add parent to path so we can import siblings
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verbnet_utils import _parse_subtask_verbnet
from QwenPlanner import QwenPlanner

logger = logging.getLogger(__name__)


def _task_from_filename(fname: str) -> str:
    """Extract task description from HDF5 filename.

    E.g. 'KITCHEN_SCENE10_close_the_top_drawer_demo.hdf5'
      -> 'close the top drawer'
    """
    stem = Path(fname).stem  # remove .hdf5
    stem = stem.removesuffix("_demo")
    # Remove leading SCENE_TYPE_SCENEN_ prefix
    m = re.match(r"^[A-Z_]+SCENE\d+_(.+)$", stem)
    if m:
        return m.group(1).replace("_", " ")
    return stem.replace("_", " ")


def _build_task_lookup(subtasks_json_path: str) -> dict[str, list[str]]:
    """Build mapping from task_language -> subtask list."""
    with open(subtasks_json_path) as f:
        tasks = json.load(f)
    return {t["task_language"]: t["subtasks"] for t in tasks}


def segment_one_demo(
    planner: QwenPlanner,
    agentview: np.ndarray,    # (T, H, W, 3) uint8
    wrist: np.ndarray,        # (T, H, W, 3) uint8
    task_language: str,
    subtasks: list[str],
    check_interval: int = 10,
) -> list[dict]:
    """Segment a single demo trajectory into subtask segments.

    Returns list of dicts:
        [{"start": int, "end": int, "subtask": str, "l1_class": str}, ...]
    """
    T = len(agentview)
    segments = []
    current_idx = 0
    start_frame = 0
    finished = []

    for t in range(0, T, check_interval):
        if current_idx >= len(subtasks):
            break

        # Use the last frame in the check window
        frame_idx = min(t + check_interval - 1, T - 1)
        img_av = agentview[frame_idx]   # (H,W,3) uint8
        img_wr = wrist[frame_idx]       # (H,W,3) uint8

        completed = planner.check_subtask(
            high_task=task_language,
            image_list=[img_av, img_wr],
            current_subtask=subtasks[current_idx],
            all_subtasks=subtasks,
            finished_subtasks=finished,
        )

        if completed:
            parsed = _parse_subtask_verbnet(subtasks[current_idx])
            segments.append({
                "start": int(start_frame),
                "end": int(frame_idx),
                "subtask": subtasks[current_idx],
                "l1_class": parsed["verbnet_class"],
                "verb_phrase": parsed["verb_phrase"],
            })
            finished.append(subtasks[current_idx])
            current_idx += 1
            start_frame = frame_idx + 1

    # If there are remaining frames after the last detected subtask,
    # assign them to the last subtask (or current if not yet completed)
    if current_idx < len(subtasks) and start_frame < T:
        parsed = _parse_subtask_verbnet(subtasks[current_idx])
        segments.append({
            "start": int(start_frame),
            "end": int(T - 1),
            "subtask": subtasks[current_idx],
            "l1_class": parsed["verbnet_class"],
            "verb_phrase": parsed["verb_phrase"],
        })

    return segments


def segment_all(
    demo_dir: str,
    subtasks_json: str,
    output_path: str,
    check_interval: int = 10,
    model_path: str | None = None,
    max_demos_per_task: int | None = None,
):
    """Segment all demos and save to JSON."""
    task_lookup = _build_task_lookup(subtasks_json)
    logger.info("Loaded %d tasks from %s", len(task_lookup), subtasks_json)

    # Initialize Qwen planner
    planner = QwenPlanner(model_path=model_path)
    logger.info("Qwen planner loaded")

    hdf5_files = sorted(
        f for f in os.listdir(demo_dir)
        if f.endswith(".hdf5")
    )
    logger.info("Found %d HDF5 files", len(hdf5_files))

    all_results = []
    skipped_tasks = []

    for fi, fname in enumerate(hdf5_files):
        task_lang = _task_from_filename(fname)
        if task_lang not in task_lookup:
            logger.warning("Task '%s' (from %s) not in subtasks JSON, skipping", task_lang, fname)
            skipped_tasks.append(task_lang)
            continue

        subtasks = task_lookup[task_lang]
        fpath = os.path.join(demo_dir, fname)
        logger.info("[%d/%d] Processing %s (%d subtasks)", fi + 1, len(hdf5_files), fname, len(subtasks))

        with h5py.File(fpath, "r") as hf:
            demo_keys = sorted(hf["data"].keys())
            if max_demos_per_task is not None:
                demo_keys = demo_keys[:max_demos_per_task]

            for dk in demo_keys:
                obs = hf["data"][dk]["obs"]
                agentview = obs["agentview_rgb"][:]       # (T, 128, 128, 3)
                wrist_img = obs["eye_in_hand_rgb"][:]     # (T, 128, 128, 3)

                segments = segment_one_demo(
                    planner=planner,
                    agentview=agentview,
                    wrist=wrist_img,
                    task_language=task_lang,
                    subtasks=subtasks,
                    check_interval=check_interval,
                )

                all_results.append({
                    "hdf5_file": fname,
                    "demo_key": dk,
                    "task_language": task_lang,
                    "subtasks": subtasks,
                    "segments": segments,
                })

    # Save results
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("Saved %d segmented demos to %s", len(all_results), output_path)
    if skipped_tasks:
        logger.warning("Skipped %d tasks: %s", len(skipped_tasks), skipped_tasks)

    # Print segment statistics per L1 class
    l1_counts: dict[str, int] = {}
    for item in all_results:
        for seg in item["segments"]:
            c = seg["l1_class"]
            l1_counts[c] = l1_counts.get(c, 0) + 1
    logger.info("Segment counts per L1 class:")
    for c, n in sorted(l1_counts.items(), key=lambda x: -x[1]):
        logger.info("  %s: %d", c, n)


def main():
    parser = argparse.ArgumentParser(description="Offline trajectory segmentation")
    parser.add_argument(
        "--demo_dir",
        default="/data/datasets/liyuxuan/datasets/libero_90/",
        help="Directory containing LIBERO HDF5 demo files",
    )
    parser.add_argument(
        "--subtasks_json",
        default="skill_lib_results_clip_film_raw/subtasks_by_task.json",
        help="Path to subtasks_by_task.json",
    )
    parser.add_argument(
        "--output",
        default="segmented_demos.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--check_interval",
        type=int,
        default=10,
        help="Check subtask completion every N frames",
    )
    parser.add_argument(
        "--model_path",
        default="/export/ra/liyuxuan/VLA/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct",
        help="Path to Qwen3-VL model",
    )
    parser.add_argument(
        "--max_demos_per_task",
        type=int,
        default=None,
        help="Limit demos per task (for debugging)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    segment_all(
        demo_dir=args.demo_dir,
        subtasks_json=args.subtasks_json,
        output_path=args.output,
        check_interval=args.check_interval,
        model_path=args.model_path,
        max_demos_per_task=args.max_demos_per_task,
    )


if __name__ == "__main__":
    main()
