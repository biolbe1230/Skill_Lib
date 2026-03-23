

import argparse
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from PIL import Image

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def build_env(task_bddl_file: Path, seed: int) -> OffScreenRenderEnv:
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(seed)
    return env


def obs_to_images(obs: dict):
    main_img  = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return main_img, wrist_img


def save_artifact(output_dir: Path, task_id: int, task_language: str,
                  main_img: np.ndarray, wrist_img: np.ndarray, subtasks: list):
    task_dir = output_dir / "task_artifacts" / f"task_{task_id:03d}"
    task_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(main_img.astype(np.uint8)).save(task_dir / "agentview.png")
    Image.fromarray(wrist_img.astype(np.uint8)).save(task_dir / "wrist.png")
    (task_dir / "subtasks.json").write_text(
        json.dumps(
            {"task_id": task_id, "task_language": task_language, "subtasks": subtasks},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"  [task {task_id:03d}] saved → {task_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True,
                        help="skill_lib output dir containing subtasks_by_task.json")
    parser.add_argument("--task_suite_name", type=str, default="libero_90")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing task_artifacts entries")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    subtasks_file = source_dir / "subtasks_by_task.json"
    if not subtasks_file.exists():
        raise FileNotFoundError(f"subtasks_by_task.json not found in {source_dir}")

    tasks_data = json.loads(subtasks_file.read_text(encoding="utf-8"))
    print(f"Loaded {len(tasks_data)} tasks from {subtasks_file}")

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()

    bddl_root = Path(get_libero_path("bddl_files"))

    for entry in tasks_data:
        task_id = entry["task_id"]
        task_language = entry["task_language"]
        subtasks = entry["subtasks"]

        task_dir = source_dir / "task_artifacts" / f"task_{task_id:03d}"
        if task_dir.exists() and not args.overwrite:
            print(f"  [task {task_id:03d}] already exists, skipping (use --overwrite to force)")
            continue

        task = task_suite.get_task(task_id)
        task_bddl_file = bddl_root / task.problem_folder / task.bddl_file

        env = build_env(task_bddl_file, args.seed)
        try:
            init_states = task_suite.get_task_init_states(task_id)
            env.reset()
            obs = env.set_init_state(init_states[args.episode_idx])
            main_img, wrist_img = obs_to_images(obs)
        finally:
            env.close()
            gc.collect()

        save_artifact(source_dir, task_id, task_language, main_img, wrist_img, subtasks)

    print(f"\nDone. task_artifacts saved under {source_dir / 'task_artifacts'}")


if __name__ == "__main__":
    main()
