import json
import sys
from pathlib import Path
import numpy as np

# Load co-located test_skill_lib
import importlib.util
_skill_lib_dir = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("test_skill_lib", str(_skill_lib_dir / "test_skill_lib.py"))
tsl = importlib.util.module_from_spec(spec)
sys.modules["test_skill_lib"] = tsl
spec.loader.exec_module(tsl)

def main():
    source_dir = Path("skill_lib_results_full")
    output_dir = Path("skill_lib_results_segmented2")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(source_dir / "subtasks_by_task.json", "r") as f:
        subtasks_by_task = json.load(f)
    s_env_task = np.load(source_dir / "s_env_embeddings.npy")
    
    all_subtask_texts = []
    all_task_ids = []
    task_languages = []
    
    for i, meta in enumerate(subtasks_by_task):
        subtasks = meta["subtasks"]
        task_languages.append(meta.get("task_language", f"Task {i}"))
        for text in subtasks:
            all_subtask_texts.append(text)
            all_task_ids.append(i)

    seg_result = tsl.build_segmented_skill_embeddings(
        subtask_texts=all_subtask_texts,
        task_ids=all_task_ids,
        s_env=s_env_task,
        sbert_model="all-MiniLM-L6-v2",
        d_action=16,
        d_object=40,
        d_context=32,
        seed=42,
    )
    
    tsl.segmented_hierarchical_cluster(
        subtask_texts=all_subtask_texts,
        task_ids=all_task_ids,
        seg_result=seg_result,
        dist_threshold=0.65,
        output_dir=output_dir,
        task_languages=task_languages,
    )

if __name__ == "__main__":
    main()
