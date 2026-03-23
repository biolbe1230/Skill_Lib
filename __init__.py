# Skill Library for LIBERO-VLA
#
# This package contains all skill library building, clustering,
# retrieval, and visualization code, separated from the core LIBERO framework.
#
# Key modules:
#   QwenPlanner.py              - Qwen VLM planner interface
#   get_planner_output.py       - Minimal planner demo script
#   test_skill_lib.py           - Master skill library builder (encoders, clustering, main)
#   verbnet_utils.py            - Shared VerbNet subtask parsing
#   build_contrastive_skill_emb.py - Contrastive projection training
#   build_task_artifacts.py     - Render & cache task initial observations
#   skill_retriever.py          - 3-layer hierarchical skill retrieval
#   film_encoder.py             - FiLM-conditioned CLIP+ResNet encoder
#   visualize_segmented.py      - t-SNE visualizations of segmented embeddings
#   recluster_verbnet.py        - Quick re-cluster with VerbNet hierarchy
#   recluster_2layer.py         - Quick re-cluster with 2-layer segmentation
#   patch_2layer.py             - Patch test_skill_lib for 2-layer clustering
#   patch_verbnet_1stlayer.py   - Patch test_skill_lib for VerbNet L1
#
# Training integration (remains in libero/lifelong/):
#   algos/skill_library.py      - SkillLibraryBuilder lifelong algo
#   datasets.py                 - SkillLabeledVLDataset
#   main_skill.py               - Hydra training entry point
