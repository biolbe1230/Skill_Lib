import re
import os
_TSL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_skill_lib.py')
with open(_TSL_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Modify build_segmented_skill_embeddings to return verbnet_classes
old_parse1 = """    # ---- 1. Parse subtasks into action / object texts ----
    action_texts: list[str] = []
    object_texts: list[str] = []
    for text in subtask_texts:
        parsed = _parse_subtask_verbnet(text)
        action_texts.append(parsed["verb_phrase"])
        object_texts.append(parsed["object_phrase"] or "object")"""
        
new_parse1 = """    # ---- 1. Parse subtasks into action / object texts ----
    action_texts: list[str] = []
    object_texts: list[str] = []
    verbnet_classes: list[str] = []
    for text in subtask_texts:
        parsed = _parse_subtask_verbnet(text)
        action_texts.append(parsed["verb_phrase"])
        object_texts.append(parsed["object_phrase"] or "object")
        verbnet_classes.append(parsed["verbnet_class"])"""
        
if old_parse1 in content:
    content = content.replace(old_parse1, new_parse1)
else:
    print("Cannot find old_parse1")

old_ret1 = """        "embeddings": embeddings,
        "action_texts": action_texts,
        "object_texts": object_texts,"""

new_ret1 = """        "embeddings": embeddings,
        "action_texts": action_texts,
        "object_texts": object_texts,
        "verbnet_classes": verbnet_classes,"""

if old_ret1 in content:
    content = content.replace(old_ret1, new_ret1)
else:
    print("Cannot find old_ret1")


# 2. Modify segmented_hierarchical_cluster to use verbnet_classes
old_seg_clust_start = """    s_action = seg_result["s_action"]
    s_object = seg_result["s_object"]
    s_context = seg_result["s_context"]
    embeddings = seg_result["embeddings"]"""

new_seg_clust_start = """    s_action = seg_result["s_action"]
    s_object = seg_result["s_object"]
    s_context = seg_result["s_context"]
    embeddings = seg_result["embeddings"]
    verbnet_classes = np.array(seg_result["verbnet_classes"])"""

if old_seg_clust_start in content:
    content = content.replace(old_seg_clust_start, new_seg_clust_start)
else:
    print("Cannot find old_seg_clust_start")

old_seg_logic = """    from sklearn.preprocessing import normalize
    
    # 1. Level 1: Action (Big action mode)
    # We use a relatively strict threshold on pure action vectors to group verbs cleanly
    print(f"\\n[Segmented-Hier] Level 1: Action clustering (threshold=0.3 on normal action)")
    l1_labels = cluster_subset(s_action, threshold=0.3)
    
    # 2. Level 2: Combined (Action + Object + Context)
    # We cluster the full embedding within each action mode
    # dist_threshold is passed from the command line (default ~0.6)
    l2_labels = np.zeros(N, dtype=int)
    max_l2 = 0
    
    # normalize full embeddings first
    norm_embeddings = normalize(embeddings)
    
    for c1 in np.unique(l1_labels):
        mask1 = (l1_labels == c1)
        sub_l2 = cluster_subset(norm_embeddings[mask1], threshold=dist_threshold)
        l2_labels[mask1] = sub_l2 + max_l2
        max_l2 += (sub_l2.max() + 1)
        
    labels = l2_labels
    n_clusters = max_l2
    print(f"[Segmented-Hier] Found {n_clusters} clusters at leaf level (Action Mode -> Full Context).")
    
    center_indices: dict[int, int] = {}
    center_texts:   dict[int, str] = {}
    for cid in range(n_clusters):
        mask = np.where(labels == cid)[0]
        centroid = norm_embeddings[mask].mean(axis=0)
        dists = np.linalg.norm(norm_embeddings[mask] - centroid, axis=1)
        closest_global = int(mask[int(np.argmin(dists))])
        center_indices[cid] = closest_global
        center_texts[cid] = f"ActionMode_{l1_labels[closest_global]}.Cluster_{cid} | " + subtask_texts[closest_global]"""

new_seg_logic = """    from sklearn.preprocessing import normalize
    
    # 1. Level 1: Action (Big action mode)
    # We use VerbNet classes directly to group verbs properly into robust modes
    unique_vn_classes = np.unique(verbnet_classes)
    print(f"\\n[Segmented-Hier] Level 1: Action grouping based on {len(unique_vn_classes)} VerbNet classes")
    
    l1_labels_str = verbnet_classes
    
    # 2. Level 2: Combined (Action + Object + Context)
    # We cluster the full structured embedding within each VerbNet action mode
    # dist_threshold is passed from the command line (default ~0.6)
    l2_labels = np.zeros(N, dtype=int)
    max_l2 = 0
    
    # normalize full embeddings first
    norm_embeddings = normalize(embeddings)
    
    for c1_str in unique_vn_classes:
        mask1 = (l1_labels_str == c1_str)
        sub_l2 = cluster_subset(norm_embeddings[mask1], threshold=dist_threshold)
        l2_labels[mask1] = sub_l2 + max_l2
        max_l2 += (sub_l2.max() + 1)
        
    labels = l2_labels
    n_clusters = max_l2
    print(f"[Segmented-Hier] Found {n_clusters} clusters at leaf level (VerbNet Class -> Composite Embedding).")
    
    center_indices: dict[int, int] = {}
    center_texts:   dict[int, str] = {}
    for cid in range(n_clusters):
        mask = np.where(labels == cid)[0]
        centroid = norm_embeddings[mask].mean(axis=0)
        dists = np.linalg.norm(norm_embeddings[mask] - centroid, axis=1)
        closest_global = int(mask[int(np.argmin(dists))])
        center_indices[cid] = closest_global
        center_texts[cid] = f"{l1_labels_str[closest_global]}.C{cid} | " + subtask_texts[closest_global]"""


if old_seg_logic in content:
    content = content.replace(old_seg_logic, new_seg_logic)
else:
    print("Cannot find old_seg_logic")

with open(_TSL_PATH, 'w', encoding='utf-8') as f:
    f.write(content)
print("Patch script completed.")
