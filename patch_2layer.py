import re
import os
_TSL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_skill_lib.py')
with open(_TSL_PATH, 'r') as f:
    content = f.read()

old_logic = """    # 1. Action
    print(f"\\n[Segmented-Hier] Level 1: Action clustering (threshold=0.3)")
    l1_labels = cluster_subset(s_action, threshold=0.3)
    
    # 2. Object
    l2_labels = np.zeros(N, dtype=int)
    max_l2 = 0
    for c1 in np.unique(l1_labels):
        mask1 = (l1_labels == c1)
        sub_l2 = cluster_subset(s_object[mask1], threshold=0.5)
        l2_labels[mask1] = sub_l2 + max_l2
        max_l2 += (sub_l2.max() + 1)
    
    # 3. Context
    l3_labels = np.zeros(N, dtype=int)
    max_l3 = 0
    for c2 in np.unique(l2_labels):
        mask2 = (l2_labels == c2)
        sub_l3 = cluster_subset(s_context[mask2], threshold=0.7)
        l3_labels[mask2] = sub_l3 + max_l3
        max_l3 += (sub_l3.max() + 1)
        
    labels = l3_labels
    n_clusters = max_l3
    print(f"[Segmented-Hier] Found {n_clusters} clusters at leaf level (Action -> Object -> Context).")
    
    center_indices: dict[int, int] = {}
    center_texts:   dict[int, str] = {}
    for cid in range(n_clusters):
        mask = np.where(labels == cid)[0]
        centroid = embeddings[mask].mean(axis=0)
        dists = np.linalg.norm(embeddings[mask] - centroid, axis=1)
        closest_global = int(mask[int(np.argmin(dists))])
        center_indices[cid] = closest_global
        center_texts[cid] = f"A{l1_labels[closest_global]}.O{l2_labels[closest_global]}.C{cid} | " + subtask_texts[closest_global]"""

new_logic = """    from sklearn.preprocessing import normalize
    
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

if old_logic in content:
    content = content.replace(old_logic, new_logic)
    
    # Also update the title
    content = content.replace('f"Segmented Hierarchical Skills (3-Level)\\n{n_clusters} clusters (thr=A:0.3/O:0.5/C:0.7)"', 'f"Segmented Hierarchical Skills (2-Level)\\n{n_clusters} clusters (thr=A:0.3/C:{dist_threshold})"')
    
    with open(_TSL_PATH, 'w') as f:
        f.write(content)
    print("Patched successfully")
else:
    print("Could not find the target string to replace.")
