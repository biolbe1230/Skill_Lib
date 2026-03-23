#!/usr/bin/env python3
"""Visualize three-segment skill embeddings with t-SNE + clustering."""
import sys, os
_SKILL_LIB_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SKILL_LIB_DIR)
sys.path.insert(0, _REPO_ROOT)
os.environ['NLTK_DATA'] = '/export/ra/liyuxuan/nltk_data'

import json
import numpy as np
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

# ---- paths (relative to Skill_Lib/) ----
_SKILL_LIB = Path(_SKILL_LIB_DIR)
SEG_DIR  = _SKILL_LIB / "skill_lib_results_segmented"
VN_DIR   = _SKILL_LIB / "skill_lib_results_verbnet"
OUT_DIR  = SEG_DIR

# ---- load data ----
seg_action  = np.load(SEG_DIR / "seg_action.npy")
seg_object  = np.load(SEG_DIR / "seg_object.npy")
seg_context = np.load(SEG_DIR / "seg_context.npy")
seg_all     = np.load(SEG_DIR / "seg_embeddings.npy")

with open(SEG_DIR / "segment_info.json") as f:
    seg_info = json.load(f)
action_texts = seg_info["action_texts"]
object_texts = seg_info["object_texts"]

with open(VN_DIR / "subtask_texts.json") as f:
    vn_meta = json.load(f)
subtask_texts = vn_meta["texts"]
task_ids      = vn_meta["task_ids"]

N = len(subtask_texts)
print(f"Loaded {N} subtasks,  seg_all={seg_all.shape}")

# ---- import VerbNet parser for labels ----
import importlib.util
spec = importlib.util.spec_from_file_location(
    "test_skill_lib",
    str(_SKILL_LIB / "test_skill_lib.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

parses = [mod._parse_subtask_verbnet(t) for t in subtask_texts]
vn_classes = [p["verbnet_class"] for p in parses]
level2_keys = [p["level2_key"] for p in parses]

# ---- unique labels ----
unique_actions = sorted(set(action_texts))
unique_objects = sorted(set(object_texts))
unique_vn      = sorted(set(vn_classes))
print(f"Unique actions: {len(unique_actions)}, objects: {len(unique_objects)}, VN classes: {len(unique_vn)}")

# ---- helper ----
def safe_perplexity(n):
    if n <= 3: return 2
    return min(30, n - 1)

cmap20 = plt.get_cmap("tab20")
cmap10 = plt.get_cmap("tab10")

def color_map(labels, cmap=cmap20):
    uniq = sorted(set(labels))
    m = {l: i for i, l in enumerate(uniq)}
    return [cmap((m[l] % 20) / 20) for l in labels], m, uniq

# ---- compute t-SNE for three views ----
perp = safe_perplexity(N)
print(f"Running t-SNE (perplexity={perp}) on 4 views ...")

tsne_all     = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000).fit_transform(normalize(seg_all))
tsne_action  = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000).fit_transform(normalize(seg_action))
tsne_object  = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000).fit_transform(normalize(seg_object))
tsne_context = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000).fit_transform(normalize(seg_context))
print("t-SNE done.")

# ==================== FIGURE 1: 2x2 overview ====================
fig, axes = plt.subplots(2, 2, figsize=(24, 20))

# -- (0,0) Combined, colored by VN class --
ax = axes[0, 0]
colors_vn, vn_map, vn_uniq = color_map(vn_classes, cmap10)
ax.scatter(tsne_all[:, 0], tsne_all[:, 1], c=colors_vn, s=14, alpha=0.6, linewidths=0)
ax.set_title("Combined (88-dim)  —  colored by VerbNet class", fontsize=14)
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
patches_vn = [mpatches.Patch(color=cmap10((i % 10) / 10), label=f"{c} ({vn_classes.count(c)})") for i, c in enumerate(vn_uniq)]
ax.legend(handles=patches_vn, fontsize=8, loc="upper left", title="VerbNet class")

# -- (0,1) Action segment, colored by action verb --
ax = axes[0, 1]
colors_act, act_map, act_uniq = color_map(action_texts, cmap20)
ax.scatter(tsne_action[:, 0], tsne_action[:, 1], c=colors_act, s=14, alpha=0.6, linewidths=0)
ax.set_title("s_action (16-dim)  —  colored by verb phrase", fontsize=14)
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
patches_act = [mpatches.Patch(color=cmap20((i % 20) / 20), label=f"{a} ({action_texts.count(a)})") for i, a in enumerate(act_uniq)]
ax.legend(handles=patches_act, fontsize=7, loc="upper left", title="Verb phrase")

# -- (1,0) Object segment, colored by object noun --
ax = axes[1, 0]
# Too many objects for legend — show top-15, rest gray
obj_counter = Counter(object_texts)
top_objs = [o for o, _ in obj_counter.most_common(15)]
top_obj_map = {o: i for i, o in enumerate(top_objs)}
colors_obj = []
for o in object_texts:
    if o in top_obj_map:
        colors_obj.append(cmap20((top_obj_map[o] % 20) / 20))
    else:
        colors_obj.append((0.75, 0.75, 0.75, 0.4))
ax.scatter(tsne_object[:, 0], tsne_object[:, 1], c=colors_obj, s=14, alpha=0.6, linewidths=0)
ax.set_title("s_object (40-dim)  —  colored by object noun (top-15)", fontsize=14)
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
patches_obj = [mpatches.Patch(color=cmap20((i % 20) / 20), label=f"{o} ({obj_counter[o]})") for i, o in enumerate(top_objs)]
patches_obj.append(mpatches.Patch(color=(0.75, 0.75, 0.75), label=f"other ({sum(c for o, c in obj_counter.items() if o not in top_obj_map)})"))
ax.legend(handles=patches_obj, fontsize=7, loc="upper left", title="Object")

# -- (1,1) Context segment, colored by task_id --
ax = axes[1, 1]
n_tasks = max(task_ids) + 1
task_colors = [cmap20((tid % 20) / 20) for tid in task_ids]
ax.scatter(tsne_context[:, 0], tsne_context[:, 1], c=task_colors, s=14, alpha=0.5, linewidths=0)
ax.set_title(f"s_context (32-dim)  —  colored by task ID (0-{n_tasks-1})", fontsize=14)
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
# Only show a few task IDs in legend
step = max(1, n_tasks // 10)
patches_task = [mpatches.Patch(color=cmap20((t % 20) / 20), label=f"task {t}") for t in range(0, n_tasks, step)]
ax.legend(handles=patches_task, fontsize=7, loc="upper left", title="Task ID (sample)")

plt.suptitle(
    "Three-Segment Skill Embeddings — LIBERO-90 (371 subtasks)\n"
    f"s_action(16) | s_object(40) | s_context(32)  =  88 dim total",
    fontsize=18, y=1.01,
)
plt.tight_layout()
fig.savefig(OUT_DIR / "segmented_tsne_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[Fig 1] Overview saved → {OUT_DIR / 'segmented_tsne_overview.png'}")


# ==================== FIGURE 2: Combined, annotated with L2 cluster reps ====================
fig, ax = plt.subplots(figsize=(20, 15))

# Color by VerbNet L1 class
ax.scatter(tsne_all[:, 0], tsne_all[:, 1], c=colors_vn, s=18, alpha=0.55, linewidths=0, zorder=2)

# Annotate one representative per L2 cluster
l2_groups = {}
for i, lk in enumerate(level2_keys):
    l2_groups.setdefault(lk, []).append(i)

for lk, idxs in l2_groups.items():
    rep = idxs[0]
    cx, cy = tsne_all[rep]
    vn = vn_classes[rep]
    color = cmap10((vn_map[vn] % 10) / 10)
    ax.scatter(cx, cy, c=[color], s=220, marker="*", edgecolors="black", linewidths=0.5, zorder=5)
    short = lk if len(lk) <= 26 else lk[:24] + "…"
    ax.annotate(
        short, xy=(cx, cy), xytext=(5, 3), textcoords="offset points",
        fontsize=6.5, color="black",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.80, linewidth=0.6),
        zorder=6,
    )

ax.legend(handles=patches_vn, fontsize=10, loc="upper left",
          title=f"VerbNet class ({len(vn_uniq)})", title_fontsize=12)
ax.set_title(
    "Segmented Skill Library — Combined 88-dim\n"
    f"★ = L2 cluster representative (verb+noun)   {len(l2_groups)} clusters",
    fontsize=16,
)
ax.set_xlabel("t-SNE dim 1", fontsize=13)
ax.set_ylabel("t-SNE dim 2", fontsize=13)
plt.tight_layout()
fig.savefig(OUT_DIR / "segmented_tsne_annotated.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[Fig 2] Annotated saved → {OUT_DIR / 'segmented_tsne_annotated.png'}")


# ==================== FIGURE 3: Action-only t-SNE, big with annotations ====================
fig, ax = plt.subplots(figsize=(14, 10))
ax.scatter(tsne_action[:, 0], tsne_action[:, 1], c=colors_act, s=20, alpha=0.6, linewidths=0, zorder=2)

# Mark each unique action verb
seen = set()
for i, a in enumerate(action_texts):
    if a not in seen:
        seen.add(a)
        cx, cy = tsne_action[i]
        color = cmap20((act_map[a] % 20) / 20)
        ax.scatter(cx, cy, c=[color], s=200, marker="D", edgecolors="black", linewidths=0.6, zorder=5)
        ax.annotate(a, xy=(cx, cy), xytext=(6, 4), textcoords="offset points",
                    fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85),
                    zorder=6)

ax.set_title("s_action segment (16-dim) — Action Verb Clustering", fontsize=15)
ax.set_xlabel("t-SNE dim 1", fontsize=12)
ax.set_ylabel("t-SNE dim 2", fontsize=12)
ax.legend(handles=patches_act, fontsize=9, loc="upper left", title="Verb")
plt.tight_layout()
fig.savefig(OUT_DIR / "segmented_tsne_action.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[Fig 3] Action saved → {OUT_DIR / 'segmented_tsne_action.png'}")


# ==================== Text summary ====================
lines = []
lines.append(f"Three-Segment Skill Embeddings Summary")
lines.append(f"=" * 60)
lines.append(f"Total subtasks: {N}")
lines.append(f"Segment dims: action={seg_action.shape[1]}, object={seg_object.shape[1]}, context={seg_context.shape[1]}, total={seg_all.shape[1]}")
lines.append(f"Unique actions: {len(unique_actions)}")
lines.append(f"Unique objects: {len(unique_objects)}")
lines.append(f"VerbNet L1 classes: {len(unique_vn)}")
lines.append(f"VerbNet L2 clusters: {len(l2_groups)}")
lines.append("")

lines.append(f"Action distribution:")
for a in sorted(unique_actions):
    lines.append(f"  {a:20s}  {action_texts.count(a):4d}")
lines.append("")

lines.append(f"Top-20 object distribution:")
for o, cnt in obj_counter.most_common(20):
    lines.append(f"  {o:30s}  {cnt:4d}")
lines.append("")

lines.append(f"VerbNet L1 class distribution:")
vn_counter = Counter(vn_classes)
for c in unique_vn:
    lines.append(f"  {c:30s}  {vn_counter[c]:4d}")
lines.append("")

lines.append(f"{'='*60}")
lines.append(f"Level-2 clusters ({len(l2_groups)}):")
for lk in sorted(l2_groups.keys()):
    idxs = l2_groups[lk]
    lines.append(f"\n  {lk}  ({len(idxs)} subtasks)")
    for idx in idxs[:5]:
        lines.append(f"    - {subtask_texts[idx]}")
    if len(idxs) > 5:
        lines.append(f"    ... and {len(idxs)-5} more")

summary_text = "\n".join(lines)
(OUT_DIR / "segmented_summary.txt").write_text(summary_text, encoding="utf-8")
print(f"[Summary] {OUT_DIR / 'segmented_summary.txt'}")
print("ALL DONE")
