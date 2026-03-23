

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Resolve _parse_subtask_verbnet from shared utility
# ---------------------------------------------------------------------------
import importlib.util as _ilu

_VERBNET_UTILS_PATH = Path(__file__).resolve().parent / "verbnet_utils.py"
_vn_spec = _ilu.spec_from_file_location("_verbnet_utils", _VERBNET_UTILS_PATH)
_vn_mod  = _ilu.module_from_spec(_vn_spec)
_vn_spec.loader.exec_module(_vn_mod)
_parse_subtask_verbnet = _vn_mod._parse_subtask_verbnet


# ---------------------------------------------------------------------------
# NT-Xent (InfoNCE) loss
# ---------------------------------------------------------------------------

def nt_xent_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07):
    """Supervised NT-Xent: positives share the same label.

    Args:
        z: (N, D) L2-normalised embeddings.
        labels: (N,) integer class labels.
        temperature: softmax temperature.

    Returns:
        Scalar loss.
    """
    device = z.device
    N = z.shape[0]
    sim = z @ z.T / temperature  # (N, N)

    # Mask: positive pairs share the same label (excluding self)
    label_mat = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N) bool
    self_mask = ~torch.eye(N, dtype=torch.bool, device=device)
    pos_mask = label_mat & self_mask

    # If a sample has no positive partner, skip it
    has_pos = pos_mask.any(dim=1)
    if not has_pos.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Log-sum-exp over all negatives+positives (denominator), then subtract
    # positive similarity (numerator).  Use logsumexp for numerical stability.
    logits_mask = self_mask.float()
    exp_logits = torch.exp(sim) * logits_mask
    log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))

    # Mean of log-prob over positive pairs per anchor
    log_prob = sim - log_sum_exp  # (N, N)
    # Average over positives for each anchor
    mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1) / pos_mask.float().sum(dim=1).clamp_min(1.0)
    loss = -mean_log_prob_pos[has_pos].mean()
    return loss


# ---------------------------------------------------------------------------
# Projector MLP
# ---------------------------------------------------------------------------

class Projector(nn.Module):
    """Small 2-layer MLP: d_in -> hidden -> d_out."""

    def __init__(self, d_in: int, d_out: int, d_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SegmentDataset(Dataset):
    """Yields (raw_vec, label) for one segment."""

    def __init__(self, vectors: np.ndarray, labels: np.ndarray):
        self.vectors = torch.from_numpy(vectors.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        return self.vectors[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Training one segment projector
# ---------------------------------------------------------------------------

def train_projector(
    name: str,
    raw_vecs: np.ndarray,
    labels: np.ndarray,
    d_out: int,
    d_hidden: int = 128,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 256,
    temperature: float = 0.07,
    device: str = "cpu",
) -> Projector:
    """Train a Projector MLP with NT-Xent on (raw_vecs, labels)."""

    from sklearn.preprocessing import normalize as sk_normalize

    # L2-normalise inputs (same as downstream usage)
    raw_norm = sk_normalize(raw_vecs).astype(np.float32)

    ds = SegmentDataset(raw_norm, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    d_in = raw_vecs.shape[1]
    proj = Projector(d_in, d_out, d_hidden).to(device)
    optimizer = torch.optim.Adam(proj.parameters(), lr=lr)

    proj.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for vecs_batch, labels_batch in loader:
            vecs_batch = vecs_batch.to(device)
            labels_batch = labels_batch.to(device)
            z = proj(vecs_batch)
            loss = nt_xent_loss(z, labels_batch, temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  [{name}] epoch {epoch+1:4d}/{epochs}  loss={total_loss/max(n_batches,1):.4f}")

    proj.eval()
    return proj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Contrastive pre-training for three-segment skill projections."
    )
    parser.add_argument("--source_dir", type=str, default="skill_lib_results_full",
                        help="Directory with subtask_texts.json + s_env_embeddings.npy")
    parser.add_argument("--output_dir", type=str, default="skill_lib_results_contrastive",
                        help="Directory to write learned projection weights")
    parser.add_argument("--sbert_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--d_action", type=int, default=16)
    parser.add_argument("--d_object", type=int, default=40)
    parser.add_argument("--d_context", type=int, default=32)
    parser.add_argument("--d_hidden", type=int, default=128,
                        help="Hidden dim of projector MLP")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mode", type=str, default="sbert",
                        choices=["sbert", "clip_film"],
                        help="Training mode: "
                             "sbert = train MLP projectors on top of SBERT (original); "
                             "clip_film = fine-tune FiLMResNet + proj heads end-to-end via CLIP")
    parser.add_argument("--clip_model", type=str,
                        default="openai/clip-vit-base-patch32")
    parser.add_argument("--task_artifacts_dir", type=str, default=None,
                        help="[clip_film] Path to task_artifacts/ containing per-task images")
    parser.add_argument("--film_lr", type=float, default=3e-4,
                        help="[clip_film] Learning rate for FiLM + proj heads (default: 3e-4)")
    args = parser.parse_args()

    source = Path(args.source_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # ---- Load pre-computed data ----
    texts_meta = json.loads((source / "subtask_texts.json").read_text("utf-8"))
    subtask_texts: list[str] = texts_meta["texts"]
    task_ids: list[int] = texts_meta["task_ids"]
    N = len(subtask_texts)
    print(f"[Info] Loaded {N} subtasks from {source / 'subtask_texts.json'}")

    s_env_task = np.load(source / "s_env_embeddings.npy").astype(np.float32)
    print(f"[Info] s_env_task shape: {s_env_task.shape}")

    # ---- Parse subtasks to get labels ----
    verbnet_classes: list[str] = []
    head_nouns: list[str] = []
    action_texts: list[str] = []
    object_texts: list[str] = []
    for text in subtask_texts:
        parsed = _parse_subtask_verbnet(text)
        verbnet_classes.append(parsed["verbnet_class"])
        head_nouns.append(parsed["head_noun"])
        action_texts.append(parsed["verb_phrase"])
        object_texts.append(parsed["object_phrase"] or "object")

    # Build integer labels for each segment
    # s_a labels: VerbNet class
    vn_uniq = sorted(set(verbnet_classes))
    vn_to_id = {v: i for i, v in enumerate(vn_uniq)}
    action_labels = np.array([vn_to_id[v] for v in verbnet_classes], dtype=np.int64)

    # s_o labels: head noun
    noun_uniq = sorted(set(head_nouns))
    noun_to_id = {n: i for i, n in enumerate(noun_uniq)}
    object_labels = np.array([noun_to_id[n] for n in head_nouns], dtype=np.int64)

    # s_c labels: task id
    context_labels = np.array(task_ids, dtype=np.int64)

    # ---- clip_film mode: delegate to separate function and return ----
    if args.mode == "clip_film":
        train_clip_film(
            args, subtask_texts, task_ids, action_texts, object_texts,
            action_labels, object_labels, context_labels,
            vn_uniq, noun_uniq, output,
        )
        return

    # ---- Encode raw vectors with SBERT ----
    from sentence_transformers import SentenceTransformer

    print(f"[Info] Loading SBERT: {args.sbert_model}")
    sbert = SentenceTransformer(args.sbert_model, device="cpu")

    print(f"[Info] Encoding {N} action phrases …")
    action_raw = sbert.encode(
        action_texts, batch_size=128, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=False,
    ).astype(np.float32)

    print(f"[Info] Encoding {N} object phrases …")
    object_raw = sbert.encode(
        object_texts, batch_size=128, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=False,
    ).astype(np.float32)

    # Per-subtask s_env lookup
    tid_arr = np.array(task_ids, dtype=np.int64)
    context_raw = s_env_task[tid_arr]  # (N, D_env)

    # ---- Train projections ----
    print(f"\n=== Training action projector [{action_raw.shape[1]} -> {args.d_action}] ===")
    print(f"    Labels: {len(vn_uniq)} VerbNet classes")
    proj_action = train_projector(
        "action", action_raw, action_labels,
        d_out=args.d_action, d_hidden=args.d_hidden,
        epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, temperature=args.temperature,
        device=args.device,
    )

    print(f"\n=== Training object projector [{object_raw.shape[1]} -> {args.d_object}] ===")
    print(f"    Labels: {len(noun_uniq)} distinct head nouns")
    proj_object = train_projector(
        "object", object_raw, object_labels,
        d_out=args.d_object, d_hidden=args.d_hidden,
        epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, temperature=args.temperature,
        device=args.device,
    )

    print(f"\n=== Training context projector [{context_raw.shape[1]} -> {args.d_context}] ===")
    print(f"    Labels: {len(set(task_ids))} tasks")
    proj_context = train_projector(
        "context", context_raw, context_labels,
        d_out=args.d_context, d_hidden=args.d_hidden,
        epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, temperature=args.temperature,
        device=args.device,
    )

    # ---- Save projectors ----
    torch.save(proj_action.state_dict(), output / "proj_action.pt")
    torch.save(proj_object.state_dict(), output / "proj_object.pt")
    torch.save(proj_context.state_dict(), output / "proj_context.pt")

    # ---- Apply projections and save final embeddings ----
    from sklearn.preprocessing import normalize as sk_normalize

    with torch.no_grad():
        s_a = proj_action(torch.from_numpy(sk_normalize(action_raw))).numpy()
        s_o = proj_object(torch.from_numpy(sk_normalize(object_raw))).numpy()
        s_c = proj_context(torch.from_numpy(sk_normalize(context_raw))).numpy()

    np.save(output / "seg_action.npy", s_a)
    np.save(output / "seg_object.npy", s_o)
    np.save(output / "seg_context.npy", s_c)
    np.save(output / "seg_embeddings.npy", np.concatenate([s_a, s_o, s_c], axis=1))

    # Save metadata
    meta = {
        "segment_dims": {
            "action": args.d_action,
            "object": args.d_object,
            "context": args.d_context,
            "total": args.d_action + args.d_object + args.d_context,
        },
        "sbert_model": args.sbert_model,
        "d_hidden": args.d_hidden,
        "epochs": args.epochs,
        "temperature": args.temperature,
        "n_action_classes": len(vn_uniq),
        "n_object_classes": len(noun_uniq),
        "n_context_classes": len(set(task_ids)),
        "action_class_names": vn_uniq,
        "object_class_names": noun_uniq,
        "action_texts": action_texts,
        "object_texts": object_texts,
        "verbnet_classes": verbnet_classes,
        "head_nouns": head_nouns,
    }
    (output / "contrastive_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Copy source metadata
    (output / "subtask_texts.json").write_text(
        (source / "subtask_texts.json").read_text("utf-8"), encoding="utf-8"
    )

    total = args.d_action + args.d_object + args.d_context
    print(f"\n[Done] Contrastive projections saved to {output}/")
    print(f"       Total embedding dim: {total} "
          f"(action={args.d_action} + object={args.d_object} + context={args.d_context})")


# ---------------------------------------------------------------------------
# clip_film end-to-end training
# ---------------------------------------------------------------------------

def train_clip_film(args, subtask_texts, task_ids, action_texts, object_texts,
                    action_labels, object_labels, context_labels,
                    vn_uniq, noun_uniq, output):
    """Fine-tune CLIPFiLMSkillEncoder (FiLM generator + projection heads)."""
    import importlib.util as _ilu2

    # Load film_encoder via importlib to avoid site-packages conflict
    _film_path = Path(__file__).resolve().parent / "models" / "modules" / "film_encoder.py"
    _film_spec = _ilu2.spec_from_file_location("_film_encoder", _film_path)
    _film_mod  = _ilu2.module_from_spec(_film_spec)
    _film_spec.loader.exec_module(_film_mod)
    CLIPFiLMSkillEncoder = _film_mod.CLIPFiLMSkillEncoder
    film_prep = _film_mod.preprocess_images

    def load_task_artifacts(ta_dir, tid):
        """Load per-task images (inline to avoid importing test_skill_lib)."""
        from PIL import Image as _PILImage
        td = ta_dir / f"task_{tid:03d}"
        agent = np.array(_PILImage.open(td / "agentview.png").convert("RGB"), dtype=np.uint8)
        return agent, None, [], ""

    ta_dir = Path(args.task_artifacts_dir) if args.task_artifacts_dir else Path(args.source_dir) / "task_artifacts"

    print(f"\n=== Mode: clip_film — fine-tuning FiLMResNet + projection heads ===")
    print(f"    CLIP model: {args.clip_model}")
    print(f"    task_artifacts: {ta_dir}")

    enc = CLIPFiLMSkillEncoder(
        clip_model_name=args.clip_model,
        d_action=args.d_action, d_object=args.d_object, d_context=args.d_context,
        pretrained_resnet=True, freeze_clip=True,
    ).to(args.device)
    enc.train()

    # Only train: FiLM generator params + projection heads
    trainable = (
        list(enc.film_resnet.film_gen.parameters())
        + list(enc.proj_a.parameters())
        + list(enc.proj_o.parameters())
        + list(enc.proj_c.parameters())
    )
    optimizer = torch.optim.Adam(trainable, lr=args.film_lr)

    # Preload one image per task_id
    task_img_cache = {}
    for tid in sorted(set(task_ids)):
        try:
            m, _, _, _ = load_task_artifacts(ta_dir, tid)
            task_img_cache[tid] = m
        except Exception:
            task_img_cache[tid] = np.zeros((256, 256, 3), dtype=np.uint8)

    N = len(subtask_texts)
    idx = np.arange(N)
    bs = args.batch_size

    print(f"    Training {args.epochs} epochs, batch={bs}, lr={args.film_lr}")

    for epoch in range(1, args.epochs + 1):
        np.random.shuffle(idx)
        total_loss = 0.0
        n_batches  = 0
        for i in range(0, N, bs):
            batch = idx[i:i + bs]
            b_a   = [action_texts[j] for j in batch]
            b_o   = [object_texts[j] for j in batch]
            b_f   = [subtask_texts[j] for j in batch]
            b_img = [task_img_cache[task_ids[j]] for j in batch]
            b_la  = action_labels[batch]
            b_lo  = object_labels[batch]
            b_lc  = context_labels[batch]

            imgs  = film_prep(b_img).to(args.device)
            out   = enc(b_a, b_o, b_f, imgs)

            la_t  = torch.from_numpy(b_la).to(args.device)
            lo_t  = torch.from_numpy(b_lo).to(args.device)
            lc_t  = torch.from_numpy(b_lc).to(args.device)

            loss = (
                nt_xent_loss(out["s_a"], la_t, args.temperature)
                + nt_xent_loss(out["s_o"], lo_t, args.temperature)
                + nt_xent_loss(out["s_c"], lc_t, args.temperature)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        if epoch % 50 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{args.epochs}  "
                  f"loss={total_loss / max(n_batches, 1):.4f}")

    # Save full encoder state dict
    enc.eval()
    torch.save(enc.state_dict(), output / "clip_film_encoder.pt")
    print(f"[clip_film] Saved encoder -> {output / 'clip_film_encoder.pt'}")

    # Apply trained encoder to produce final segment arrays
    all_a, all_o, all_c = [], [], []
    with torch.no_grad():
        for i in range(0, N, bs):
            b = list(range(i, min(i + bs, N)))
            out = enc(
                [action_texts[j] for j in b],
                [object_texts[j] for j in b],
                [subtask_texts[j] for j in b],
                film_prep([task_img_cache[task_ids[j]] for j in b]).to(args.device),
            )
            all_a.append(out["s_a"].cpu().numpy())
            all_o.append(out["s_o"].cpu().numpy())
            all_c.append(out["s_c"].cpu().numpy())

    sa = np.concatenate(all_a).astype(np.float32)
    so = np.concatenate(all_o).astype(np.float32)
    sc = np.concatenate(all_c).astype(np.float32)
    np.save(output / "seg_action.npy",    sa)
    np.save(output / "seg_object.npy",    so)
    np.save(output / "seg_context.npy",   sc)
    np.save(output / "seg_embeddings.npy", np.concatenate([sa, so, sc], axis=1))

    # Metadata
    meta = {
        "mode": "clip_film",
        "clip_model": args.clip_model,
        "segment_dims": {
            "action": args.d_action, "object": args.d_object,
            "context": args.d_context,
            "total": args.d_action + args.d_object + args.d_context,
        },
        "epochs": args.epochs,
        "film_lr": args.film_lr,
        "temperature": args.temperature,
        "n_action_classes": len(vn_uniq),
        "n_object_classes": len(noun_uniq),
        "n_context_classes": len(set(task_ids)),
    }
    (output / "contrastive_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "subtask_texts.json").write_text(
        (Path(args.source_dir) / "subtask_texts.json").read_text("utf-8"),
        encoding="utf-8",
    )

    total_dim = args.d_action + args.d_object + args.d_context
    print(f"\n[Done] clip_film contrastive training saved to {output}/")
    print(f"       Total embedding dim: {total_dim}")


if __name__ == "__main__":
    main()
