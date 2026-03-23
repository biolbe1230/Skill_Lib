"""
Hierarchical skill retriever for 3-layer multimodal skill library (V4).

Given a parsed subtask (verb phrase, object description) and a current-scene
visual encoding, retrieves the closest matching skill from a pre-built
hierarchical library in three steps:

  L1:  verb_phrase → VerbNet class  (O(1) lookup)
  L2:  SBERT(object_desc) → project → nearest L2 centroid in s_o space
  L3:  encode_env(scene) → project → nearest L3 centroid in s_c space

Returns the leaf-level (s_a, s_o, s_c) triple ready for cross-attention
injection into a VLA policy network.

Usage
-----
    retriever = HierarchicalSkillRetriever.load(
        library_dir="skill_lib_results_contrastive",
        sbert_model="all-MiniLM-L6-v2",
    )
    result = retriever.query(
        verb_phrase="pick up",
        object_text="small white ceramic bowl",
        s_env_raw=np.ndarray_of_shape_D,
    )
    # result["s_a"], result["s_o"], result["s_c"] — each numpy 1-D
    # result["path"] — e.g. "get-13.5.1.0.2"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize as sk_normalize


# ---------------------------------------------------------------------------
# Import parse logic from test_skill_lib (avoid circular dep by lazy import)
# ---------------------------------------------------------------------------

def _lazy_parse_verbnet(text: str) -> dict[str, str]:
    """Thin wrapper that lazily imports _parse_subtask_verbnet."""
    from Skill_Lib.test_skill_lib import _parse_subtask_verbnet
    return _parse_subtask_verbnet(text)


# ---------------------------------------------------------------------------
# Projector MLP (must match build_contrastive_skill_emb.Projector)
# ---------------------------------------------------------------------------

class _Projector(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, d_hidden: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, d_hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ---------------------------------------------------------------------------
# Hierarchical Skill Retriever
# ---------------------------------------------------------------------------

class HierarchicalSkillRetriever:
    """Three-layer hierarchical skill retriever.

    Attributes
    ----------
    library : dict
        Nested JSON tree loaded from ``hierarchical_3layer_library.json``.
    seg_action, seg_object, seg_context : np.ndarray
        (N, d_a/d_o/d_c) segment embeddings for all library members.
    proj_action, proj_object, proj_context : _Projector | None
        Learned projection MLPs.  If None, raw embeddings are used directly
        (for random-projection fallback compatibility).
    sbert : SentenceTransformer
        For encoding on-the-fly object descriptions / action phrases.
    """

    def __init__(
        self,
        library: dict,
        seg_action: np.ndarray,
        seg_object: np.ndarray,
        seg_context: np.ndarray,
        proj_action: Optional[_Projector],
        proj_object: Optional[_Projector],
        proj_context: Optional[_Projector],
        sbert_model: str = "all-MiniLM-L6-v2",
        meta: Optional[dict] = None,
    ):
        self.library = library
        self.seg_action = seg_action
        self.seg_object = seg_object
        self.seg_context = seg_context
        self.proj_action = proj_action
        self.proj_object = proj_object
        self.proj_context = proj_context
        self.meta = meta or {}

        # Lazy-load SBERT only when first needed
        self._sbert_model_name = sbert_model
        self._sbert = None

        # Pre-build L2/L3 centroid index for fast NN
        self._l2_centroids: dict[str, dict] = {}   # l1_class -> {ids: list, centroids: (K, d_o)}
        self._l3_centroids: dict[str, dict] = {}   # l2_id -> {ids: list, centroids: (K, d_c)}
        self._l3_data: dict[str, dict] = {}         # l3_id -> {s_a, s_o, s_c, representative}

        self._build_index()

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(self):
        """Pre-compute centroid arrays for fast nearest-neighbour lookup."""
        for l1_class, l1_info in self.library.items():
            l2_ids = []
            l2_cents = []
            for l2_id, l2_info in l1_info.get("l2_clusters", {}).items():
                l2_ids.append(l2_id)
                l2_cents.append(np.array(l2_info["centroid_s_o"], dtype=np.float32))

                l3_ids = []
                l3_cents = []
                for l3_id, l3_info in l2_info.get("l3_clusters", {}).items():
                    l3_ids.append(l3_id)
                    l3_cents.append(np.array(l3_info["centroid_s_c"], dtype=np.float32))

                    # Store representative leaf data
                    ri = l3_info["representative_idx"]
                    self._l3_data[l3_id] = {
                        "s_a": self.seg_action[ri].copy(),
                        "s_o": self.seg_object[ri].copy(),
                        "s_c": self.seg_context[ri].copy(),
                        "representative": l3_info["representative"],
                        "representative_idx": ri,
                    }

                if l3_cents:
                    self._l3_centroids[l2_id] = {
                        "ids": l3_ids,
                        "centroids": np.stack(l3_cents, axis=0),
                    }

            if l2_cents:
                self._l2_centroids[l1_class] = {
                    "ids": l2_ids,
                    "centroids": np.stack(l2_cents, axis=0),
                }

    # ------------------------------------------------------------------
    # SBERT lazy loading
    # ------------------------------------------------------------------

    @property
    def sbert(self):
        if self._sbert is None:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer(self._sbert_model_name, device="cpu")
        return self._sbert

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------

    def _project_object(self, text: str) -> np.ndarray:
        """Encode text with SBERT → normalise → project → (d_o,)."""
        raw = self.sbert.encode(
            [text], convert_to_numpy=True, normalize_embeddings=False
        ).astype(np.float32)
        raw_norm = sk_normalize(raw)
        if self.proj_object is not None:
            with torch.no_grad():
                return self.proj_object(torch.from_numpy(raw_norm)).numpy()[0]
        # Fallback: no learned projection — just return normalised raw
        return raw_norm[0]

    def _project_context(self, s_env_raw: np.ndarray) -> np.ndarray:
        """Normalise s_env → project → (d_c,)."""
        raw = s_env_raw.reshape(1, -1).astype(np.float32)
        raw_norm = sk_normalize(raw)
        if self.proj_context is not None:
            with torch.no_grad():
                return self.proj_context(torch.from_numpy(raw_norm)).numpy()[0]
        return raw_norm[0]

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        verb_phrase: str,
        object_text: str,
        s_env_raw: np.ndarray,
    ) -> dict:
        """Hierarchical three-step retrieval.

        Parameters
        ----------
        verb_phrase : str
            e.g. "pick up", "place", "open"
        object_text : str
            A visual description of the target object (from VLM or bare noun).
        s_env_raw : np.ndarray
            Raw environment embedding vector (before projection).

        Returns
        -------
        dict with keys:
            s_a, s_o, s_c   – numpy 1-D segment vectors
            path             – full 3-level cluster ID string
            representative   – representative subtask text from matched leaf
            l1_class, l2_id, l3_id  – hierarchy IDs
        """
        # L1: VerbNet class lookup
        parsed = _lazy_parse_verbnet(verb_phrase + " something")
        l1_class = parsed["verbnet_class"]

        l2_index = self._l2_centroids.get(l1_class)
        if l2_index is None:
            # Fallback: try exact match on any L1
            for candidate in self._l2_centroids:
                l1_class = candidate
                l2_index = self._l2_centroids[candidate]
                break
            if l2_index is None:
                return self._fallback_result()

        # L2: nearest centroid in s_object space
        q_obj = self._project_object(object_text)
        sims = l2_index["centroids"] @ q_obj
        best_l2_local = int(np.argmax(sims))
        l2_id = l2_index["ids"][best_l2_local]

        l3_index = self._l3_centroids.get(l2_id)
        if l3_index is None:
            return self._fallback_result()

        # L3: nearest centroid in s_context space
        q_ctx = self._project_context(s_env_raw)
        sims_ctx = l3_index["centroids"] @ q_ctx
        best_l3_local = int(np.argmax(sims_ctx))
        l3_id = l3_index["ids"][best_l3_local]

        leaf = self._l3_data[l3_id]
        return {
            "s_a": leaf["s_a"],
            "s_o": leaf["s_o"],
            "s_c": leaf["s_c"],
            "path": l3_id,
            "representative": leaf["representative"],
            "l1_class": l1_class,
            "l2_id": l2_id,
            "l3_id": l3_id,
        }

    def _fallback_result(self) -> dict:
        """Return zero vectors when hierarchy cannot resolve."""
        d_a = self.seg_action.shape[1] if len(self.seg_action) > 0 else 16
        d_o = self.seg_object.shape[1] if len(self.seg_object) > 0 else 40
        d_c = self.seg_context.shape[1] if len(self.seg_context) > 0 else 32
        return {
            "s_a": np.zeros(d_a, dtype=np.float32),
            "s_o": np.zeros(d_o, dtype=np.float32),
            "s_c": np.zeros(d_c, dtype=np.float32),
            "path": "unknown",
            "representative": "",
            "l1_class": "unknown",
            "l2_id": "unknown",
            "l3_id": "unknown",
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        library_dir: str | Path,
        sbert_model: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> "HierarchicalSkillRetriever":
        """Load a pre-built hierarchical library from disk.

        Expected files in *library_dir*:
          - hierarchical_3layer_library.json
          - seg_action.npy, seg_object.npy, seg_context.npy
          - (optional) proj_action.pt, proj_object.pt, proj_context.pt
          - (optional) contrastive_meta.json
        """
        d = Path(library_dir)

        library = json.loads(
            (d / "hierarchical_3layer_library.json").read_text("utf-8")
        )
        seg_a = np.load(d / "seg_action.npy")
        seg_o = np.load(d / "seg_object.npy")
        seg_c = np.load(d / "seg_context.npy")

        # Load meta if available
        meta_path = d / "contrastive_meta.json"
        meta = json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else {}

        d_hidden = meta.get("d_hidden", 128)

        # Load learned projectors if available
        def _load_proj(name: str, d_in: int, d_out: int):
            pt_path = d / f"proj_{name}.pt"
            if not pt_path.exists():
                return None
            proj = _Projector(d_in, d_out, d_hidden)
            proj.load_state_dict(torch.load(pt_path, map_location=device, weights_only=True))
            proj.eval()
            return proj

        # Infer input dims from meta or SBERT default
        from sentence_transformers import SentenceTransformer
        _tmp = SentenceTransformer(sbert_model, device="cpu")
        d_sbert = _tmp.get_sentence_embedding_dimension()
        del _tmp

        # Context input dim: infer from s_env if saved
        s_env_path = d / "s_env_embeddings.npy"
        d_env = int(np.load(s_env_path).shape[1]) if s_env_path.exists() else seg_c.shape[1]

        proj_a = _load_proj("action", d_sbert, seg_a.shape[1])
        proj_o = _load_proj("object", d_sbert, seg_o.shape[1])
        proj_c = _load_proj("context", d_env, seg_c.shape[1])

        return cls(
            library=library,
            seg_action=seg_a,
            seg_object=seg_o,
            seg_context=seg_c,
            proj_action=proj_a,
            proj_object=proj_o,
            proj_context=proj_c,
            sbert_model=sbert_model,
            meta=meta,
        )
