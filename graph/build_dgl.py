"""
graph/build_dgl.py
==================
Build DGL heterogeneous graphs from two possible sources:

1. **H5 file** (training) – delegates to ``graph.mfcad_h5_dataset``.
2. **Neo4j** (inference on new parts) – queries the KG adjacency and
   wraps it in a DGL heterograph using topology features from
   ``graph.extract``.

Entry points
------------
``build_from_h5(h5_path, key)``
    Single-sample graph from an H5 group.

``build_from_neo4j(model_id, client, topology)``
    Graph built from Neo4j adjacency + OCC-extracted features.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import torch
from loguru import logger

from graph.extract import extract_face_features, FACE_FEAT_DIM
from graph.mfcad_h5_dataset import MFCAD_H5_Dataset

try:
    import dgl
    _DGL_AVAILABLE = True
except ImportError:
    _DGL_AVAILABLE = False


# ---------------------------------------------------------------------------
# H5-based single-sample builder
# ---------------------------------------------------------------------------

def build_from_h5(h5_path: str, key: str) -> tuple:
    """
    Load a single graph from an H5 file by part key.

    Returns
    -------
    (g, labels) – same format as ``MFCAD_H5_Dataset.__getitem__``
    """
    import h5py

    with h5py.File(h5_path, "r") as fh:
        grp = fh[key]
        V1      = np.array(grp["V_1"],        dtype=np.float32)
        Y       = np.array(grp["Y"],           dtype=np.int64)
        A1_idx  = np.array(grp["A_1_idx"],    dtype=np.int64)
        A1_vals = np.array(grp["A_1_values"], dtype=np.float32)
        E1      = np.array(grp["E_1"],         dtype=np.int32)
        E2      = np.array(grp["E_2"],         dtype=np.int32)
        E3      = np.array(grp["E_3"],         dtype=np.int32)

    g = MFCAD_H5_Dataset._build_heterograph(A1_idx, A1_vals, E1, E2, E3, V1)
    labels = torch.from_numpy(Y)
    logger.debug(f"H5 graph built: key={key}  n_faces={V1.shape[0]}")
    return g, labels


# ---------------------------------------------------------------------------
# Neo4j-based builder (for inference on new STEP files)
# ---------------------------------------------------------------------------

def build_from_neo4j(
    model_id: str,
    client: Any,   # Neo4jClient
    topology: dict[str, Any],
    cfg: dict[str, Any],
) -> "dgl.DGLHeteroGraph":
    """
    Build a DGL heterograph for an already-ingested STEP model using:
    - Face features from OCC topology extraction (``graph.extract``).
    - Adjacency from Neo4j ADJACENT_TO relationships.

    Parameters
    ----------
    model_id  : str  (SHA-16 from ingest pipeline)
    client    : Neo4jClient
    topology  : dict output of ``cad.parser.load_step``
    cfg       : full pipeline config dict

    Returns
    -------
    dgl.DGLHeteroGraph with node type "face" and edge types
    "convex", "concave", "smooth", "adj"
    """
    if not _DGL_AVAILABLE:
        raise ImportError("DGL is required for graph construction.")

    # ── Node features ─────────────────────────────────────────────────────────
    feat_mat = extract_face_features(topology)   # (N, 10)
    N = feat_mat.shape[0]
    print(f"extract_face_features shape = {feat_mat.shape}")
    # ── Adjacency from Neo4j ──────────────────────────────────────────────────
    adjacency = _query_adjacency(model_id, client)

    # ── Partition edges by convexity ──────────────────────────────────────────
    src_conv, dst_conv = [], []
    src_conc, dst_conc = [], []
    src_smooth, dst_smooth = [], []
    src_adj, dst_adj = [], []

    for (fa_id, fb_id, conv) in adjacency:
        if conv == "convex":
            src_conv.append(fa_id); dst_conv.append(fb_id)
        elif conv == "concave":
            src_conc.append(fa_id); dst_conc.append(fb_id)
        elif conv == "smooth":
            src_smooth.append(fa_id); dst_smooth.append(fb_id)
        else:
            src_adj.append(fa_id); dst_adj.append(fb_id)

    data: dict[tuple, tuple] = {}
    if src_conv:
        data[("face", "convex",  "face")] = (
            torch.tensor(src_conv), torch.tensor(dst_conv))
    if src_conc:
        data[("face", "concave", "face")] = (
            torch.tensor(src_conc), torch.tensor(dst_conc))
    if src_smooth:
        data[("face", "smooth",  "face")] = (
            torch.tensor(src_smooth), torch.tensor(dst_smooth))
    if src_adj:
        data[("face", "adj",     "face")] = (
            torch.tensor(src_adj),   torch.tensor(dst_adj))

    # Guard: no edges → self-loops
    if not data:
        logger.warning(
            "No adjacency found in Neo4j for model_id=%s; "
            "using self-loops.", model_id
        )
        data[("face", "adj", "face")] = (
            torch.arange(N), torch.arange(N))

    g = dgl.heterograph(data, num_nodes_dict={"face": N})
    g.nodes["face"].data["feat"] = torch.from_numpy(feat_mat)
    logger.info(
        f"Neo4j graph built: model_id={model_id}  "
        f"N={N}  edge_types={list(data.keys())}"
    )
    return g


def _query_adjacency(
    model_id: str,
    client: Any,
) -> list[tuple[int, int, str]]:
    """Pull ADJACENT_TO edges from Neo4j for the given model."""
    cypher = """
    MATCH (fa:Face {model_id: $model_id})-[r:ADJACENT_TO]->(fb:Face)
    WHERE fa.model_id = $model_id
    RETURN fa.face_id AS src,
           fb.face_id AS dst,
           r.convexity AS conv
    """
    results: list[tuple[int, int, str]] = []
    with client.session() as sess:
        for rec in sess.run(cypher, model_id=model_id):
            results.append((
                int(rec["src"]),
                int(rec["dst"]),
                str(rec["conv"]),
            ))
    logger.debug(f"Adjacency from Neo4j: {len(results)} pairs.")
    return results
