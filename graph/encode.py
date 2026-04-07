"""
graph/encode.py
===============
Feature encoding utilities: converts categorical/geometric face attributes
into the numeric tensor format expected by the GNN.

Also provides helpers for edge feature encoding when edge features are used
(currently stubbed; set ``graph.edge_feat_dim > 0`` in config to enable).
"""

from __future__ import annotations

import numpy as np
import torch
from utils.logger import logger

from graph.extract import SURF_TYPE_ENC, FACE_FEAT_DIM


# ---------------------------------------------------------------------------
# Face encoding
# ---------------------------------------------------------------------------

def encode_face_features(
    raw_features: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> torch.Tensor:
    """
    Normalise and convert face feature matrix to a float32 tensor.

    Parameters
    ----------
    raw_features : np.ndarray, shape (N, FACE_FEAT_DIM)
    mean, std    : optional precomputed stats for z-score normalisation

    Returns
    -------
    torch.FloatTensor, shape (N, FACE_FEAT_DIM)
    """
    feat = raw_features.astype(np.float32)
    if mean is not None and std is not None:
        feat = (feat - mean) / (std + 1e-8)
    return torch.from_numpy(feat)


# ---------------------------------------------------------------------------
# Edge encoding
# ---------------------------------------------------------------------------

CONVEXITY_ENC: dict[str, int] = {
    "convex":   0,
    "concave":  1,
    "smooth":   2,
    "unknown":  3,
}


def encode_edge_features(edges: list[dict]) -> torch.Tensor | None:
    """
    Encode a list of edge dicts into a float32 edge feature tensor.

    Currently encodes only convexity as a one-hot of size 4.
    Returns None if edge feature dim is 0 (disabled in config).

    Extension point: add curvature, length, dihedral angle here.
    """
    if not edges:
        return None

    edge_feats = np.zeros((len(edges), 4), dtype=np.float32)
    for i, edge in enumerate(edges):
        conv_idx = CONVEXITY_ENC.get(
            str(edge.get("convexity", "unknown")).lower(), 3
        )
        edge_feats[i, conv_idx] = 1.0
    return torch.from_numpy(edge_feats)


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

def encode_labels(feature_predictions: list[dict], num_classes: int) -> torch.Tensor:
    """
    Convert a list of feature-prediction dicts into a LongTensor of class ids.

    Parameters
    ----------
    feature_predictions : list of dicts with 'face_id' and 'feature_type'
    num_classes         : int

    Returns
    -------
    torch.LongTensor, shape (N,)
    """
    # Build sorted face_id → class_id mapping
    FEATURE_CLASS_MAP: dict[str, int] = {
        "Chamfer":    0,
        "Through_Hole": 1,
        "Triangular_Passage": 2,
        "Rectangular_Passage": 3,
        "6Sides_Passage": 4,
        "Triangular_Through_Slot": 5,
        "Rectangular_Through_Slot": 6,
        "Circular_Through_Slot": 7,
        "Rectangular_Through_Step": 8,
        "2Sides_Through_Step": 9,
        "Slanted_Through_Step": 10,
        "Oring": 11,
        "Blind_Hole": 12,
        "Triangular_Pocket": 13,
        "Rectangular_Pocket": 14,
        "6Sides_Pocket": 15,
        "Circular_End_Pocket": 16,
        "Rectangular_Blind_Step": 17,
        "2Sides_Blind_Step": 18,
        "Slanted_Blind_Step": 19,
        "Rectangular_Blind_Slot": 20,
        "Vertical_Circular_End_Blind_Slot": 21,
        "Horizontal_Circular_End_Blind_Slot": 22,
        "Triangular_Blind_Step": 23,
        "Stock": 24,
    }

    sorted_preds = sorted(feature_predictions, key=lambda x: x["face_id"])
    labels = [
        FEATURE_CLASS_MAP.get(p.get("feature_type", "Stock"), 24)
        for p in sorted_preds
    ]
    return torch.tensor(labels, dtype=torch.long)
