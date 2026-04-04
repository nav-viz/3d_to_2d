"""
reasoning/view_selector.py
==========================
Feature-confidence driven view selection and adaptive routing.

Given a list of per-face GNN predictions (with confidences), this module:
1. Scores candidate views based on which feature types benefit most from
   each projection direction.
2. Applies complexity-based adaptive routing to limit the number of views.
3. Returns an ordered list of view names with associated metadata.

View scoring logic
------------------
Feature → view importance weights (heuristic, extendable):
  Hole      → Top view (circular outline), Front/Right for depth
  Pocket    → Top, Isometric
  Boss      → Front, Right
  Slot      → Front, Right (length visible)
  Chamfer   → Isometric (angle visible)
  Fillet    → Isometric
  Flat_Face → Front (dominant)

Complexity routing
------------------
  simple  (≤ simple_max_faces)  → max 3 views
  medium  (≤ medium_max_faces)  → max 5 views
  complex (> medium_max_faces)  → max 6 views (all candidates)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from loguru import logger


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ViewSpec:
    """Specification for a single drawing view."""
    name: str
    direction: tuple[float, float, float]  # unit normal of projection plane
    priority: int                           # 1 = highest
    features_visible: list[str]            = field(default_factory=list)
    confidence: float                       = 1.0


# ---------------------------------------------------------------------------
# View library
# ---------------------------------------------------------------------------

VIEW_LIBRARY: dict[str, ViewSpec] = {
    "front":     ViewSpec("front",     (0, -1, 0), priority=1),
    "top":       ViewSpec("top",       (0, 0, 1),  priority=2),
    "right":     ViewSpec("right",     (1, 0, 0),  priority=3),
    "bottom":    ViewSpec("bottom",    (0, 0, -1), priority=4),
    "left":      ViewSpec("left",      (-1, 0, 0), priority=5),
    "isometric": ViewSpec("isometric", (1, 1, 1),  priority=6),
}

# Feature → views that best expose it (in priority order)
FEATURE_VIEW_AFFINITY: dict[str, list[str]] = {
    "Through_Hole":      ["top", "front", "right"],
    "Blind_Hole":        ["top", "front"],
    "Rectangular_Pocket":["top", "isometric"],
    "Circular_End_Pocket":["top", "isometric"],
    "6Sides_Pocket":     ["top", "isometric"],
    "Triangular_Pocket": ["top", "isometric"],
    "Rectangular_Through_Slot": ["front", "right"],
    "Rectangular_Blind_Slot":   ["front", "right"],
    "Boss":              ["front", "right"],
    "Chamfer":           ["isometric", "front"],
    "Fillet":            ["isometric"],
    "Rectangular_Passage":["front", "top"],
    "Rectangular_Through_Step":["front", "right"],
    "Stock":             ["front", "top", "right"],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_views(
    predictions: list[dict[str, Any]],
    cfg: dict[str, Any],
    num_faces: int,
) -> list[ViewSpec]:
    """
    Select the optimal set of drawing views.

    Parameters
    ----------
    predictions : list of inference prediction dicts
                  (face_id, feature_type, confidence)
    cfg         : full pipeline config
    num_faces   : total face count of the part

    Returns
    -------
    Ordered list of ``ViewSpec`` objects (highest priority first).
    """
    vc = cfg["view_selection"]
    max_views    = vc.get("max_views", 6)
    min_conf     = vc.get("min_confidence", 0.5)
    simple_max   = vc.get("simple_max_faces", 50)
    medium_max   = vc.get("medium_max_faces", 300)
    candidates   = vc.get("view_candidates", list(VIEW_LIBRARY.keys()))

    # ── Complexity routing ────────────────────────────────────────────────────
    if num_faces <= simple_max:
        complexity = "simple";  limit = min(max_views, 3)
    elif num_faces <= medium_max:
        complexity = "medium";  limit = min(max_views, 5)
    else:
        complexity = "complex"; limit = max_views

    logger.info(
        f"View selection: num_faces={num_faces} → complexity={complexity} "
        f"limit={limit}"
    )

    # ── Score views ───────────────────────────────────────────────────────────
    view_scores: dict[str, float] = {v: 0.0 for v in candidates}

    for pred in predictions:
        ftype = pred.get("feature_type", "Stock")
        conf  = float(pred.get("confidence", 0.0))
        if conf < min_conf:
            continue
        for view_name in FEATURE_VIEW_AFFINITY.get(ftype, ["front"]):
            if view_name in view_scores:
                # Higher priority (lower rank) gets more weight
                rank = VIEW_LIBRARY[view_name].priority
                view_scores[view_name] += conf / rank

    # Always include the primary orthographic views
    for base_view in ["front", "top", "right"]:
        if base_view in view_scores:
            view_scores[base_view] += 2.0  # baseline priority bias

    # ── Sort and limit ────────────────────────────────────────────────────────
    sorted_views = sorted(
        view_scores.items(), key=lambda kv: kv[1], reverse=True
    )[:limit]

    selected: list[ViewSpec] = []
    for name, score in sorted_views:
        spec = VIEW_LIBRARY[name]
        spec.confidence = score
        spec.features_visible = _features_for_view(name, predictions, min_conf)
        selected.append(spec)

    logger.info(
        f"Selected views ({len(selected)}): "
        + ", ".join(f"{v.name}(score={v.confidence:.2f})" for v in selected)
    )
    return selected


def _features_for_view(
    view_name: str,
    predictions: list[dict],
    min_conf: float,
) -> list[str]:
    """Return feature types visible in a given view."""
    visible = set()
    for pred in predictions:
        ftype = pred.get("feature_type", "Stock")
        conf  = float(pred.get("confidence", 0.0))
        if conf >= min_conf and view_name in FEATURE_VIEW_AFFINITY.get(ftype, []):
            visible.add(ftype)
    return list(visible)
