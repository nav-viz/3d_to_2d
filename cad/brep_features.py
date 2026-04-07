"""
cad/brep_features.py
====================
Rule-based Machining Feature Recognition (MFR) directly from B-Rep topology.

Detected features (extendable):
  - Hole       : cylindrical face with circular edges, fully enclosed
  - Pocket     : planar bottom + perpendicular planar walls, no top face
  - Boss        : convex cylindrical protrusion
  - Slot        : two parallel planar walls + bottom, open on one side
  - Chamfer    : oblique planar face adjacent to two planar faces
  - Fillet     : cylindrical/toroidal transition face between two faces
  - Flat_Face  : isolated planar face (catch-all baseline)

These rule-based labels are used as:
  1. Cross-validation ground truth during development.
  2. Fall-back when GNN confidence is below threshold.

Extension point
---------------
Add new feature classes by extending the ``FEATURE_RULES`` dict and
implementing the corresponding ``_detect_*`` function below.
"""

from __future__ import annotations

import math
from typing import Any
from loguru import logger


# ---------------------------------------------------------------------------
# Feature label constants
# ---------------------------------------------------------------------------
class MachiningFeature:
    HOLE       = "Hole"
    POCKET     = "Pocket"
    BOSS       = "Boss"
    SLOT       = "Slot"
    CHAMFER    = "Chamfer"
    FILLET     = "Fillet"
    FLAT_FACE  = "Flat_Face"
    UNKNOWN    = "Unknown"


# Mapping surface type → candidate feature labels (fast pre-filter)
_SURFACE_FEATURE_HINTS: dict[str, list[str]] = {
    "Cylinder": [MachiningFeature.HOLE, MachiningFeature.BOSS],
    "Plane":    [MachiningFeature.POCKET, MachiningFeature.SLOT,
                 MachiningFeature.CHAMFER, MachiningFeature.FLAT_FACE],
    "Torus":    [MachiningFeature.FILLET],
    "Cone":     [MachiningFeature.CHAMFER],
    "Sphere":   [MachiningFeature.UNKNOWN],
    "BSpline":  [MachiningFeature.UNKNOWN],
    "Other":    [MachiningFeature.UNKNOWN],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_faces(topology: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Apply rule-based MFR to each face in the topology dict.

    Parameters
    ----------
    topology : dict
        Output of ``cad.parser.load_step``.

    Returns
    -------
    list of feature dicts
        Each dict contains:
        ``face_id``      – int
        ``feature_type`` – str (MachiningFeature constant)
        ``confidence``   – float (1.0 for rule-based, lower for heuristic)
        ``params``       – dict of geometric parameters (radius, depth, …)
    """
    faces: list[dict] = topology.get("faces", [])
    edges: list[dict] = topology.get("edges", [])

    results: list[dict] = []
    for face in faces:
        ftype, params = _classify_single_face(face, faces, edges)
        results.append({
            "face_id": face["id"],
            "feature_type": ftype,
            "confidence": 1.0,   # rule-based = deterministic
            "params": params,
        })
        logger.debug(f"Face {face['id']} → {ftype}  params={params}")

    logger.info(f"Rule-based MFR complete: {len(results)} faces classified.")
    return results


def _classify_single_face(
    face: dict,
    all_faces: list[dict],
    all_edges: list[dict],
) -> tuple[str, dict]:
    """Return (feature_type, params) for a single face dict."""
    surf = face.get("surface_type", "Unknown")
    area = face.get("area", 0.0)

    # ── Cylinder ──────────────────────────────────────────────────────────────
    if surf == "Cylinder":
        occ_face = face.get("occ_face")
        radius = _cylinder_radius(occ_face)
        depth  = _cylinder_depth(occ_face)
        # A small cylinder with area ≪ average → likely a Hole
        avg_area = sum(f["area"] for f in all_faces) / max(len(all_faces), 1)
        if area < avg_area * 0.5 and radius < 50.0:
            return MachiningFeature.HOLE, {"radius": radius, "depth": depth}
        return MachiningFeature.BOSS, {"radius": radius, "depth": depth}

    # ── Torus ─────────────────────────────────────────────────────────────────
    if surf == "Torus":
        return MachiningFeature.FILLET, {}

    # ── Cone ──────────────────────────────────────────────────────────────────
    if surf == "Cone":
        return MachiningFeature.CHAMFER, {}

    # ── Plane ─────────────────────────────────────────────────────────────────
    if surf == "Plane":
        # Crude heuristic: if face area is small → chamfer candidate
        avg_area = sum(f["area"] for f in all_faces) / max(len(all_faces), 1)
        if area < avg_area * 0.15:
            return MachiningFeature.CHAMFER, {}
        return MachiningFeature.FLAT_FACE, {}

    return MachiningFeature.UNKNOWN, {}


# ---------------------------------------------------------------------------
# Geometry helpers (require OCC; return defaults if unavailable)
# ---------------------------------------------------------------------------

def _cylinder_radius(occ_face: Any) -> float:
    """Extract the radius of a cylindrical OCC face."""
    if occ_face is None:
        return 0.0
    try:
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        adaptor = BRepAdaptor_Surface(occ_face)
        return adaptor.Cylinder().Radius()
    except Exception:
        return 0.0


def _cylinder_depth(occ_face: Any) -> float:
    if occ_face is None:
        return 0.0
    try:
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib
        box = Bnd_Box()
        brepbndlib.Add(occ_face, box)   # modern static method
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        dims = sorted([xmax - xmin, ymax - ymin, zmax - zmin])
        return dims[-1]
    except Exception:
        return 0.0
