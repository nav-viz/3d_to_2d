"""
graph/extract.py
================
Topology feature extractor: converts parsed STEP topology → flat feature
arrays suitable for Neo4j storage and downstream graph construction.

Features per face (10 dims matching MFCAD++ V_1):
    [0]  surface_type_encoded   (int, see SURF_TYPE_ENC)
    [1]  area_normalised        (float)
    [2]  bbox_x                 (float)  -- stub 0 if OCC unavailable
    [3]  bbox_y
    [4]  bbox_z
    [5]  normal_x               (float)  -- stub 0 if OCC unavailable
    [6]  normal_y
    [7]  normal_z
    [8]  perimeter_norm         (float)  -- stub 0 if OCC unavailable
    [9]  is_outer_shell         (int, 0/1)

Extension point: add V_2 vertex features and V_3 edge features here.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from loguru import logger
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepGProp import brepgprop
# Surface type encoding (must match MFCAD++ convention)
SURF_TYPE_ENC: dict[str, int] = {
    "Plane":    0,
    "Cylinder": 1,
    "Cone":     2,
    "Sphere":   3,
    "Torus":    4,
    "BSpline":  5,
    "Other":    6,
    "Unknown":  7,
}

FACE_FEAT_DIM = 13


def extract_face_features(topology: dict[str, Any]) -> np.ndarray:
    """
    Enhanced: original 10 features + 3 convexity neighbor ratios (total 13 dims).
    """
    faces = topology.get("faces", [])
    edges = topology.get("edges", [])
    N = len(faces)
    if N == 0:
        return np.zeros((0, 13), dtype=np.float32)

    # Build per-face convexity counts
    adj_convex = [0] * N
    adj_concave = [0] * N
    adj_smooth = [0] * N

    for edge in edges:
        fids = edge.get("face_ids", [])
        conv = str(edge.get("convexity", "unknown")).lower()
        if len(fids) == 2:
            fa, fb = fids
            if conv == "convex":
                adj_convex[fa] += 1
                adj_convex[fb] += 1
            elif conv == "concave":
                adj_concave[fa] += 1
                adj_concave[fb] += 1
            elif conv == "smooth":
                adj_smooth[fa] += 1
                adj_smooth[fb] += 1

    feat = np.zeros((N, 13), dtype=np.float32)
    all_areas = np.array([f.get("area", 1.0) for f in faces], dtype=np.float32)
    area_max = all_areas.max() + 1e-8

    for i, face in enumerate(faces):
        # Original 10 features (indices 0-8)
        feat[i, 0] = float(SURF_TYPE_ENC.get(
            face.get("surface_type", "Unknown"), 7
        ))
        feat[i, 1] = face.get("area", 0.0) / area_max

        occ_face = face.get("occ_face")
        if occ_face is not None:
            bbox = _get_bbox(occ_face)
            feat[i, 2:5] = bbox
            normal = _get_face_normal(occ_face)
            feat[i, 5:8] = normal
            feat[i, 8] = _get_perimeter(occ_face) / (area_max ** 0.5 + 1e-8)

        # New features (9,10,11,12)
        total_nbr = adj_convex[i] + adj_concave[i] + adj_smooth[i] + 1e-8
        feat[i, 9]  = adj_convex[i] / total_nbr
        feat[i, 10] = adj_concave[i] / total_nbr
        feat[i, 11] = adj_smooth[i] / total_nbr
        feat[i, 12] = min(total_nbr / 30.0, 1.0)   # normalized degree

    logger.debug(f"Extracted enhanced feature matrix: {feat.shape} (13 dims)")
    return feat


def _get_bbox(occ_face: Any) -> np.ndarray:
    """Return normalised (dx, dy, dz) bounding box extents."""
    try:
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib_Add
        box = Bnd_Box()
        brepbndlib.Add(occ_face, box)
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        return np.array([xmax - xmin, ymax - ymin, zmax - zmin],
                        dtype=np.float32)
    except Exception:
        return np.zeros(3, dtype=np.float32)


def _get_face_normal(occ_face: Any) -> np.ndarray:
    """Return the face normal at the UV parameter-space centroid."""
    try:
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.GeomLProp import GeomLProp_SLProps
        adaptor = BRepAdaptor_Surface(occ_face)
        u = (adaptor.FirstUParameter() + adaptor.LastUParameter()) / 2
        v = (adaptor.FirstVParameter() + adaptor.LastVParameter()) / 2
        props = GeomLProp_SLProps(adaptor.Surface().Surface(), u, v, 1, 1e-6)
        if props.IsNormalDefined():
            n = props.Normal()
            return np.array([n.X(), n.Y(), n.Z()], dtype=np.float32)
    except Exception:
        pass
    return np.zeros(3, dtype=np.float32)


def _get_perimeter(occ_face: Any) -> float:
    """Compute face perimeter (sum of edge lengths)."""
    try:        
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_EDGE
        total = 0.0
        explorer = TopExp_Explorer(occ_face, TopAbs_EDGE)
        while explorer.More():
            edge = explorer.Current()
            props = GProp_GProps()
            brepgprop.LinearProperties(edge, props)
            total += props.Mass()
            explorer.Next()
        return total
    except Exception:
        return 0.0
