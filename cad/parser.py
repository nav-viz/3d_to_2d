"""
cad/parser.py
=============
STEP file loading, normalisation, and topology extraction.

Uses OpenCASCADE (pythonocc-core) for:
  - Reading STEP via STEPControl_Reader
  - Healing / fixing shapes via ShapeFix_Shape
  - Extracting faces, edges, vertices with reliable deduplication
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import yaml
from utils.logger import logger

# ---------------------------------------------------------------------------
# Optional OCC import
# ---------------------------------------------------------------------------
try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopoDS import TopoDS_Compound
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import (
        TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_SHELL
    )
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.ShapeFix import ShapeFix_Shape
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import (
        GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
        GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface
    )
    _OCC_AVAILABLE = True
    logger.debug("pythonocc-core available – full STEP parsing enabled.")
except ImportError:
    _OCC_AVAILABLE = False
    logger.warning(
        "pythonocc-core NOT found. STEP parsing will return stub data. "
        "Install via: conda install -c conda-forge pythonocc-core=7.7.2"
    )

# ---------------------------------------------------------------------------
# Surface type mapping
# ---------------------------------------------------------------------------
SURF_TYPE_MAP: dict[int, str] = {
    0: "Plane",
    1: "Cylinder",
    2: "Cone",
    3: "Sphere",
    4: "Torus",
    7: "BSpline",
}

def _get_surface_type(face: Any) -> str:
    """Return a human-readable surface type string for an OCC face."""
    if not _OCC_AVAILABLE:
        return "Unknown"
    adaptor = BRepAdaptor_Surface(face)
    stype = adaptor.GetType()
    return SURF_TYPE_MAP.get(int(stype), "Other")


def _face_area(face: Any) -> float:
    """Compute the surface area using the static method."""
    if not _OCC_AVAILABLE:
        return 0.0
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)   # static method
    return props.Mass()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_step(step_path: str | Path) -> dict[str, Any]:
    """
    Parse a STEP file with improved error handling.
    """
    step_path = Path(step_path).resolve()
    if not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    logger.info(f"Loading STEP: {step_path.name} (full path: {step_path})")

    if not _OCC_AVAILABLE:
        logger.warning("Returning stub topology (OCC unavailable).")
        return _stub_topology(str(step_path))

    # ── Read STEP ─────────────────────────────────────────────────────────────
    try:
        reader = STEPControl_Reader()
        status = reader.ReadFile(str(step_path))
        if status != IFSelect_RetDone:
            raise RuntimeError(f"STEPControl_Reader failed with status {status}")
        reader.TransferRoots()
        shape = reader.OneShape()
        logger.debug("STEP read successfully ✓")
    except Exception as e:
        logger.error(f"Exception while reading STEP file {step_path}: {e}")
        raise RuntimeError(f"STEPControl_Reader failed on: {step_path}") from e

    # ── Heal shape ────────────────────────────────────────────────────────────
    shape = _heal_shape(shape)

    # ── Extract topology ──────────────────────────────────────────────────────
    faces, edges, vertices = _extract_topology(shape)

    result: dict[str, Any] = {
        "shape": shape,
        "faces": faces,
        "edges": edges,
        "vertices": vertices,
        "num_faces": len(faces),
        "num_edges": len(edges),
        "num_vertices": len(vertices),
        "source_file": str(step_path),
    }

    logger.info(
        f"Parsed: {len(faces)} faces | {len(edges)} edges | "
        f"{len(vertices)} vertices"
    )
    return result


def _heal_shape(shape: Any) -> Any:
    """Apply ShapeFix healing to improve shape validity."""
    fixer = ShapeFix_Shape(shape)
    fixer.Perform()
    healed = fixer.Shape()
    logger.debug("ShapeFix ✓")
    return healed


def _extract_topology(shape: Any) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Extract faces, edges, and vertices using TopTools_IndexedMapOfShape.
    Stores the exact OCC wrapper object for reliable adjacency later.
    """
    if not _OCC_AVAILABLE:
        return [], [], []

    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
    from OCC.Core.TopExp import TopExp_Explorer

    # ── Faces ────────────────────────────────────────────────────────────────
    face_map = TopTools_IndexedMapOfShape()
    face_list: list[dict] = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = explorer.Current()
        if face_map.Add(face):
            surf_type = _get_surface_type(face)
            area = _face_area(face)
            face_list.append({
                "id": len(face_list),
                "surface_type": surf_type,
                "area": float(area),
                "occ_face": face,          # ← Critical for adjacency
            })
        explorer.Next()

    # ── Edges ────────────────────────────────────────────────────────────────
    edge_map = TopTools_IndexedMapOfShape()
    edge_list: list[dict] = []
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while edge_explorer.More():
        edge = edge_explorer.Current()
        if edge_map.Add(edge):
            edge_list.append({
                "id": len(edge_list),
                "occ_edge": edge,
                "convexity": "unknown",
                "face_ids": [],
            })
        edge_explorer.Next()

    # ── Vertices ─────────────────────────────────────────────────────────────
    vertex_map = TopTools_IndexedMapOfShape()
    vertex_list: list[dict] = []
    vert_explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    while vert_explorer.More():
        vert = vert_explorer.Current()
        if vertex_map.Add(vert):
            try:
                from OCC.Core.BRep import BRep_Tool
                from OCC.Core.gp import gp_Pnt
                pnt: gp_Pnt = BRep_Tool.Pnt(vert)
                xyz = (float(pnt.X()), float(pnt.Y()), float(pnt.Z()))
            except Exception:
                xyz = (0.0, 0.0, 0.0)
            vertex_list.append({
                "id": len(vertex_list),
                "xyz": xyz,
            })
        vert_explorer.Next()

    logger.debug(
        f"Extracted topology: {len(face_list)} faces, "
        f"{len(edge_list)} edges, {len(vertex_list)} vertices"
    )
    return face_list, edge_list, vertex_list


def _stub_topology(source_file: str) -> dict[str, Any]:
    """Minimal stub when OCC is not available."""
