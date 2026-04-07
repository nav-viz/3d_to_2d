"""
cad/parser.py
=============
STEP file loading, normalisation, and topology extraction.

Uses OpenCASCADE (pythonocc-core) for:
  - Reading STEP via STEPControl_Reader
  - Healing / fixing shapes via ShapeFix_Shape
  - Iterating topology: shells → faces → edges → vertices

If pythonocc-core is not installed the module degrades gracefully and
returns a minimal stub dict so the rest of the pipeline can still run
with synthetic data.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import yaml
from loguru import logger
from OCC.Core.BRepGProp import brepgprop
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
    from OCC.Core.BRepGProp import brepgprop_SurfaceProperties, brepgprop_LinearProperties
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


def _get_surface_type(face: Any) -> str:  # noqa: ANN401
    """Return a human-readable surface type string for an OCC face."""
    if not _OCC_AVAILABLE:
        return "Unknown"
    adaptor = BRepAdaptor_Surface(face)
    stype = adaptor.GetType()
    return SURF_TYPE_MAP.get(int(stype), "Other")


# cad/parser.py - Replace the _face_area function

def _face_area(face: Any) -> float:
    """Compute the surface area of an OCC face using the recommended static method."""
    props = GProp_GProps()
    # Updated: use static method instead of deprecated function
    brepgprop.SurfaceProperties(face, props)
    return props.Mass()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_step(step_path: str | Path) -> dict[str, Any]:
    """
    Parse a STEP file with improved error handling and path validation.
    """
    step_path = Path(step_path).resolve()   # Make absolute and clean
    if not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    logger.info(f"Loading STEP: {step_path.name} (full path: {step_path})")

    if not _OCC_AVAILABLE:
        logger.warning("Returning stub topology (OCC unavailable).")
        return _stub_topology(str(step_path))

    # ── Read with better error reporting ─────────────────────────────────────
    try:
        reader = STEPControl_Reader()
        status = reader.ReadFile(str(step_path))
        
        if status != IFSelect_RetDone:
            error_msg = f"STEPControl_Reader failed with status {status} on: {step_path}"
            logger.error(error_msg)
            # Try to give more info
            raise RuntimeError(error_msg)
            
        reader.TransferRoots()
        shape = reader.OneShape()
        logger.debug("STEP read successfully ✓")

    except Exception as e:
        logger.error(f"Exception while reading STEP file {step_path}: {e}")
        raise RuntimeError(f"STEPControl_Reader failed on: {step_path}") from e

    # ── Heal ─────────────────────────────────────────────────────────────────
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


def _extract_topology(shape: Any) -> tuple[list, list, list]:
    """Enumerate all faces, edges, and vertices from a healed OCC shape.
    
    Uses TopTools_IndexedMapOfShape for reliable deduplication (recommended way in modern OCC).
    """
    if not _OCC_AVAILABLE:
        return [], [], []

    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
    from OCC.Core.TopExp import TopExp_Explorer

    # ── Faces ─────────────────────────────────────────────────────────────────
    face_map = TopTools_IndexedMapOfShape()
    face_list: list[dict] = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = explorer.Current()
        if face_map.Add(face):   # Add returns True if newly added (deduplicates)
            surf_type = _get_surface_type(face)
            area = _face_area(face)
            face_list.append({
                "id": len(face_list),                    # sequential id
                "surface_type": surf_type,
                "area": area,
                "occ_face": face,
            })
        explorer.Next()

    # ── Edges ─────────────────────────────────────────────────────────────────
    edge_map = TopTools_IndexedMapOfShape()
    edge_list: list[dict] = []
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while edge_explorer.More():
        edge = edge_explorer.Current()
        if edge_map.Add(edge):
            edge_list.append({
                "id": len(edge_list),
                "occ_edge": edge,
                "convexity": "unknown",   # will be filled later in brep_features or adjacency
                "face_ids": [],           # filled during adjacency pass
            })
        edge_explorer.Next()

    # ── Vertices ──────────────────────────────────────────────────────────────
    vertex_map = TopTools_IndexedMapOfShape()
    vertex_list: list[dict] = []
    vert_explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    while vert_explorer.More():
        vert = vert_explorer.Current()
        if vertex_map.Add(vert):
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.gp import gp_Pnt
            pnt: gp_Pnt = BRep_Tool.Pnt(vert)
            vertex_list.append({
                "id": len(vertex_list),
                "xyz": (pnt.X(), pnt.Y(), pnt.Z()),
            })
        vert_explorer.Next()

    logger.debug(
        f"Extracted topology: {len(face_list)} faces, "
        f"{len(edge_list)} edges, {len(vertex_list)} vertices"
    )
    return face_list, edge_list, vertex_list


def _stub_topology(source_file: str) -> dict[str, Any]:
    """Return a minimal topology dict when OCC is not available."""
    stub_faces = [
        {"id": i, "surface_type": "Plane", "area": 1.0 + i * 0.5,
         "occ_face": None}
        for i in range(6)
    ]
    stub_edges = [
        {"id": i, "occ_edge": None, "convexity": "unknown", "face_ids": []}
        for i in range(12)
    ]
    stub_vertices = [
        {"id": i, "xyz": (float(i), 0.0, 0.0)}
        for i in range(8)
    ]
    return {
        "shape": None,
        "faces": stub_faces,
        "edges": stub_edges,
        "vertices": stub_vertices,
        "num_faces": 6,
        "num_edges": 12,
        "num_vertices": 8,
        "source_file": source_file,
    }


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load the YAML configuration file."""
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)
