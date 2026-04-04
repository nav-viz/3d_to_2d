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


def _face_area(face: Any) -> float:
    """Compute the surface area of an OCC face in native units."""
    props = GProp_GProps()
    brepgprop_SurfaceProperties(face, props)
    return props.Mass()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_step(step_path: str | Path) -> dict[str, Any]:
    """
    Parse a STEP file and return a normalised topology dictionary.

    Parameters
    ----------
    step_path : str or Path
        Absolute or relative path to the ``.step`` / ``.stp`` file.

    Returns
    -------
    dict with keys:
        ``shape``      – raw OCC TopoDS_Shape (or None if OCC unavailable)
        ``faces``      – list of face dicts (id, surface_type, area, normal)
        ``edges``      – list of edge dicts (id, face_ids, convexity)
        ``vertices``   – list of vertex dicts (id, xyz)
        ``num_faces``  – int
        ``num_edges``  – int
        ``num_vertices`` – int
        ``source_file`` – str
    """
    step_path = Path(step_path)
    if not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    logger.info(f"Loading STEP: {step_path.name}")

    if not _OCC_AVAILABLE:
        logger.warning("Returning stub topology (OCC unavailable).")
        return _stub_topology(str(step_path))

    # ── Read ─────────────────────────────────────────────────────────────────
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEPControl_Reader failed on: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    logger.debug("STEP read ✓")

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
    """Enumerate all faces, edges, and vertices from a healed OCC shape."""
    # ── Faces ─────────────────────────────────────────────────────────────────
    face_list: list[dict] = []
    face_hash_to_id: dict[int, int] = {}
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    fid = 0
    while explorer.More():
        face = explorer.Current()
        fhash = face.HashCode(2**31)
        if fhash not in face_hash_to_id:
            face_hash_to_id[fhash] = fid
            surf_type = _get_surface_type(face)
            area = _face_area(face)
            face_list.append({
                "id": fid,
                "surface_type": surf_type,
                "area": area,
                "occ_face": face,  # keep for downstream HLR
            })
            fid += 1
        explorer.Next()

    # ── Edges ─────────────────────────────────────────────────────────────────
    edge_list: list[dict] = []
    eid = 0
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    seen_edges: set[int] = set()
    while edge_explorer.More():
        edge = edge_explorer.Current()
        ehash = edge.HashCode(2**31)
        if ehash not in seen_edges:
            seen_edges.add(ehash)
            edge_list.append({
                "id": eid,
                "occ_edge": edge,
                "convexity": "unknown",  # filled by brep_features.py
                "face_ids": [],           # filled by adjacency pass
            })
            eid += 1
        edge_explorer.Next()

    # ── Vertices ─────────────────────────────────────────────────────────────
    vertex_list: list[dict] = []
    vid = 0
    vert_explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    seen_verts: set[int] = set()
    while vert_explorer.More():
        vert = vert_explorer.Current()
        vhash = vert.HashCode(2**31)
        if vhash not in seen_verts:
            seen_verts.add(vhash)
            from OCC.Core.BRep import BRep_Tool
            pnt: gp_Pnt = BRep_Tool.Pnt(vert)
            vertex_list.append({
                "id": vid,
                "xyz": (pnt.X(), pnt.Y(), pnt.Z()),
            })
            vid += 1
        vert_explorer.Next()

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
