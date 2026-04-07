"""
drawing/projection.py
=====================
Hidden Line Removal (HLR) projection stub using OpenCASCADE.

For each selected view, projects the 3D shape onto a 2D plane and
computes visible + hidden edge polylines.

If pythonocc-core is not installed, returns synthetic stub wireframes
so the rest of the pipeline can still produce DXF output.

Extension point
---------------
Replace the stub wireframe with full OCC HLRBRep_Algo computation:
    from OCC.Core.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
    from OCC.Core.HLRAlgo import HLRAlgo_Projector
"""

from __future__ import annotations

import math
from typing import Any
from utils.logger import logger

try:
    # Full HLR pipeline (requires pythonocc-core >= 7.7)
    from OCC.Core.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
    from OCC.Core.HLRAlgo import HLRAlgo_Projector
    from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir, gp_XYZ
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    _OCC_HLR = True
    logger.debug("OCC HLR available – full projection enabled.")
except ImportError:
    _OCC_HLR = False
    logger.warning(
        "OCC HLR not available – using stub wireframe projection. "
        "Install pythonocc-core for real hidden-line removal."
    )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field


@dataclass
class Polyline:
    """A sequence of 2D points forming a projected edge."""
    points: list[tuple[float, float]]
    is_hidden: bool = False
    layer: str = "0"


@dataclass
class ProjectionResult:
    """Result of HLR projection for one view."""
    view_name: str
    visible_lines: list[Polyline]
    hidden_lines:  list[Polyline]
    bounding_box:  tuple[float, float, float, float]  # xmin, ymin, xmax, ymax
    scale: float = 1.0


def _get_bounding_box(shape: Any, use_mesh: bool = True) -> tuple[float, float, float, float]:
    """Compute axis-aligned OCC bounding box; returns stub extents if OCC unavailable."""
    if shape is None:
        return (0.0, 0.0, 0.0, 100.0, 100.0, 100.0)
    try:
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib_Add
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh

        bbox = Bnd_Box()
        if use_mesh:
            mesh = BRepMesh_IncrementalMesh(shape, 0.1)  # deflection
            mesh.Perform()
        brepbndlib_Add(shape, bbox, use_mesh)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        return (xmin, ymin, zmin, xmax, ymax, zmax)
    except Exception:
        # Fallback if box is void/infinite
        return (0.0, 0.0, 0.0, 100.0, 80.0, 50.0)
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def project_view(
    shape: Any,
    view_name: str,
    direction: tuple[float, float, float],
    scale: float = 1.0,
) -> ProjectionResult:
    """
    Compute the 2D HLR projection of ``shape`` along ``direction``.

    Parameters
    ----------
    shape      : OCC TopoDS_Shape (or None for stub)
    view_name  : e.g., "front", "top"
    direction  : unit projection normal, e.g., (0, -1, 0)
    scale      : draw scale factor

    Returns
    -------
    ProjectionResult
    """
    if shape is not None and _OCC_HLR:
        return _project_occ(shape, view_name, direction, scale)
    else:
        logger.debug(
            f"Stub projection for view='{view_name}' direction={direction}"
        )
        return _stub_projection(view_name, scale)


# ---------------------------------------------------------------------------
# OCC HLR implementation
# ---------------------------------------------------------------------------

def _project_occ(shape, view_name: str, direction: tuple, scale: float = 1.0):
    """Safe HLR projection compatible with pythonocc-core 7.7.x"""
    from OCC.Core.HLRAlgo import HLRAlgo_Projector
    from OCC.Core.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
    from OCC.Core.gp import gp_Dir, gp_Ax2, gp_Pnt
    from loguru import logger

    # View definitions
    if view_name == "front":
        dir_vec = gp_Dir(0, 0, -1)
        up_vec = gp_Dir(0, 1, 0)
    elif view_name == "top":
        dir_vec = gp_Dir(0, -1, 0)
        up_vec = gp_Dir(0, 0, 1)
    elif view_name == "right":
        dir_vec = gp_Dir(1, 0, 0)
        up_vec = gp_Dir(0, 1, 0)
    elif view_name == "bottom":
        dir_vec = gp_Dir(0, 1, 0)
        up_vec = gp_Dir(0, 0, -1)
    elif view_name == "isometric":
        # Safe normalization for isometric
        dir_vec = gp_Dir(1.0, 1.0, 1.0)
        # Use the static way to avoid method name issues
        norm = (dir_vec.X()**2 + dir_vec.Y()**2 + dir_vec.Z()**2)**0.5
        dir_vec = gp_Dir(dir_vec.X()/norm, dir_vec.Y()/norm, dir_vec.Z()/norm)
        up_vec = gp_Dir(0.0, 0.0, 1.0)
    else:  # default
        dir_vec = gp_Dir(0, 0, -1)
        up_vec = gp_Dir(0, 1, 0)

    ax2 = gp_Ax2(gp_Pnt(0, 0, 0), dir_vec, up_vec)

    # Safe projector creation
    try:
        projector = HLRAlgo_Projector(ax2)
    except Exception as e:
        logger.warning(f"HLR Projector failed with ax2, trying fallback: {e}")
        try:
            projector = HLRAlgo_Projector(ax2, False, 0.01)
        except Exception:
            logger.warning("Using bare HLRAlgo_Projector")
            projector = HLRAlgo_Projector()

    # Run HLR
    hlr = HLRBRep_Algo()
    hlr.Add(shape)
    hlr.Projector(projector)
    hlr.Update()
    hlr.Hide()

    hlr_shape = HLRBRep_HLRToShape(hlr)

    visible = hlr_shape.VCompound()
    hidden = hlr_shape.HCompound()

    bbox = _get_bounding_box(shape)

    return {
        "visible": visible,
        "hidden": hidden,
        "view_name": view_name,
        "direction": direction,
        "bounding_box": bbox
    }
def _extract_edges(compound: Any, hidden: bool) -> list[Polyline]:
    """Extract polylines from an OCC compound (visible or hidden)."""
    polylines: list[Polyline] = []
    if compound.IsNull():
        return polylines

    explorer = TopExp_Explorer(compound, TopAbs_EDGE)
    while explorer.More():
        edge = explorer.Current()
        pts = _discretise_edge(edge)
        if pts:
            polylines.append(Polyline(
                points=pts,
                is_hidden=hidden,
                layer="HIDDEN" if hidden else "VISIBLE",
            ))
        explorer.Next()
    return polylines


def _discretise_edge(
    edge: Any,
    n_samples: int = 20,
) -> list[tuple[float, float]]:
    """Sample an OCC 2D edge into a polyline of (x, y) tuples."""
    try:
        adaptor = BRepAdaptor_Curve(edge)
        u0, u1 = adaptor.FirstParameter(), adaptor.LastParameter()
        pts = []
        for i in range(n_samples + 1):
            u = u0 + (u1 - u0) * i / n_samples
            p = adaptor.Value(u)
            pts.append((p.X(), p.Y()))
        return pts
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Stub implementation
# ---------------------------------------------------------------------------

def _stub_projection(
    view_name: str,
    scale: float,
) -> ProjectionResult:
    """
    Return a simple rectangular wireframe as a placeholder when OCC is absent.
    Suitable for testing the DXF writer without a real CAD shape.
    """
    w, h = 100.0 * scale, 80.0 * scale
    corners = [(0, 0), (w, 0), (w, h), (0, h), (0, 0)]
    diag    = [(0+5, 0+5), (w-5, h-5)]

    visible = [
        Polyline(corners, is_hidden=False, layer="VISIBLE"),
        Polyline(diag,    is_hidden=False, layer="VISIBLE"),
    ]
    hidden = [
        Polyline([(w/2, 0), (w/2, h)], is_hidden=True, layer="HIDDEN"),
    ]
    return ProjectionResult(
        view_name=view_name,
        visible_lines=visible,
        hidden_lines=hidden,
        bounding_box=(0.0, 0.0, w, h),
        scale=scale,
    )


def _compute_bbox_2d(
    polylines: list[Polyline],
) -> tuple[float, float, float, float]:
    """Compute bounding box of all 2D polylines."""
    xs = [p[0] for pl in polylines for p in pl.points]
    ys = [p[1] for pl in polylines for p in pl.points]
    if not xs:
        return (0.0, 0.0, 100.0, 80.0)
    return (min(xs), min(ys), max(xs), max(ys))
