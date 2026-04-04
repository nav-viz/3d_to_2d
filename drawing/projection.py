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
from loguru import logger

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

def _project_occ(
    shape: Any,
    view_name: str,
    direction: tuple[float, float, float],
    scale: float,
) -> ProjectionResult:
    """Full OpenCASCADE HLR projection."""
    dx, dy, dz = direction
    norm = math.sqrt(dx*dx + dy*dy + dz*dz) + 1e-12
    dx, dy, dz = dx/norm, dy/norm, dz/norm

    proj_dir = gp_Dir(dx, dy, dz)
    eye = gp_Pnt(dx * 1000, dy * 1000, dz * 1000)
    projector = HLRAlgo_Projector(
        gp_Ax2(eye, proj_dir), False, scale
    )

    algo = HLRBRep_Algo()
    algo.Add(shape)
    algo.Projector(projector)
    algo.Update()
    algo.Hide()

    to_shape = HLRBRep_HLRToShape(algo)

    visible_lines = _extract_edges(to_shape.VCompound(), hidden=False)
    hidden_lines  = _extract_edges(to_shape.HCompound(), hidden=True)

    bbox = _compute_bbox_2d(visible_lines + hidden_lines)

    logger.debug(
        f"OCC HLR '{view_name}': "
        f"{len(visible_lines)} visible, {len(hidden_lines)} hidden lines"
    )
    return ProjectionResult(
        view_name=view_name,
        visible_lines=visible_lines,
        hidden_lines=hidden_lines,
        bounding_box=bbox,
        scale=scale,
    )


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
