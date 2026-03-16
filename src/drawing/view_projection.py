"""Orthographic view projector for BRep graphs."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from src.brep_graph.graph_builder import BRepGraph
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Projection directions for the four standard views
VIEW_DIRECTIONS: Dict[str, Tuple[float, float, float]] = {
    "FRONT": (0.0, -1.0, 0.0),
    "TOP": (0.0, 0.0, -1.0),
    "RIGHT": (1.0, 0.0, 0.0),
    "ISOMETRIC": (1.0, -1.0, 1.0),
}


@dataclass
class ProjectedLine:
    """A 2-D projected line segment."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    line_type: str = "VISIBLE"  # VISIBLE | HIDDEN | CENTER


@dataclass
class ProjectedArc:
    """A 2-D projected arc."""
    center: Tuple[float, float]
    radius: float
    start_angle: float = 0.0
    end_angle: float = 360.0
    line_type: str = "VISIBLE"


@dataclass
class ProjectedView:
    """
    A single orthographic view consisting of lines and arcs.

    Attributes
    ----------
    view_name:
        One of ``FRONT``, ``TOP``, ``RIGHT``, ``ISOMETRIC``.
    lines:
        List of 2-D line segments.
    arcs:
        List of 2-D arcs (circles).
    origin:
        2-D placement offset for this view on the sheet.
    scale:
        Drawing scale factor applied.
    """
    view_name: str = "FRONT"
    lines: List[ProjectedLine] = field(default_factory=list)
    arcs: List[ProjectedArc] = field(default_factory=list)
    origin: Tuple[float, float] = (0.0, 0.0)
    scale: float = 1.0


class ViewProjector:
    """
    Generates orthographic projections of a BRep graph.

    Uses parallel projection along view-specific directions.  The output is
    a dict of :class:`ProjectedView` objects keyed by view name.
    """

    def __init__(self) -> None:
        self._log = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(self, brep_graph: BRepGraph, view_name: str) -> ProjectedView:
        """
        Generate a single orthographic projection.

        Args:
            brep_graph: The BRep graph to project.
            view_name: One of ``FRONT``, ``TOP``, ``RIGHT``, ``ISOMETRIC``.

        Returns:
            A :class:`ProjectedView` with 2-D lines and arcs.
        """
        view_name = view_name.upper()
        direction = VIEW_DIRECTIONS.get(view_name, VIEW_DIRECTIONS["FRONT"])
        u, v = self._build_uv_axes(direction)

        lines: List[ProjectedLine] = []
        arcs: List[ProjectedArc] = []

        for node in brep_graph.node_list():
            if node.surface_type.upper() == "CYLINDER":
                arcs.extend(self._project_cylinder(node, u, v))
            else:
                lines.extend(self._project_plane_face(node, u, v))

        view = ProjectedView(view_name=view_name, lines=lines, arcs=arcs)
        self._log.debug(
            "View %s: %d lines, %d arcs.", view_name, len(lines), len(arcs)
        )
        return view

    def project_all(
        self,
        brep_graph: BRepGraph,
        view_names: List[str],
        sheet_size: str = "A3",
        scale: float = 1.0,
    ) -> Dict[str, ProjectedView]:
        """
        Generate multiple views and lay them out on the sheet.

        Args:
            brep_graph: The BRep graph.
            view_names: List of view names to generate.
            sheet_size: ISO sheet size string used for layout.
            scale: Drawing scale.

        Returns:
            Dictionary mapping view name → :class:`ProjectedView`.
        """
        views: Dict[str, ProjectedView] = {}
        offsets = self._layout_offsets(view_names, sheet_size, scale)
        for vname in view_names:
            view = self.project(brep_graph, vname)
            view.scale = scale
            view.origin = offsets.get(vname, (10.0, 10.0))
            views[vname] = view
        return views

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_uv_axes(
        direction: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build orthonormal UV axes perpendicular to the viewing direction.
        """
        d = np.array(direction, dtype=float)
        d /= np.linalg.norm(d) + 1e-12

        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(d, ref)) > 0.9:
            ref = np.array([1.0, 0.0, 0.0])

        u = ref - np.dot(ref, d) * d
        u /= np.linalg.norm(u) + 1e-12
        v = np.cross(d, u)
        v /= np.linalg.norm(v) + 1e-12
        return u, v

    @staticmethod
    def _project_point(
        point3d: Tuple[float, float, float],
        u: np.ndarray,
        v: np.ndarray,
    ) -> Tuple[float, float]:
        """Project a 3-D point to 2-D using UV axes."""
        p = np.array(point3d, dtype=float)
        return (float(np.dot(p, u)), float(np.dot(p, v)))

    def _project_plane_face(
        self, node, u: np.ndarray, v: np.ndarray
    ) -> List[ProjectedLine]:
        """Project a planar face by projecting its bounding-box corners."""
        bb = node.bounding_box
        corners3d = [
            (bb[0][0], bb[0][1], bb[0][2]),
            (bb[1][0], bb[0][1], bb[0][2]),
            (bb[1][0], bb[1][1], bb[0][2]),
            (bb[0][0], bb[1][1], bb[0][2]),
        ]
        pts2d = [self._project_point(c, u, v) for c in corners3d]
        lines: List[ProjectedLine] = []
        n = len(pts2d)
        for i in range(n):
            lines.append(ProjectedLine(
                start=pts2d[i],
                end=pts2d[(i + 1) % n],
                line_type="VISIBLE",
            ))
        return lines

    def _project_cylinder(
        self, node, u: np.ndarray, v: np.ndarray
    ) -> List[ProjectedArc]:
        """Project a cylindrical face as an arc/circle."""
        cx2d = self._project_point(node.center, u, v)
        area = max(node.area, 1.0)
        radius = math.sqrt(area / (2 * math.pi * max(1.0, area ** 0.3)))
        radius = max(2.0, min(radius, 50.0))
        return [ProjectedArc(
            center=cx2d,
            radius=radius,
            line_type="VISIBLE",
        )]

    # ------------------------------------------------------------------
    # Sheet layout
    # ------------------------------------------------------------------

    @staticmethod
    def _layout_offsets(
        view_names: List[str], sheet_size: str, scale: float
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute 2-D placement offsets for each view on the sheet.

        Layout convention (third-angle projection):
        - FRONT at (80, 80)
        - TOP directly above FRONT
        - RIGHT to the right of FRONT
        - ISOMETRIC top-right
        """
        offsets: Dict[str, Tuple[float, float]] = {}
        base_x, base_y = 80.0, 80.0
        step = 120.0 * max(scale, 0.1)
        for vname in view_names:
            if vname == "FRONT":
                offsets[vname] = (base_x, base_y)
            elif vname == "TOP":
                offsets[vname] = (base_x, base_y + step)
            elif vname == "RIGHT":
                offsets[vname] = (base_x + step, base_y)
            elif vname == "ISOMETRIC":
                offsets[vname] = (base_x + step, base_y + step)
            else:
                offsets[vname] = (base_x, base_y)
        return offsets
