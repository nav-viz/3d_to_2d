"""Synthetic CAD model generator for training data and demos."""
from __future__ import annotations

import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

from src.step_parser.parse_step import (
    Edge,
    Face,
    ParsedCADModel,
    Surface,
    Vertex,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from OCC.Core.BRepPrimAPI import (
        BRepPrimAPI_MakeBox,
        BRepPrimAPI_MakeCylinder,
    )
    OCC_AVAILABLE = True
except ImportError:
    OCC_AVAILABLE = False


class SyntheticCADGenerator:
    """
    Generates synthetic CAD models with randomly placed features.

    Supported features: Hole, Pocket, Slot, Boss, Fillet, Chamfer, Rib.
    The generator produces :class:`~src.step_parser.ParsedCADModel` instances
    that can be fed directly into the pipeline without any real STEP file.
    """

    FEATURE_TYPES = ["Hole", "Pocket", "Slot", "Boss", "Fillet", "Chamfer", "Rib"]
    SURFACE_TYPES = ["PLANE", "CYLINDER", "CONE", "TORUS"]

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed if seed is not None else 42)
        self._log = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_part(self, complexity: int = 1) -> ParsedCADModel:
        """
        Generate a synthetic part with a specified complexity level.

        Args:
            complexity: Integer 1–5.  Higher values add more features
                        and irregular geometry.

        Returns:
            A :class:`ParsedCADModel` populated with synthetic data.
        """
        complexity = max(1, min(complexity, 5))
        num_features = complexity * 2 + self._rng.randint(0, 2)
        self._log.info(
            "Generating synthetic part  complexity=%d  features=%d",
            complexity, num_features,
        )

        # Base box dimensions
        bx = self._rng.uniform(50, 200) * complexity
        by = self._rng.uniform(30, 120) * complexity
        bz = self._rng.uniform(20, 60) * complexity

        vertices = self._make_box_vertices(bx, by, bz)
        edges = self._make_box_edges(vertices)
        faces, surfaces, labels = self._make_faces(
            bx, by, bz, vertices, edges, num_features
        )

        model = ParsedCADModel(
            faces=faces,
            edges=edges,
            vertices=vertices,
            surfaces=surfaces,
            source_file="<synthetic>",
            bounding_box=((0.0, 0.0, 0.0), (bx, by, bz)),
            metadata={
                "source": "synthetic",
                "complexity": complexity,
                "num_features": num_features,
                "feature_labels": labels,
            },
        )
        self._log.info(
            "  Generated: %d faces  %d edges  %d vertices",
            len(faces), len(edges), len(vertices),
        )
        return model

    def save_step(self, part: ParsedCADModel, path: str) -> bool:
        """
        Save the part as a STEP file (requires pythonOCC).

        Falls back gracefully when OCC is not installed.

        Args:
            part: The synthetic model to export.
            path: Output file path.

        Returns:
            ``True`` when the file was written.
        """
        if not OCC_AVAILABLE:
            self._log.warning("pythonOCC not available – cannot save STEP file.")
            return False
        try:  # pragma: no cover
            from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
            from OCC.Core.IFSelect import IFSelect_RetDone
            from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
            bb = part.bounding_box
            dx = bb[1][0] - bb[0][0]
            dy = bb[1][1] - bb[0][1]
            dz = bb[1][2] - bb[0][2]
            shape = BRepPrimAPI_MakeBox(dx, dy, dz).Shape()
            writer = STEPControl_Writer()
            writer.Transfer(shape, STEPControl_AsIs)
            status = writer.Write(path)
            if status == IFSelect_RetDone:
                self._log.info("STEP saved → '%s'", path)
                return True
            self._log.warning("STEP writer returned non-success status.")
        except Exception as exc:
            self._log.error("STEP save failed: %s", exc)
        return False

    def save_labels(self, labels: Dict, path: str) -> None:
        """
        Save feature labels to a JSON file.

        Args:
            labels: Dictionary of face_id → feature_label.
            path: Output JSON file path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2)
        self._log.info("Labels saved → '%s'", path)

    # ------------------------------------------------------------------
    # Private geometry builders
    # ------------------------------------------------------------------

    def _make_box_vertices(
        self, bx: float, by: float, bz: float
    ) -> List[Vertex]:
        """Generate the 8 corners of an axis-aligned box."""
        corners = [
            (0, 0, 0), (bx, 0, 0), (bx, by, 0), (0, by, 0),
            (0, 0, bz), (bx, 0, bz), (bx, by, bz), (0, by, bz),
        ]
        return [Vertex(i, x, y, z) for i, (x, y, z) in enumerate(corners)]

    @staticmethod
    def _make_box_edges(vertices: List[Vertex]) -> List[Edge]:
        """Generate the 12 edges of a box."""
        connectivity = [
            (0, 1), (1, 2), (2, 3), (3, 0),   # bottom ring
            (4, 5), (5, 6), (6, 7), (7, 4),   # top ring
            (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
        ]
        edges: List[Edge] = []
        for eid, (si, ti) in enumerate(connectivity):
            vs, vt = vertices[si], vertices[ti]
            length = math.sqrt(
                (vt.x - vs.x) ** 2 + (vt.y - vs.y) ** 2 + (vt.z - vs.z) ** 2
            )
            mid = ((vs.x + vt.x) / 2, (vs.y + vt.y) / 2, (vs.z + vt.z) / 2)
            edges.append(Edge(eid, si, ti, "LINE", length, mid))
        return edges

    def _make_faces(
        self,
        bx: float,
        by: float,
        bz: float,
        vertices: List[Vertex],
        edges: List[Edge],
        num_features: int,
    ) -> Tuple[List[Face], List[Surface], Dict[int, str]]:
        """Generate box faces plus additional feature faces."""
        faces: List[Face] = []
        surfaces: List[Surface] = []
        labels: Dict[int, str] = {}

        # Six box faces
        box_faces = [
            (0, "PLANE", (0.0, 0.0, -1.0), (bx / 2, by / 2, 0)),
            (1, "PLANE", (0.0, 0.0, 1.0), (bx / 2, by / 2, bz)),
            (2, "PLANE", (0.0, -1.0, 0.0), (bx / 2, 0, bz / 2)),
            (3, "PLANE", (1.0, 0.0, 0.0), (bx, by / 2, bz / 2)),
            (4, "PLANE", (0.0, 1.0, 0.0), (bx / 2, by, bz / 2)),
            (5, "PLANE", (-1.0, 0.0, 0.0), (0, by / 2, bz / 2)),
        ]
        for fid, stype, normal, center in box_faces:
            area = self._rng.uniform(bx * by * 0.5, bx * by * 2)
            face = Face(
                id=fid,
                surface_type=stype,
                area=area,
                normal=normal,  # type: ignore[arg-type]
                center=center,  # type: ignore[arg-type]
                curvature=0.0,
                bounding_box=((0.0, 0.0, 0.0), (bx, by, bz)),
            )
            faces.append(face)
            surfaces.append(Surface(fid, stype))
            labels[fid] = "Unknown"

        # Feature faces
        fid = 6
        for i in range(num_features):
            feature_type = self._rng.choice(self.FEATURE_TYPES)
            cx = self._rng.uniform(bx * 0.1, bx * 0.9)
            cy = self._rng.uniform(by * 0.1, by * 0.9)
            cz = self._rng.uniform(bz * 0.1, bz * 0.9)

            if feature_type in ("Hole", "Boss"):
                stype = "CYLINDER"
                curvature = self._rng.uniform(0.03, 0.15)
                area = self._rng.uniform(100, 500)
            elif feature_type in ("Fillet",):
                stype = "TORUS"
                curvature = self._rng.uniform(0.1, 0.5)
                area = self._rng.uniform(20, 100)
            elif feature_type in ("Chamfer",):
                stype = "CONE"
                curvature = 0.0
                area = self._rng.uniform(30, 150)
            else:
                stype = "PLANE"
                curvature = 0.0
                area = self._rng.uniform(200, 1500)

            face = Face(
                id=fid,
                surface_type=stype,
                area=area,
                normal=(0.0, 0.0, 1.0),
                center=(cx, cy, cz),
                curvature=curvature,
                bounding_box=((0.0, 0.0, 0.0), (bx, by, bz)),
            )
            faces.append(face)
            surfaces.append(Surface(fid, stype))
            labels[fid] = feature_type
            fid += 1

        return faces, surfaces, labels
