"""Geometry extractor that wraps a ParsedCADModel."""
from __future__ import annotations

from typing import Dict, List, Tuple

from src.step_parser.parse_step import Face, ParsedCADModel
from src.utils.logger import get_logger
from src.utils.geometry_utils import GeometryUtils

logger = get_logger(__name__)


class GeometryExtractor:
    """
    Extracts structured geometry data from a :class:`ParsedCADModel`.

    All methods return plain Python / NumPy structures so they remain
    independent of any CAD kernel.
    """

    def __init__(self, model: ParsedCADModel) -> None:
        """
        Initialise the extractor.

        Args:
            model: The parsed CAD model to extract from.
        """
        self._model = model
        self._log = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def extract_faces(self) -> List[Dict]:
        """
        Return a list of face attribute dictionaries.

        Each dictionary contains:
        ``id``, ``surface_type``, ``area``, ``normal``,
        ``center``, ``curvature``, ``bounding_box``.
        """
        result = []
        for face in self._model.faces:
            result.append({
                "id": face.id,
                "surface_type": face.surface_type,
                "area": face.area,
                "normal": face.normal,
                "center": face.center,
                "curvature": face.curvature,
                "bounding_box": face.bounding_box,
                "vertex_ids": face.vertices,
                "edge_ids": face.edges,
            })
        self._log.debug("Extracted %d faces.", len(result))
        return result

    def extract_edges(self) -> List[Dict]:
        """
        Return a list of edge attribute dictionaries.

        Each dictionary contains:
        ``id``, ``edge_type``, ``start_vertex_id``, ``end_vertex_id``,
        ``length``, ``mid_point``.
        """
        result = []
        for edge in self._model.edges:
            result.append({
                "id": edge.id,
                "edge_type": edge.edge_type,
                "start_vertex_id": edge.start_vertex_id,
                "end_vertex_id": edge.end_vertex_id,
                "length": edge.length,
                "mid_point": edge.mid_point,
            })
        self._log.debug("Extracted %d edges.", len(result))
        return result

    def extract_vertices(self) -> List[Dict]:
        """
        Return a list of vertex attribute dictionaries.

        Each dictionary contains: ``id``, ``x``, ``y``, ``z``.
        """
        result = []
        for vertex in self._model.vertices:
            result.append({
                "id": vertex.id,
                "x": vertex.x,
                "y": vertex.y,
                "z": vertex.z,
                "coords": vertex.coords,
            })
        self._log.debug("Extracted %d vertices.", len(result))
        return result

    def extract_bounding_box(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Compute the overall axis-aligned bounding box of the model.

        Returns:
            ``(min_point, max_point)`` as nested 3-tuples.
        """
        vertex_coords = [(v.x, v.y, v.z) for v in self._model.vertices]
        if vertex_coords:
            bb = GeometryUtils.compute_bounding_box(vertex_coords)
        else:
            bb = self._model.bounding_box
        self._log.debug("Bounding box: min=%s  max=%s", bb[0], bb[1])
        return bb

    def extract_surface_types(self) -> Dict[str, int]:
        """Return a frequency count of surface types in the model."""
        counts: Dict[str, int] = {}
        for face in self._model.faces:
            counts[face.surface_type] = counts.get(face.surface_type, 0) + 1
        return counts

    def extract_feature_regions(self) -> List[Dict]:
        """
        Heuristically group faces into candidate feature regions.

        Cylindrical / conical faces are candidate holes or bosses; planar
        clusters are candidate pockets or slots.

        Returns:
            List of region dictionaries, each with ``face_ids``,
            ``region_type``, and ``centroid``.
        """
        regions: List[Dict] = []
        cylindrical_faces = [f for f in self._model.faces if f.surface_type == "CYLINDER"]
        for face in cylindrical_faces:
            regions.append({
                "face_ids": [face.id],
                "region_type": "CYLINDRICAL",
                "centroid": face.center,
                "area": face.area,
            })

        planar_faces = [f for f in self._model.faces if f.surface_type == "PLANE"]
        if planar_faces:
            regions.append({
                "face_ids": [f.id for f in planar_faces],
                "region_type": "PLANAR_GROUP",
                "centroid": self._mean_center(planar_faces),
                "area": sum(f.area for f in planar_faces),
            })

        self._log.debug("Found %d feature regions.", len(regions))
        return regions

    def compute_complexity_score(self) -> float:
        """
        Compute a rough complexity score for the model.

        Score = (num_faces * 0.4) + (num_edges * 0.3) + (curvature_sum * 0.3)
        """
        num_faces = len(self._model.faces)
        num_edges = len(self._model.edges)
        curvature_sum = sum(f.curvature for f in self._model.faces)
        score = num_faces * 0.4 + num_edges * 0.3 + min(curvature_sum, 10.0) * 0.3
        self._log.debug("Complexity score: %.2f", score)
        return round(score, 4)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_center(faces: List[Face]) -> Tuple[float, float, float]:
        """Return the mean center of a list of faces."""
        if not faces:
            return (0.0, 0.0, 0.0)
        cx = sum(f.center[0] for f in faces) / len(faces)
        cy = sum(f.center[1] for f in faces) / len(faces)
        cz = sum(f.center[2] for f in faces) / len(faces)
        return (cx, cy, cz)
