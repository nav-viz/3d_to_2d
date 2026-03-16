"""Dimension candidate generator following ISO 129 / ASME Y14 rules."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.brep_graph.graph_builder import BRepGraph
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DimensionCandidate:
    """
    A single proposed dimension annotation.

    Attributes
    ----------
    id:
        Unique string identifier.
    dim_type:
        One of ``DIAMETER``, ``DEPTH``, ``LENGTH``, ``WIDTH``,
        ``RADIUS``, ``ANGLE``, ``SPACING``, ``GENERAL``.
    value:
        Numerical dimension value in mm (or degrees for angles).
    start_point:
        2-D start anchor point in drawing space (x, y).
    end_point:
        2-D end anchor point in drawing space (x, y).
    text:
        Human-readable annotation string.
    feature_label:
        Manufacturing feature class this dimension belongs to.
    face_id:
        Source face identifier.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    dim_type: str = "GENERAL"
    value: float = 0.0
    start_point: Tuple[float, float] = (0.0, 0.0)
    end_point: Tuple[float, float] = (0.0, 0.0)
    text: str = ""
    feature_label: str = "Unknown"
    face_id: int = -1
    score: float = 0.0


class DimensionGenerator:
    """
    Generates dimension candidates from a BRep graph and feature predictions.

    Applies ISO 129 / ASME Y14.5 dimensioning rules:
    - Holes and bosses → diameter + depth
    - Pockets / Slots → length + width + depth
    - Fillets / Chamfers → radius / angle
    - Overall bounding box → width + height + depth
    - Feature spacing → centre-to-centre distances
    """

    def __init__(self) -> None:
        self._log = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        brep_graph: BRepGraph,
        feature_predictions: Dict,
    ) -> List[DimensionCandidate]:
        """
        Generate dimension candidates.

        Args:
            brep_graph: The BRep graph with node geometry.
            feature_predictions: Output of
                :class:`~src.ai_models.GNNFeatureModel.predict` containing
                ``face_labels`` and (optionally) ``feature_params`` per face.

        Returns:
            List of :class:`DimensionCandidate` objects.
        """
        self._log.info("Generating dimension candidates …")
        candidates: List[DimensionCandidate] = []
        face_labels: Dict[int, str] = feature_predictions.get("face_labels", {})
        feature_params: Dict[int, Dict] = feature_predictions.get("feature_params", {})

        for node in brep_graph.node_list():
            label = face_labels.get(node.id, "Unknown")
            params = feature_params.get(node.id, {})
            candidates.extend(
                self._dims_for_feature(node.id, label, params, node.center)
            )

        # Overall bounding-box dimensions
        all_centers = [n.center for n in brep_graph.node_list()]
        if all_centers:
            candidates.extend(self._overall_dims(all_centers))

        # Spacing between cylindrical features
        cylinders = [
            n for n in brep_graph.node_list()
            if face_labels.get(n.id) in ("Hole", "Boss")
        ]
        candidates.extend(self._spacing_dims(cylinders))

        self._log.info("Generated %d dimension candidates.", len(candidates))
        return candidates

    # ------------------------------------------------------------------
    # Feature-specific dimension rules
    # ------------------------------------------------------------------

    @staticmethod
    def _dims_for_feature(
        face_id: int,
        label: str,
        params: Dict,
        center: Tuple[float, float, float],
    ) -> List[DimensionCandidate]:
        """Produce dimension candidates for a single recognised feature."""
        cx, cy = float(center[0]), float(center[1])
        candidates: List[DimensionCandidate] = []

        if label == "Hole":
            diam = params.get("diameter", 10.0)
            depth = params.get("depth", 15.0)
            candidates.append(DimensionCandidate(
                dim_type="DIAMETER",
                value=diam,
                start_point=(cx - diam / 2, cy),
                end_point=(cx + diam / 2, cy),
                text=f"⌀{diam:.2f}",
                feature_label=label,
                face_id=face_id,
            ))
            candidates.append(DimensionCandidate(
                dim_type="DEPTH",
                value=depth,
                start_point=(cx + diam / 2 + 5, cy),
                end_point=(cx + diam / 2 + 5, cy - depth),
                text=f"↧{depth:.2f}",
                feature_label=label,
                face_id=face_id,
            ))

        elif label == "Boss":
            diam = params.get("diameter", 20.0)
            height = params.get("height", 10.0)
            candidates.append(DimensionCandidate(
                dim_type="DIAMETER",
                value=diam,
                start_point=(cx - diam / 2, cy),
                end_point=(cx + diam / 2, cy),
                text=f"⌀{diam:.2f}",
                feature_label=label,
                face_id=face_id,
            ))
            candidates.append(DimensionCandidate(
                dim_type="LENGTH",
                value=height,
                start_point=(cx + diam / 2 + 5, cy),
                end_point=(cx + diam / 2 + 5, cy + height),
                text=f"{height:.2f}",
                feature_label=label,
                face_id=face_id,
            ))

        elif label in ("Pocket", "Slot"):
            length = params.get("length", 30.0)
            width = params.get("width", 15.0)
            depth = params.get("depth", 10.0)
            candidates.append(DimensionCandidate(
                dim_type="LENGTH",
                value=length,
                start_point=(cx - length / 2, cy - width / 2 - 5),
                end_point=(cx + length / 2, cy - width / 2 - 5),
                text=f"{length:.2f}",
                feature_label=label,
                face_id=face_id,
            ))
            candidates.append(DimensionCandidate(
                dim_type="WIDTH",
                value=width,
                start_point=(cx + length / 2 + 5, cy - width / 2),
                end_point=(cx + length / 2 + 5, cy + width / 2),
                text=f"{width:.2f}",
                feature_label=label,
                face_id=face_id,
            ))
            candidates.append(DimensionCandidate(
                dim_type="DEPTH",
                value=depth,
                start_point=(cx, cy),
                end_point=(cx, cy - depth),
                text=f"↧{depth:.2f}",
                feature_label=label,
                face_id=face_id,
            ))

        elif label == "Fillet":
            radius = params.get("radius", 3.0)
            candidates.append(DimensionCandidate(
                dim_type="RADIUS",
                value=radius,
                start_point=(cx, cy),
                end_point=(cx + radius, cy),
                text=f"R{radius:.2f}",
                feature_label=label,
                face_id=face_id,
            ))

        elif label == "Chamfer":
            angle = params.get("angle", 45.0)
            offset = params.get("offset", 2.0)
            candidates.append(DimensionCandidate(
                dim_type="ANGLE",
                value=angle,
                start_point=(cx, cy),
                end_point=(cx + offset, cy + offset),
                text=f"{offset:.2f}×{angle:.0f}°",
                feature_label=label,
                face_id=face_id,
            ))

        return candidates

    @staticmethod
    def _overall_dims(centers: List[Tuple[float, float, float]]) -> List[DimensionCandidate]:
        """Generate overall bounding-box dimension candidates."""
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        zs = [c[2] for c in centers]
        if not xs:
            return []

        width = max(xs) - min(xs) if max(xs) != min(xs) else 10.0
        height = max(ys) - min(ys) if max(ys) != min(ys) else 10.0
        depth = max(zs) - min(zs) if max(zs) != min(zs) else 10.0
        my = (min(ys) + max(ys)) / 2

        return [
            DimensionCandidate(
                dim_type="LENGTH",
                value=width,
                start_point=(min(xs), my - height / 2 - 15),
                end_point=(max(xs), my - height / 2 - 15),
                text=f"{width:.2f}",
                feature_label="Overall",
                face_id=-1,
            ),
            DimensionCandidate(
                dim_type="WIDTH",
                value=height,
                start_point=(max(xs) + 15, min(ys)),
                end_point=(max(xs) + 15, max(ys)),
                text=f"{height:.2f}",
                feature_label="Overall",
                face_id=-1,
            ),
            DimensionCandidate(
                dim_type="DEPTH",
                value=depth,
                start_point=(max(xs) + 30, min(ys)),
                end_point=(max(xs) + 30, min(ys) + depth),
                text=f"{depth:.2f}",
                feature_label="Overall",
                face_id=-1,
            ),
        ]

    @staticmethod
    def _spacing_dims(cylinder_nodes: List) -> List[DimensionCandidate]:
        """Generate centre-to-centre spacing dimensions for hole/boss pairs."""
        candidates = []
        for i in range(len(cylinder_nodes)):
            for j in range(i + 1, len(cylinder_nodes)):
                ni = cylinder_nodes[i]
                nj = cylinder_nodes[j]
                import math
                dx = nj.center[0] - ni.center[0]
                dy = nj.center[1] - ni.center[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 1e-3:
                    continue
                candidates.append(DimensionCandidate(
                    dim_type="SPACING",
                    value=round(dist, 3),
                    start_point=(float(ni.center[0]), float(ni.center[1])),
                    end_point=(float(nj.center[0]), float(nj.center[1])),
                    text=f"{dist:.2f}",
                    feature_label="Spacing",
                    face_id=-1,
                ))
        return candidates
