"""Feature parameter regression model."""
from __future__ import annotations

import importlib.util
import math
from typing import Any, Dict

from src.utils.logger import get_logger

logger = get_logger(__name__)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    logger.info("PyTorch available – MLP regression active.")
else:
    logger.info("PyTorch not available – using geometric computation fallback.")


class FeatureRegressionModel:
    """
    Predicts manufacturing feature parameters from face geometry data.

    Given a face's surface type and a predicted feature label, outputs
    geometric parameters:
    - Hole / Boss: ``diameter``, ``depth``, ``axis``, ``center``
    - Pocket: ``length``, ``width``, ``depth``, ``center``
    - Slot: ``length``, ``width``, ``depth``, ``axis``
    - Fillet / Chamfer: ``radius`` / ``angle``

    A 3-layer MLP is used when PyTorch is available; otherwise parameters
    are derived directly from bounding-box geometry.
    """

    # Input feature dimension (must match GraphFeatureExtractor.NODE_FEATURE_DIM)
    _IN_DIM: int = 9
    _HIDDEN_DIM: int = 64
    _OUT_DIM: int = 6  # [diameter, depth, length, width, angle, radius]

    def __init__(self, model_path: str = "") -> None:
        self._log = get_logger(self.__class__.__name__)
        self._model: Any = None
        if TORCH_AVAILABLE:
            self._model = self._build_mlp()
            if model_path:
                self._try_load(model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, face_data: Dict, feature_label: str) -> Dict:
        """
        Predict feature parameters for a single face.

        Args:
            face_data: Dictionary containing at minimum ``area``,
                       ``normal``, ``center``, ``bounding_box``,
                       ``curvature``, ``surface_type``.
            feature_label: Predicted feature class (e.g., ``"Hole"``).

        Returns:
            Parameter dictionary relevant to the feature class.
        """
        if TORCH_AVAILABLE and self._model is not None:
            return self._predict_mlp(face_data, feature_label)
        return self._predict_geometric(face_data, feature_label)

    # ------------------------------------------------------------------
    # MLP path
    # ------------------------------------------------------------------

    def _build_mlp(self):
        """Build a 3-layer MLP regressor."""
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(self._IN_DIM, self._HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(self._HIDDEN_DIM, self._HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(self._HIDDEN_DIM, self._OUT_DIM),
        )

    def _predict_mlp(self, face_data: Dict, feature_label: str) -> Dict:
        """Run the MLP and map outputs to the feature parameter dict."""
        import torch
        x = self._face_to_tensor(face_data)
        with torch.no_grad():
            out = self._model(x).squeeze(0).numpy()  # type: ignore[union-attr]
        diameter = abs(float(out[0])) * 100.0 + 5.0
        depth = abs(float(out[1])) * 50.0 + 2.0
        length = abs(float(out[2])) * 100.0 + 10.0
        width = abs(float(out[3])) * 50.0 + 5.0
        return self._package_params(
            feature_label, face_data, diameter, depth, length, width
        )

    def _face_to_tensor(self, face_data: Dict):
        """Convert face_data dict to a (1, 9) tensor."""
        import torch
        from src.brep_graph.graph_features import SURFACE_TYPE_MAP
        stype = SURFACE_TYPE_MAP.get(
            str(face_data.get("surface_type", "PLANE")).upper(), 6
        )
        n = face_data.get("normal", (0, 0, 1))
        c = face_data.get("center", (0, 0, 0))
        vec = [
            stype / len(SURFACE_TYPE_MAP),
            face_data.get("area", 0) / 10000.0,
            float(n[0]), float(n[1]), float(n[2]),
            face_data.get("curvature", 0),
            float(c[0]) / 1000.0,
            float(c[1]) / 1000.0,
            float(c[2]) / 1000.0,
        ]
        return torch.tensor([vec], dtype=torch.float32)

    # ------------------------------------------------------------------
    # Geometric fallback
    # ------------------------------------------------------------------

    def _predict_geometric(self, face_data: Dict, feature_label: str) -> Dict:
        """Derive feature parameters from bounding-box geometry."""
        bb = face_data.get("bounding_box", ((0, 0, 0), (10, 10, 10)))
        mn, mx = bb
        dx = abs(mx[0] - mn[0])
        dy = abs(mx[1] - mn[1])
        dz = abs(mx[2] - mn[2])

        area = float(face_data.get("area", 1.0))
        curvature = float(face_data.get("curvature", 0.0))

        if curvature > 1e-6:
            diameter = max(2.0, 2.0 / (curvature + 1e-9))
            diameter = min(diameter, max(dx, dy, dz))
        else:
            diameter = 2.0 * math.sqrt(area / math.pi) if area > 0 else 10.0

        depth = min(dx, dy, dz) if min(dx, dy, dz) > 0 else 10.0
        length = max(dx, dy, dz)
        width = sorted([dx, dy, dz])[1] if sorted([dx, dy, dz])[1] > 0 else 5.0

        return self._package_params(feature_label, face_data, diameter, depth, length, width)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _package_params(
        label: str,
        face_data: Dict,
        diameter: float,
        depth: float,
        length: float,
        width: float,
    ) -> Dict:
        """Map computed scalars to a label-specific parameter dict."""
        center = face_data.get("center", (0.0, 0.0, 0.0))
        normal = face_data.get("normal", (0.0, 0.0, 1.0))

        if label in ("Hole",):
            return {
                "diameter": round(diameter, 3),
                "depth": round(depth, 3),
                "axis": normal,
                "center": center,
            }
        if label in ("Boss",):
            return {
                "diameter": round(diameter, 3),
                "height": round(depth, 3),
                "axis": normal,
                "center": center,
            }
        if label in ("Pocket",):
            return {
                "length": round(length, 3),
                "width": round(width, 3),
                "depth": round(depth, 3),
                "center": center,
            }
        if label in ("Slot",):
            return {
                "length": round(length, 3),
                "width": round(width, 3),
                "depth": round(depth, 3),
                "axis": normal,
            }
        if label in ("Fillet",):
            return {"radius": round(diameter / 2, 3), "center": center}
        if label in ("Chamfer",):
            return {"angle": 45.0, "offset": round(depth, 3)}
        if label in ("Rib",):
            return {
                "length": round(length, 3),
                "width": round(width, 3),
                "height": round(depth, 3),
            }
        return {"length": round(length, 3), "width": round(width, 3)}

    def _try_load(self, path: str) -> None:
        """Attempt to load saved weights."""
        import os
        import torch
        if not os.path.exists(path):
            return
        try:
            self._model.load_state_dict(torch.load(path, map_location="cpu"))
            self._log.info("Loaded regression model from '%s'.", path)
        except Exception as exc:
            self._log.warning("Could not load regression model: %s", exc)
