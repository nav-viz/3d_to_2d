"""View prediction model – recommends orthographic views and sheet size."""
from __future__ import annotations

from typing import Dict, List, Tuple

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    logger.info("XGBoost available – using trained view predictor.")
except ImportError:
    XGB_AVAILABLE = False
    logger.info("XGBoost not available – using rule-based view predictor.")


class ViewPredictionModel:
    """
    Predicts which orthographic views to include in the drawing, the
    recommended drawing scale, and the target ISO sheet size.

    XGBoost classifiers are used when available; otherwise a rule-based
    approach selects views based on bounding-box aspect ratios and
    feature counts.
    """

    def __init__(self, model_path: str = "") -> None:
        self._log = get_logger(self.__class__.__name__)
        self._xgb_model: object = None
        if XGB_AVAILABLE and model_path:
            self._try_load(model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        bounding_box: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        feature_count: int,
        complexity_score: float,
    ) -> Dict:
        """
        Predict drawing configuration.

        Args:
            bounding_box: ``(min_point, max_point)`` axis-aligned bounding box.
            feature_count: Total number of detected manufacturing features.
            complexity_score: Scalar complexity metric from the geometry extractor.

        Returns:
            Dictionary with:
            - ``views``: list of view names (e.g., ``["FRONT", "TOP", "RIGHT"]``)
            - ``scale``: recommended drawing scale (e.g., ``1.0``)
            - ``sheet_size``: ISO sheet size string (e.g., ``"A3"``)
            - ``include_isometric``: bool
        """
        if XGB_AVAILABLE and self._xgb_model is not None:
            return self._predict_xgb(bounding_box, feature_count, complexity_score)
        return self._predict_rules(bounding_box, feature_count, complexity_score)

    # ------------------------------------------------------------------
    # XGBoost path
    # ------------------------------------------------------------------

    def _predict_xgb(
        self,
        bounding_box: Tuple,
        feature_count: int,
        complexity_score: float,
    ) -> Dict:
        """XGBoost-based prediction (stub – model not yet trained)."""
        # Fall through to rules; real implementation would call xgb_model.predict
        return self._predict_rules(bounding_box, feature_count, complexity_score)

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------

    def _predict_rules(
        self,
        bounding_box: Tuple,
        feature_count: int,
        complexity_score: float,
    ) -> Dict:
        """Determine views and sheet using geometry rules."""
        mn, mx = bounding_box
        dx = abs(mx[0] - mn[0])
        dy = abs(mx[1] - mn[1])
        dz = abs(mx[2] - mn[2])
        max_dim = max(dx, dy, dz, 1.0)

        # Always include front; add top/right when non-planar
        views: List[str] = ["FRONT"]
        if dz > 5.0:
            views.append("TOP")
        if dy > 5.0 or feature_count > 2:
            views.append("RIGHT")
        if complexity_score > 5.0 or feature_count > 4:
            views.append("ISOMETRIC")

        # Scale to fit A3 sheet (420 × 297 mm) with 20 mm margin
        usable_width = 400.0
        scale = min(1.0, usable_width / (max_dim + 1e-6))
        scale = self._round_to_standard_scale(scale)

        sheet_size = self._select_sheet(max_dim, scale)

        return {
            "views": views,
            "scale": scale,
            "sheet_size": sheet_size,
            "include_isometric": "ISOMETRIC" in views,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _round_to_standard_scale(raw: float) -> float:
        """Snap to a standard drawing scale."""
        standards = [0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
        return min(standards, key=lambda s: abs(s - raw))

    @staticmethod
    def _select_sheet(max_dim: float, scale: float) -> str:
        """Choose the smallest ISO sheet that fits the scaled model."""
        scaled = max_dim * scale
        sheet_sizes = settings.SHEET_SIZES
        for sheet in ("A4", "A3", "A2", "A1", "A0"):
            w, h = sheet_sizes[sheet]
            if scaled <= w - 40 and scaled <= h - 40:
                return sheet
        return "A0"

    def _try_load(self, path: str) -> None:
        """Attempt to load a saved XGBoost model."""
        import os
        if not os.path.exists(path):
            self._log.warning("XGBoost model not found at '%s'.", path)
            return
        try:
            self._xgb_model = xgb.Booster()
            self._xgb_model.load_model(path)  # type: ignore[union-attr]
            self._log.info("Loaded XGBoost view model from '%s'.", path)
        except Exception as exc:
            self._log.warning("Could not load XGBoost model: %s", exc)
