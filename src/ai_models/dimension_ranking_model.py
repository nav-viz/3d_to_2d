"""Dimension ranking model."""
from __future__ import annotations

import importlib.util
from typing import Any, Dict, List

from src.utils.logger import get_logger

logger = get_logger(__name__)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    logger.info("PyTorch available – MLP dimension ranker active.")
else:
    logger.info("PyTorch not available – using priority-based dimension ranker.")


class DimensionRankingModel:
    """
    Ranks dimension candidates by estimated engineering importance.

    A small MLP learns importance scores when PyTorch is available.
    Otherwise a priority table assigns scores based on dimension type
    and feature category.
    """

    # Priority scores for the rule-based fallback
    _PRIORITY: Dict[str, float] = {
        "DIAMETER": 1.0,
        "DEPTH": 0.9,
        "LENGTH": 0.85,
        "WIDTH": 0.8,
        "RADIUS": 0.75,
        "ANGLE": 0.7,
        "SPACING": 0.65,
        "GENERAL": 0.5,
    }

    _FEATURE_BOOST: Dict[str, float] = {
        "Hole": 0.2,
        "Pocket": 0.15,
        "Slot": 0.15,
        "Boss": 0.1,
        "Fillet": 0.05,
        "Chamfer": 0.05,
        "Rib": 0.1,
        "Unknown": 0.0,
    }

    # MLP input dimension
    _IN_DIM: int = 6  # [dim_type_enc, value_norm, start_x, start_y, end_x, end_y]

    def __init__(self, model_path: str = "") -> None:
        self._log = get_logger(self.__class__.__name__)
        self._mlp: Any = None
        if TORCH_AVAILABLE:
            self._mlp = self._build_mlp()
            if model_path:
                self._try_load(model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(self, dimension_candidates: List[Dict]) -> List[Dict]:
        """
        Score and rank a list of dimension candidates.

        Args:
            dimension_candidates: List of candidate dicts as returned by
                :class:`~src.dimensioning.DimensionGenerator`.
                Each candidate must contain at minimum: ``id``,
                ``dim_type``, ``value``, ``start_point``, ``end_point``.

        Returns:
            Candidates with an added ``score`` key, sorted descending.
        """
        if not dimension_candidates:
            return []

        if TORCH_AVAILABLE and self._mlp is not None:
            scored = self._score_mlp(dimension_candidates)
        else:
            scored = self._score_priority(dimension_candidates)

        scored.sort(key=lambda c: c["score"], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # MLP path
    # ------------------------------------------------------------------

    def _build_mlp(self):
        """Build a compact MLP scorer."""
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(self._IN_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def _score_mlp(self, candidates: List[Dict]) -> List[Dict]:
        """Score candidates with the MLP."""
        import torch
        import numpy as np
        features = np.array(
            [self._candidate_to_vec(c) for c in candidates], dtype=np.float32
        )
        x = torch.from_numpy(features)
        with torch.no_grad():
            scores = self._mlp(x).squeeze(-1).numpy()  # type: ignore[union-attr]
        result = []
        for c, s in zip(candidates, scores):
            c = dict(c)
            c["score"] = float(s)
            result.append(c)
        return result

    # ------------------------------------------------------------------
    # Priority-based fallback
    # ------------------------------------------------------------------

    def _score_priority(self, candidates: List[Dict]) -> List[Dict]:
        """Score candidates using a priority table."""
        result = []
        for candidate in candidates:
            c = dict(candidate)
            base = self._PRIORITY.get(c.get("dim_type", "GENERAL").upper(), 0.5)
            boost = self._FEATURE_BOOST.get(c.get("feature_label", "Unknown"), 0.0)
            # Slight boost for larger dimensions (more critical)
            val = float(c.get("value", 0))
            size_boost = min(val / 1000.0, 0.1)
            c["score"] = min(1.0, base + boost + size_boost)
            result.append(c)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _candidate_to_vec(c: Dict) -> List[float]:
        """Encode a candidate as a fixed-length feature vector."""
        dim_type_enc = {
            "DIAMETER": 0, "DEPTH": 1, "LENGTH": 2,
            "WIDTH": 3, "RADIUS": 4, "ANGLE": 5,
            "SPACING": 6, "GENERAL": 7,
        }.get(str(c.get("dim_type", "GENERAL")).upper(), 7) / 7.0

        val = float(c.get("value", 0)) / 200.0
        sp = c.get("start_point", (0, 0))
        ep = c.get("end_point", (0, 0))
        return [
            dim_type_enc, val,
            float(sp[0]) / 200.0, float(sp[1]) / 200.0,
            float(ep[0]) / 200.0, float(ep[1]) / 200.0,
        ]

    def _try_load(self, path: str) -> None:
        """Load saved weights if available."""
        import os
        import torch
        if not os.path.exists(path):
            return
        try:
            self._mlp.load_state_dict(torch.load(path, map_location="cpu"))
            self._log.info("Loaded ranking model from '%s'.", path)
        except Exception as exc:
            self._log.warning("Could not load ranking model: %s", exc)
