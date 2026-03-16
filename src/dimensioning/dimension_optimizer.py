"""Dimension optimizer – removes redundant / overlapping candidates."""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

from src.dimensioning.dimension_generator import DimensionCandidate
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from ortools.sat.python import cp_model as _cp_model  # noqa: F401
    ORTOOLS_AVAILABLE = True
    logger.info("OR-Tools available – using ILP optimizer.")
except ImportError:
    ORTOOLS_AVAILABLE = False
    logger.info("OR-Tools not available – using greedy dimension optimizer.")


class DimensionOptimizer:
    """
    Selects the best non-conflicting subset of dimension candidates.

    Conflict = two candidates whose annotation lines overlap within
    ``overlap_threshold`` mm in drawing space.

    When OR-Tools is available the selection is formulated as an integer
    linear program (maximum-weight independent set); otherwise a greedy
    score-descending algorithm is used.
    """

    def __init__(self, overlap_threshold: float = settings.OVERLAP_THRESHOLD) -> None:
        self._threshold = overlap_threshold
        self._log = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        candidates: List[DimensionCandidate],
        ranked_scores: List[Dict],
    ) -> List[DimensionCandidate]:
        """
        Select the best non-overlapping subset of dimension candidates.

        Args:
            candidates: All generated :class:`DimensionCandidate` objects.
            ranked_scores: Scored list returned by
                :class:`~src.ai_models.DimensionRankingModel.rank`.

        Returns:
            Filtered list of selected :class:`DimensionCandidate` objects.
        """
        if not candidates:
            return []

        # Attach scores from the ranker
        score_map = {s["id"]: s.get("score", 0.5) for s in ranked_scores}
        for c in candidates:
            c.score = score_map.get(c.id, 0.5)

        # Sort descending by score
        sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)

        conflict_graph = self._build_conflict_graph(sorted_candidates)

        if ORTOOLS_AVAILABLE:
            selected = self._ilp_select(sorted_candidates, conflict_graph)
        else:
            selected = self._greedy_select(sorted_candidates, conflict_graph)

        # Apply MAX_DIMENSIONS cap
        selected = selected[: settings.MAX_DIMENSIONS]
        self._log.info(
            "Optimized %d → %d dimensions.", len(candidates), len(selected)
        )
        return selected

    # ------------------------------------------------------------------
    # Conflict graph
    # ------------------------------------------------------------------

    def _build_conflict_graph(
        self, candidates: List[DimensionCandidate]
    ) -> Dict[int, Set[int]]:
        """
        Build a conflict graph where nodes are candidate indices.

        Two candidates conflict if their annotation segments overlap within
        the configured threshold.
        """
        n = len(candidates)
        conflicts: Dict[int, Set[int]] = {i: set() for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if self._overlaps(candidates[i], candidates[j]):
                    conflicts[i].add(j)
                    conflicts[j].add(i)
        return conflicts

    def _overlaps(self, c1: DimensionCandidate, c2: DimensionCandidate) -> bool:
        """Check whether two dimension candidates overlap in 2-D space."""
        # Use bounding-box overlap heuristic
        bb1 = self._bbox(c1)
        bb2 = self._bbox(c2)
        return self._bbox_overlap(bb1, bb2, self._threshold)

    @staticmethod
    def _bbox(c: DimensionCandidate) -> Tuple[float, float, float, float]:
        """Return (min_x, min_y, max_x, max_y) bounding box of a candidate."""
        x0, y0 = c.start_point
        x1, y1 = c.end_point
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    @staticmethod
    def _bbox_overlap(
        bb1: Tuple[float, float, float, float],
        bb2: Tuple[float, float, float, float],
        tol: float,
    ) -> bool:
        """Return True when two axis-aligned bounding boxes overlap within tol."""
        return not (
            bb1[2] + tol < bb2[0]
            or bb2[2] + tol < bb1[0]
            or bb1[3] + tol < bb2[1]
            or bb2[3] + tol < bb1[1]
        )

    # ------------------------------------------------------------------
    # OR-Tools ILP solver
    # ------------------------------------------------------------------

    def _ilp_select(
        self,
        candidates: List[DimensionCandidate],
        conflicts: Dict[int, Set[int]],
    ) -> List[DimensionCandidate]:
        """Select a maximum-weight independent set via CP-SAT."""
        from ortools.sat.python import cp_model
        model = cp_model.CpModel()
        n = len(candidates)
        # Scale scores to integers (OR-Tools needs integer objectives)
        scale = 1000
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]
        for i, cand in enumerate(candidates):
            for j in conflicts[i]:
                if j > i:
                    model.AddAtMostOne([x[i], x[j]])
        model.Maximize(
            sum(int(c.score * scale) * x[i] for i, c in enumerate(candidates))
        )
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0
        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [candidates[i] for i in range(n) if solver.Value(x[i])]
        # Fallback to greedy if solver fails
        return self._greedy_select(candidates, conflicts)

    # ------------------------------------------------------------------
    # Greedy fallback
    # ------------------------------------------------------------------

    def _greedy_select(
        self,
        candidates: List[DimensionCandidate],
        conflicts: Dict[int, Set[int]],
    ) -> List[DimensionCandidate]:
        """Greedy maximum independent set selection (descending score)."""
        selected: List[DimensionCandidate] = []
        excluded: Set[int] = set()
        for i, candidate in enumerate(candidates):
            if i in excluded:
                continue
            selected.append(candidate)
            excluded.update(conflicts[i])
        return selected
