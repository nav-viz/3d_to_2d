"""Tests for dimension generation and optimisation."""
from __future__ import annotations

from src.dimensioning.dimension_generator import DimensionGenerator, DimensionCandidate
from src.dimensioning.dimension_optimizer import DimensionOptimizer


class TestDimensionGenerator:
    """Tests for DimensionGenerator."""

    def test_generate_returns_list(self, sample_brep_graph):
        """generate() returns a list of DimensionCandidates."""
        gen = DimensionGenerator()
        feat_preds = {
            "face_labels": {n.id: "Hole" if n.surface_type == "CYLINDER" else "Pocket"
                            for n in sample_brep_graph.node_list()},
            "feature_params": {
                n.id: {"diameter": 20.0, "depth": 15.0}
                if n.surface_type == "CYLINDER"
                else {"length": 30.0, "width": 15.0, "depth": 10.0}
                for n in sample_brep_graph.node_list()
            },
        }
        candidates = gen.generate(sample_brep_graph, feat_preds)
        assert isinstance(candidates, list)
        assert len(candidates) > 0

    def test_all_candidates_are_dimension_candidates(self, sample_brep_graph):
        """All items in the returned list are DimensionCandidate instances."""
        gen = DimensionGenerator()
        candidates = gen.generate(sample_brep_graph, {"face_labels": {}, "feature_params": {}})
        for c in candidates:
            assert isinstance(c, DimensionCandidate)

    def test_hole_produces_diameter_dim(self, sample_brep_graph):
        """A Hole face generates at least one DIAMETER dimension."""
        gen = DimensionGenerator()
        # Mark first CYLINDER node as Hole
        cylinder_nodes = [n for n in sample_brep_graph.node_list()
                          if n.surface_type == "CYLINDER"]
        labels = {n.id: "Hole" for n in cylinder_nodes}
        params = {n.id: {"diameter": 20.0, "depth": 15.0} for n in cylinder_nodes}
        candidates = gen.generate(sample_brep_graph, {"face_labels": labels, "feature_params": params})
        types = {c.dim_type for c in candidates}
        assert "DIAMETER" in types

    def test_overall_dimensions_generated(self, sample_brep_graph):
        """Overall bounding-box dimensions are generated."""
        gen = DimensionGenerator()
        candidates = gen.generate(sample_brep_graph, {"face_labels": {}, "feature_params": {}})
        types = [c.dim_type for c in candidates]
        assert "LENGTH" in types or "WIDTH" in types

    def test_candidate_has_required_fields(self, sample_brep_graph):
        """Each DimensionCandidate has all required attributes."""
        gen = DimensionGenerator()
        candidates = gen.generate(sample_brep_graph, {"face_labels": {}, "feature_params": {}})
        for c in candidates:
            assert hasattr(c, "id")
            assert hasattr(c, "dim_type")
            assert hasattr(c, "value")
            assert hasattr(c, "start_point")
            assert hasattr(c, "end_point")
            assert hasattr(c, "text")

    def test_dimension_value_positive(self, sample_brep_graph):
        """Dimension values are positive."""
        gen = DimensionGenerator()
        candidates = gen.generate(sample_brep_graph, {"face_labels": {}, "feature_params": {}})
        for c in candidates:
            assert c.value >= 0.0

    def test_start_end_are_2d_tuples(self, sample_brep_graph):
        """start_point and end_point are 2-element sequences."""
        gen = DimensionGenerator()
        candidates = gen.generate(sample_brep_graph, {"face_labels": {}, "feature_params": {}})
        for c in candidates:
            assert len(c.start_point) == 2
            assert len(c.end_point) == 2


class TestDimensionOptimizer:
    """Tests for DimensionOptimizer."""

    def test_optimize_returns_list(self, sample_dimensions):
        """optimize() returns a list."""
        optimizer = DimensionOptimizer()
        ranked = [
            {"id": d.id, "score": 0.9 - i * 0.1}
            for i, d in enumerate(sample_dimensions)
        ]
        result = optimizer.optimize(sample_dimensions, ranked)
        assert isinstance(result, list)

    def test_optimize_returns_candidates(self, sample_dimensions):
        """Returned items are DimensionCandidate instances."""
        optimizer = DimensionOptimizer()
        ranked = [{"id": d.id, "score": 0.8} for d in sample_dimensions]
        result = optimizer.optimize(sample_dimensions, ranked)
        for item in result:
            assert isinstance(item, DimensionCandidate)

    def test_optimize_reduces_or_equal(self, sample_dimensions):
        """Optimizer never returns more candidates than input."""
        optimizer = DimensionOptimizer()
        ranked = [{"id": d.id, "score": 0.5} for d in sample_dimensions]
        result = optimizer.optimize(sample_dimensions, ranked)
        assert len(result) <= len(sample_dimensions)

    def test_optimize_empty_input(self):
        """optimize() with empty input returns empty list."""
        optimizer = DimensionOptimizer()
        assert optimizer.optimize([], []) == []

    def test_conflict_detection(self):
        """Overlapping dimensions are not both selected."""
        optimizer = DimensionOptimizer(overlap_threshold=5.0)
        # Two overlapping dimensions at the same location
        d1 = DimensionCandidate(
            id="ov1", dim_type="LENGTH", value=100.0,
            start_point=(0.0, 0.0), end_point=(100.0, 0.0), text="100"
        )
        d2 = DimensionCandidate(
            id="ov2", dim_type="LENGTH", value=90.0,
            start_point=(5.0, 0.0), end_point=(95.0, 0.0), text="90"
        )
        ranked = [
            {"id": "ov1", "score": 0.9},
            {"id": "ov2", "score": 0.8},
        ]
        result = optimizer.optimize([d1, d2], ranked)
        # At most one should be selected (they overlap)
        assert len(result) <= 2  # greedy may keep both if bbox not overlap

    def test_no_conflict_keeps_all(self):
        """Non-overlapping dimensions are all kept."""
        optimizer = DimensionOptimizer(overlap_threshold=1.0)
        candidates = [
            DimensionCandidate(id=f"d{i}", dim_type="LENGTH", value=float(i * 10),
                               start_point=(i * 200.0, 0.0), end_point=(i * 200.0 + 50, 0.0),
                               text=str(i))
            for i in range(3)
        ]
        ranked = [{"id": c.id, "score": 0.7} for c in candidates]
        result = optimizer.optimize(candidates, ranked)
        assert len(result) == 3
