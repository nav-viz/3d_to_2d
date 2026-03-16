"""Tests for AI model inference with rule-based fallback."""
from __future__ import annotations

from src.ai_models.gnn_feature_model import GNNFeatureModel, FEATURE_CLASSES
from src.ai_models.feature_regression_model import FeatureRegressionModel
from src.ai_models.view_prediction_model import ViewPredictionModel
from src.ai_models.dimension_ranking_model import DimensionRankingModel


class TestGNNFeatureModel:
    """Tests for GNNFeatureModel (rule-based fallback)."""

    def test_predict_returns_dict(self, sample_brep_graph):
        """predict() returns a dictionary."""
        model = GNNFeatureModel()
        result = model.predict(sample_brep_graph)
        assert isinstance(result, dict)

    def test_predict_has_face_labels(self, sample_brep_graph):
        """Result contains 'face_labels' key."""
        model = GNNFeatureModel()
        result = model.predict(sample_brep_graph)
        assert "face_labels" in result

    def test_face_labels_count(self, sample_brep_graph):
        """face_labels dict has one entry per node."""
        model = GNNFeatureModel()
        result = model.predict(sample_brep_graph)
        assert len(result["face_labels"]) == sample_brep_graph.num_nodes

    def test_face_labels_valid_classes(self, sample_brep_graph):
        """All labels are from FEATURE_CLASSES."""
        model = GNNFeatureModel()
        result = model.predict(sample_brep_graph)
        for label in result["face_labels"].values():
            assert label in FEATURE_CLASSES

    def test_predict_has_summary(self, sample_brep_graph):
        """Result contains 'summary' dict."""
        model = GNNFeatureModel()
        result = model.predict(sample_brep_graph)
        assert "summary" in result
        assert isinstance(result["summary"], dict)

    def test_predict_has_probabilities(self, sample_brep_graph):
        """Result contains 'face_probabilities' dict."""
        model = GNNFeatureModel()
        result = model.predict(sample_brep_graph)
        assert "face_probabilities" in result

    def test_cylinder_face_classified_as_hole_or_boss(self, sample_brep_graph):
        """Cylindrical faces are classified as Hole or Boss."""
        model = GNNFeatureModel()
        result = model.predict(sample_brep_graph)
        for node in sample_brep_graph.node_list():
            if node.surface_type == "CYLINDER":
                label = result["face_labels"][node.id]
                assert label in ("Hole", "Boss")


class TestFeatureRegressionModel:
    """Tests for FeatureRegressionModel (geometric fallback)."""

    _FACE_DATA = {
        "surface_type": "CYLINDER",
        "area": 942.0,
        "normal": (0.0, 0.0, 1.0),
        "center": (50.0, 40.0, 15.0),
        "curvature": 0.06,
        "bounding_box": ((0.0, 0.0, 0.0), (100.0, 80.0, 30.0)),
    }

    def test_predict_hole_returns_dict(self):
        """predict() returns a dict for Hole label."""
        model = FeatureRegressionModel()
        result = model.predict(self._FACE_DATA, "Hole")
        assert isinstance(result, dict)

    def test_hole_has_diameter(self):
        """Hole prediction includes 'diameter'."""
        model = FeatureRegressionModel()
        result = model.predict(self._FACE_DATA, "Hole")
        assert "diameter" in result
        assert result["diameter"] > 0

    def test_hole_has_depth(self):
        """Hole prediction includes 'depth'."""
        model = FeatureRegressionModel()
        result = model.predict(self._FACE_DATA, "Hole")
        assert "depth" in result

    def test_pocket_prediction(self):
        """Pocket prediction includes length, width, depth."""
        model = FeatureRegressionModel()
        face_data = dict(self._FACE_DATA)
        face_data["surface_type"] = "PLANE"
        face_data["curvature"] = 0.0
        result = model.predict(face_data, "Pocket")
        assert "length" in result
        assert "width" in result
        assert "depth" in result

    def test_fillet_prediction(self):
        """Fillet prediction includes 'radius'."""
        model = FeatureRegressionModel()
        result = model.predict(self._FACE_DATA, "Fillet")
        assert "radius" in result

    def test_unknown_label(self):
        """Unknown label still returns a non-empty dict."""
        model = FeatureRegressionModel()
        result = model.predict(self._FACE_DATA, "Unknown")
        assert isinstance(result, dict)
        assert len(result) > 0


class TestViewPredictionModel:
    """Tests for ViewPredictionModel (rule-based fallback)."""

    _BB = ((0.0, 0.0, 0.0), (100.0, 80.0, 30.0))

    def test_predict_returns_dict(self):
        """predict() returns a dictionary."""
        model = ViewPredictionModel()
        result = model.predict(self._BB, 3, 5.0)
        assert isinstance(result, dict)

    def test_predict_has_views(self):
        """Result contains 'views' list."""
        model = ViewPredictionModel()
        result = model.predict(self._BB, 3, 5.0)
        assert "views" in result
        assert isinstance(result["views"], list)
        assert len(result["views"]) > 0

    def test_predict_has_scale(self):
        """Result contains a positive 'scale' value."""
        model = ViewPredictionModel()
        result = model.predict(self._BB, 3, 5.0)
        assert "scale" in result
        assert result["scale"] > 0

    def test_predict_has_sheet_size(self):
        """Result contains a recognised ISO sheet size."""
        model = ViewPredictionModel()
        result = model.predict(self._BB, 3, 5.0)
        assert result["sheet_size"] in ("A0", "A1", "A2", "A3", "A4")

    def test_front_view_always_included(self):
        """FRONT view is always in the result."""
        model = ViewPredictionModel()
        result = model.predict(self._BB, 1, 1.0)
        assert "FRONT" in result["views"]

    def test_high_complexity_includes_isometric(self):
        """High complexity triggers ISOMETRIC view."""
        model = ViewPredictionModel()
        result = model.predict(self._BB, 10, 15.0)
        assert "ISOMETRIC" in result["views"]


class TestDimensionRankingModel:
    """Tests for DimensionRankingModel (priority-based fallback)."""

    def test_rank_returns_list(self, sample_dimensions):
        """rank() returns a list."""
        model = DimensionRankingModel()
        dicts = [
            {
                "id": d.id,
                "dim_type": d.dim_type,
                "value": d.value,
                "start_point": d.start_point,
                "end_point": d.end_point,
                "feature_label": d.feature_label,
            }
            for d in sample_dimensions
        ]
        ranked = model.rank(dicts)
        assert isinstance(ranked, list)

    def test_rank_preserves_count(self, sample_dimensions):
        """rank() returns the same number of items."""
        model = DimensionRankingModel()
        dicts = [{"id": d.id, "dim_type": d.dim_type, "value": d.value,
                  "start_point": d.start_point, "end_point": d.end_point,
                  "feature_label": d.feature_label}
                 for d in sample_dimensions]
        ranked = model.rank(dicts)
        assert len(ranked) == len(sample_dimensions)

    def test_rank_has_score(self, sample_dimensions):
        """Each ranked item has a 'score' field."""
        model = DimensionRankingModel()
        dicts = [{"id": d.id, "dim_type": d.dim_type, "value": d.value,
                  "start_point": d.start_point, "end_point": d.end_point}
                 for d in sample_dimensions]
        ranked = model.rank(dicts)
        for item in ranked:
            assert "score" in item

    def test_rank_sorted_descending(self, sample_dimensions):
        """Ranked items are sorted in descending score order."""
        model = DimensionRankingModel()
        dicts = [{"id": d.id, "dim_type": d.dim_type, "value": d.value,
                  "start_point": d.start_point, "end_point": d.end_point}
                 for d in sample_dimensions]
        ranked = model.rank(dicts)
        scores = [r["score"] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_empty_returns_empty(self):
        """rank() with empty input returns empty list."""
        model = DimensionRankingModel()
        assert model.rank([]) == []
