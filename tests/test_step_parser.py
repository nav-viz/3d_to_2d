"""Tests for STEP parser and geometry extractor."""
from __future__ import annotations

from src.step_parser.parse_step import (
    StepParser,
    ParsedCADModel,
    Face,
    Edge,
    Vertex,
    Surface,
)
from src.step_parser.geometry_extractor import GeometryExtractor


class TestStepParser:
    """Tests for StepParser."""

    def test_parse_returns_parsed_cad_model(self):
        """parse() without a file path returns a ParsedCADModel."""
        parser = StepParser()
        model = parser.parse("")
        assert isinstance(model, ParsedCADModel)

    def test_synthetic_model_has_faces(self):
        """Synthetic model contains at least one face."""
        parser = StepParser()
        model = parser.parse("")
        assert len(model.faces) > 0

    def test_synthetic_model_has_edges(self):
        """Synthetic model contains edges."""
        parser = StepParser()
        model = parser.parse("")
        assert len(model.edges) > 0

    def test_synthetic_model_has_vertices(self):
        """Synthetic model contains vertices."""
        parser = StepParser()
        model = parser.parse("")
        assert len(model.vertices) > 0

    def test_face_dataclass_fields(self):
        """Face dataclass contains expected fields."""
        face = Face(
            id=0,
            surface_type="PLANE",
            area=1000.0,
            normal=(0.0, 0.0, 1.0),
            center=(50.0, 40.0, 0.0),
        )
        assert face.id == 0
        assert face.surface_type == "PLANE"
        assert face.area == 1000.0
        assert face.normal == (0.0, 0.0, 1.0)

    def test_edge_dataclass_fields(self):
        """Edge dataclass contains expected fields."""
        edge = Edge(1, 0, 1, "LINE", 100.0, (50.0, 0.0, 0.0))
        assert edge.id == 1
        assert edge.start_vertex_id == 0
        assert edge.end_vertex_id == 1
        assert edge.length == 100.0

    def test_vertex_coords_property(self):
        """Vertex.coords returns a (x, y, z) tuple."""
        v = Vertex(0, 1.0, 2.0, 3.0)
        assert v.coords == (1.0, 2.0, 3.0)

    def test_bounding_box_is_tuple(self, sample_parsed_model):
        """Model bounding_box is a nested tuple pair."""
        bb = sample_parsed_model.bounding_box
        assert len(bb) == 2
        assert len(bb[0]) == 3
        assert len(bb[1]) == 3

    def test_parse_nonexistent_file_returns_synthetic(self):
        """Parsing a non-existent file falls back to synthetic model."""
        parser = StepParser()
        model = parser.parse("/no/such/file.step")
        assert isinstance(model, ParsedCADModel)
        assert len(model.faces) > 0

    def test_surfaces_list_populated(self):
        """Synthetic model populates the surfaces list."""
        parser = StepParser()
        model = parser.parse("")
        assert len(model.surfaces) > 0
        assert all(isinstance(s, Surface) for s in model.surfaces)

    def test_metadata_source_key(self):
        """Synthetic model metadata contains 'source' key."""
        parser = StepParser()
        model = parser.parse("")
        assert "source" in model.metadata


class TestGeometryExtractor:
    """Tests for GeometryExtractor."""

    def test_extract_faces_returns_list(self, sample_parsed_model):
        """extract_faces() returns a list of dicts."""
        ext = GeometryExtractor(sample_parsed_model)
        faces = ext.extract_faces()
        assert isinstance(faces, list)
        assert len(faces) == len(sample_parsed_model.faces)

    def test_extract_faces_keys(self, sample_parsed_model):
        """Each face dict has required keys."""
        ext = GeometryExtractor(sample_parsed_model)
        for face in ext.extract_faces():
            assert "id" in face
            assert "surface_type" in face
            assert "area" in face
            assert "normal" in face

    def test_extract_edges_returns_list(self, sample_parsed_model):
        """extract_edges() returns a list of dicts."""
        ext = GeometryExtractor(sample_parsed_model)
        edges = ext.extract_edges()
        assert len(edges) == len(sample_parsed_model.edges)

    def test_extract_vertices_returns_list(self, sample_parsed_model):
        """extract_vertices() returns a list of dicts."""
        ext = GeometryExtractor(sample_parsed_model)
        vertices = ext.extract_vertices()
        assert len(vertices) == len(sample_parsed_model.vertices)

    def test_extract_bounding_box(self, sample_parsed_model):
        """extract_bounding_box() returns a 2-tuple of 3-tuples."""
        ext = GeometryExtractor(sample_parsed_model)
        bb = ext.extract_bounding_box()
        assert len(bb) == 2
        assert len(bb[0]) == 3
        assert len(bb[1]) == 3

    def test_bounding_box_min_le_max(self, sample_parsed_model):
        """Min components are ≤ max components."""
        ext = GeometryExtractor(sample_parsed_model)
        mn, mx = ext.extract_bounding_box()
        for lo, hi in zip(mn, mx):
            assert lo <= hi

    def test_extract_surface_types(self, sample_parsed_model):
        """extract_surface_types() returns a dict of counts."""
        ext = GeometryExtractor(sample_parsed_model)
        types = ext.extract_surface_types()
        assert isinstance(types, dict)
        assert "PLANE" in types

    def test_compute_complexity_score(self, sample_parsed_model):
        """Complexity score is a non-negative float."""
        ext = GeometryExtractor(sample_parsed_model)
        score = ext.compute_complexity_score()
        assert isinstance(score, float)
        assert score >= 0.0

    def test_extract_feature_regions(self, sample_parsed_model):
        """Feature regions are returned as a non-empty list."""
        ext = GeometryExtractor(sample_parsed_model)
        regions = ext.extract_feature_regions()
        assert isinstance(regions, list)
        assert len(regions) > 0
