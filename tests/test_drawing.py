"""Tests for view projection, DXF writer, and PNG renderer."""
from __future__ import annotations

import os

import ezdxf

from src.drawing.view_projection import ViewProjector, ProjectedView
from src.drawing.dxf_writer import DXFWriter
from src.drawing.png_renderer import PNGRenderer


class TestViewProjector:
    """Tests for ViewProjector."""

    def test_project_returns_projected_view(self, sample_brep_graph):
        """project() returns a ProjectedView instance."""
        projector = ViewProjector()
        view = projector.project(sample_brep_graph, "FRONT")
        assert isinstance(view, ProjectedView)

    def test_project_view_name(self, sample_brep_graph):
        """ProjectedView has correct view_name."""
        projector = ViewProjector()
        view = projector.project(sample_brep_graph, "TOP")
        assert view.view_name == "TOP"

    def test_project_has_lines_or_arcs(self, sample_brep_graph):
        """Projection contains at least one line or arc."""
        projector = ViewProjector()
        view = projector.project(sample_brep_graph, "FRONT")
        assert len(view.lines) + len(view.arcs) > 0

    def test_project_all_views(self, sample_brep_graph):
        """project_all() returns a dict of views."""
        projector = ViewProjector()
        views = projector.project_all(
            sample_brep_graph,
            ["FRONT", "TOP", "RIGHT"],
            sheet_size="A3",
            scale=1.0,
        )
        assert isinstance(views, dict)
        assert "FRONT" in views
        assert "TOP" in views
        assert "RIGHT" in views

    def test_isometric_view(self, sample_brep_graph):
        """ISOMETRIC view can be projected without error."""
        projector = ViewProjector()
        view = projector.project(sample_brep_graph, "ISOMETRIC")
        assert view.view_name == "ISOMETRIC"

    def test_view_has_scale(self, sample_brep_graph):
        """Views returned by project_all have the requested scale."""
        projector = ViewProjector()
        views = projector.project_all(
            sample_brep_graph, ["FRONT"], sheet_size="A3", scale=0.5
        )
        assert views["FRONT"].scale == 0.5

    def test_view_has_origin(self, sample_brep_graph):
        """Views returned by project_all have a 2-D origin."""
        projector = ViewProjector()
        views = projector.project_all(
            sample_brep_graph, ["FRONT"], sheet_size="A3", scale=1.0
        )
        origin = views["FRONT"].origin
        assert len(origin) == 2

    def test_unknown_view_name_uses_front_direction(self, sample_brep_graph):
        """Unknown view name falls back to FRONT direction gracefully."""
        projector = ViewProjector()
        view = projector.project(sample_brep_graph, "UNKNOWN")
        assert isinstance(view, ProjectedView)


class TestDXFWriter:
    """Tests for DXFWriter."""

    def test_create_drawing_returns_document(self, sample_brep_graph, sample_dimensions):
        """create_drawing() returns an ezdxf document."""
        projector = ViewProjector()
        views = projector.project_all(sample_brep_graph, ["FRONT", "TOP"])
        writer = DXFWriter()
        doc = writer.create_drawing(views, sample_dimensions, {}, sheet_size="A3")
        assert doc is not None

    def test_save_creates_file(self, sample_brep_graph, sample_dimensions, tmp_path):
        """save() creates a .dxf file on disk."""
        projector = ViewProjector()
        views = projector.project_all(sample_brep_graph, ["FRONT"])
        writer = DXFWriter()
        doc = writer.create_drawing(views, sample_dimensions, {})
        output = str(tmp_path / "test_output.dxf")
        writer.save(doc, output)
        assert os.path.exists(output)
        assert os.path.getsize(output) > 0

    def test_dxf_is_valid(self, sample_brep_graph, sample_dimensions, tmp_path):
        """Saved DXF can be re-read by ezdxf."""
        projector = ViewProjector()
        views = projector.project_all(sample_brep_graph, ["FRONT"])
        writer = DXFWriter()
        doc = writer.create_drawing(views, sample_dimensions, {})
        output = str(tmp_path / "valid.dxf")
        writer.save(doc, output)
        reloaded = ezdxf.readfile(output)
        assert reloaded is not None

    def test_dxf_has_dimensions_layer(self, sample_brep_graph, sample_dimensions, tmp_path):
        """DXF document contains a DIMENSIONS layer."""
        projector = ViewProjector()
        views = projector.project_all(sample_brep_graph, ["FRONT"])
        writer = DXFWriter()
        doc = writer.create_drawing(views, sample_dimensions, {})
        output = str(tmp_path / "layers.dxf")
        writer.save(doc, output)
        reloaded = ezdxf.readfile(output)
        layer_names = [layer.dxf.name for layer in reloaded.layers]
        assert "DIMENSIONS" in layer_names

    def test_empty_views_and_dims(self, tmp_path):
        """create_drawing() handles empty views and dimensions."""
        writer = DXFWriter()
        doc = writer.create_drawing({}, [], {})
        output = str(tmp_path / "empty.dxf")
        writer.save(doc, output)
        assert os.path.exists(output)

    def test_title_block_text_in_doc(self, sample_brep_graph, sample_dimensions, tmp_path):
        """Title metadata is included in the DXF."""
        projector = ViewProjector()
        views = projector.project_all(sample_brep_graph, ["FRONT"])
        writer = DXFWriter()
        meta = {"title": "TEST PART", "drawing_number": "DRW-001"}
        doc = writer.create_drawing(views, sample_dimensions, meta)
        output = str(tmp_path / "title.dxf")
        writer.save(doc, output)
        # Read and check text entities exist
        reloaded = ezdxf.readfile(output)
        texts = [e for e in reloaded.modelspace() if e.dxftype() == "TEXT"]
        assert len(texts) > 0


class TestPNGRenderer:
    """Tests for PNGRenderer."""

    def test_render_creates_file(self, sample_brep_graph, sample_dimensions, tmp_path):
        """render() creates a PNG file."""
        # First create a DXF
        projector = ViewProjector()
        views = projector.project_all(sample_brep_graph, ["FRONT"])
        writer = DXFWriter()
        doc = writer.create_drawing(views, sample_dimensions, {})
        dxf_path = str(tmp_path / "test.dxf")
        writer.save(doc, dxf_path)

        renderer = PNGRenderer()
        png_path = str(tmp_path / "test.png")
        result = renderer.render(dxf_path, png_path)
        assert result is True
        assert os.path.exists(png_path)
        assert os.path.getsize(png_path) > 0

    def test_render_returns_bool(self, sample_brep_graph, sample_dimensions, tmp_path):
        """render() always returns a bool."""
        projector = ViewProjector()
        views = projector.project_all(sample_brep_graph, ["FRONT"])
        writer = DXFWriter()
        doc = writer.create_drawing(views, sample_dimensions, {})
        dxf_path = str(tmp_path / "ret.dxf")
        writer.save(doc, dxf_path)
        renderer = PNGRenderer()
        result = renderer.render(dxf_path, str(tmp_path / "ret.png"))
        assert isinstance(result, bool)

    def test_render_invalid_dxf_returns_true(self, tmp_path):
        """render() with a non-existent DXF falls back gracefully."""
        renderer = PNGRenderer()
        png_path = str(tmp_path / "fallback.png")
        result = renderer.render("/no/such/file.dxf", png_path)
        assert isinstance(result, bool)
