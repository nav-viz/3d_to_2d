"""Shared test fixtures for the CAD Drawing Generator test suite."""
from __future__ import annotations

import pytest

from src.step_parser.parse_step import ParsedCADModel, Face, Edge, Vertex, Surface
from src.brep_graph.graph_builder import BRepGraph, BRepNode, BRepEdge
from src.dimensioning.dimension_generator import DimensionCandidate


# ---------------------------------------------------------------------------
# ParsedCADModel fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_parsed_model() -> ParsedCADModel:
    """A minimal but representative ParsedCADModel for testing."""
    vertices = [
        Vertex(0, 0.0, 0.0, 0.0),
        Vertex(1, 100.0, 0.0, 0.0),
        Vertex(2, 100.0, 80.0, 0.0),
        Vertex(3, 0.0, 80.0, 0.0),
        Vertex(4, 0.0, 0.0, 30.0),
        Vertex(5, 100.0, 0.0, 30.0),
        Vertex(6, 100.0, 80.0, 30.0),
        Vertex(7, 0.0, 80.0, 30.0),
    ]

    edges = [
        Edge(0, 0, 1, "LINE", 100.0, (50.0, 0.0, 0.0)),
        Edge(1, 1, 2, "LINE", 80.0, (100.0, 40.0, 0.0)),
        Edge(2, 2, 3, "LINE", 100.0, (50.0, 80.0, 0.0)),
        Edge(3, 3, 0, "LINE", 80.0, (0.0, 40.0, 0.0)),
        Edge(4, 4, 5, "LINE", 100.0, (50.0, 0.0, 30.0)),
        Edge(5, 5, 6, "LINE", 80.0, (100.0, 40.0, 30.0)),
        Edge(6, 6, 7, "LINE", 100.0, (50.0, 80.0, 30.0)),
        Edge(7, 7, 4, "LINE", 80.0, (0.0, 40.0, 30.0)),
        Edge(8, 0, 4, "LINE", 30.0, (0.0, 0.0, 15.0)),
        Edge(9, 1, 5, "LINE", 30.0, (100.0, 0.0, 15.0)),
        Edge(10, 2, 6, "LINE", 30.0, (100.0, 80.0, 15.0)),
        Edge(11, 3, 7, "LINE", 30.0, (0.0, 80.0, 15.0)),
    ]

    faces = [
        Face(0, "PLANE", 8000.0, (0.0, 0.0, -1.0), (50.0, 40.0, 0.0),
             [0, 1, 2, 3], [0, 1, 2, 3], 0.0,
             ((0.0, 0.0, 0.0), (100.0, 80.0, 30.0))),
        Face(1, "PLANE", 8000.0, (0.0, 0.0, 1.0), (50.0, 40.0, 30.0),
             [4, 5, 6, 7], [4, 5, 6, 7], 0.0,
             ((0.0, 0.0, 0.0), (100.0, 80.0, 30.0))),
        Face(2, "PLANE", 3000.0, (0.0, -1.0, 0.0), (50.0, 0.0, 15.0),
             [0, 1, 5, 4], [0, 8, 4, 9], 0.0,
             ((0.0, 0.0, 0.0), (100.0, 80.0, 30.0))),
        Face(3, "PLANE", 2400.0, (1.0, 0.0, 0.0), (100.0, 40.0, 15.0),
             [1, 2, 6, 5], [1, 9, 5, 10], 0.0,
             ((0.0, 0.0, 0.0), (100.0, 80.0, 30.0))),
        Face(4, "PLANE", 3000.0, (0.0, 1.0, 0.0), (50.0, 80.0, 15.0),
             [2, 3, 7, 6], [2, 10, 6, 11], 0.0,
             ((0.0, 0.0, 0.0), (100.0, 80.0, 30.0))),
        Face(5, "PLANE", 2400.0, (-1.0, 0.0, 0.0), (0.0, 40.0, 15.0),
             [3, 0, 4, 7], [3, 11, 7, 8], 0.0,
             ((0.0, 0.0, 0.0), (100.0, 80.0, 30.0))),
        Face(6, "CYLINDER", 942.0, (0.0, 0.0, 1.0), (50.0, 40.0, 15.0),
             [], [], 0.06,
             ((0.0, 0.0, 0.0), (100.0, 80.0, 30.0))),
    ]

    surfaces = [Surface(f.id, f.surface_type) for f in faces]

    return ParsedCADModel(
        faces=faces,
        edges=edges,
        vertices=vertices,
        surfaces=surfaces,
        source_file="test_model.step",
        bounding_box=((0.0, 0.0, 0.0), (100.0, 80.0, 30.0)),
        metadata={"source": "test"},
    )


# ---------------------------------------------------------------------------
# BRepGraph fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_brep_graph() -> BRepGraph:
    """A small BRepGraph with 4 nodes and 4 edges for graph tests."""
    nodes = {
        0: BRepNode(0, "PLANE", 5000.0, (0.0, 0.0, 1.0), (50.0, 40.0, 0.0), 0.0,
                    ((0.0, 0.0, 0.0), (100.0, 80.0, 0.0))),
        1: BRepNode(1, "PLANE", 3000.0, (0.0, -1.0, 0.0), (50.0, 0.0, 15.0), 0.0,
                    ((0.0, 0.0, 0.0), (100.0, 0.0, 30.0))),
        2: BRepNode(2, "PLANE", 2400.0, (1.0, 0.0, 0.0), (100.0, 40.0, 15.0), 0.0,
                    ((100.0, 0.0, 0.0), (100.0, 80.0, 30.0))),
        3: BRepNode(3, "CYLINDER", 942.0, (0.0, 0.0, 1.0), (50.0, 40.0, 15.0), 0.06,
                    ((40.0, 30.0, 0.0), (60.0, 50.0, 30.0))),
    }
    edges = {
        0: BRepEdge(0, 0, 1, [], 1.5708),
        1: BRepEdge(1, 1, 2, [], 1.5708),
        2: BRepEdge(2, 2, 0, [], 1.5708),
        3: BRepEdge(3, 0, 3, [], 0.0),
    }
    adjacency = {
        0: [1, 2, 3],
        1: [0, 2],
        2: [1, 0],
        3: [0],
    }
    return BRepGraph(nodes=nodes, edges=edges, adjacency=adjacency)


# ---------------------------------------------------------------------------
# DimensionCandidate fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_dimensions() -> list:
    """A small set of dimension candidates for testing."""
    return [
        DimensionCandidate(
            id="d001",
            dim_type="DIAMETER",
            value=20.0,
            start_point=(30.0, 30.0),
            end_point=(70.0, 30.0),
            text="⌀20.00",
            feature_label="Hole",
            face_id=6,
        ),
        DimensionCandidate(
            id="d002",
            dim_type="LENGTH",
            value=100.0,
            start_point=(0.0, -10.0),
            end_point=(100.0, -10.0),
            text="100.00",
            feature_label="Overall",
            face_id=-1,
        ),
        DimensionCandidate(
            id="d003",
            dim_type="WIDTH",
            value=80.0,
            start_point=(110.0, 0.0),
            end_point=(110.0, 80.0),
            text="80.00",
            feature_label="Overall",
            face_id=-1,
        ),
        DimensionCandidate(
            id="d004",
            dim_type="DEPTH",
            value=30.0,
            start_point=(120.0, 0.0),
            end_point=(120.0, 30.0),
            text="↧30.00",
            feature_label="Overall",
            face_id=-1,
        ),
    ]
