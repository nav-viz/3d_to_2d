"""Tests for BRep graph builder and feature extractor."""
from __future__ import annotations

import numpy as np

from src.brep_graph.graph_builder import GraphBuilder, BRepGraph
from src.brep_graph.graph_features import GraphFeatureExtractor


class TestGraphBuilder:
    """Tests for GraphBuilder."""

    def test_build_graph_returns_brep_graph(self, sample_parsed_model):
        """build_graph() returns a BRepGraph instance."""
        builder = GraphBuilder()
        graph = builder.build_graph(sample_parsed_model)
        assert isinstance(graph, BRepGraph)

    def test_graph_has_correct_node_count(self, sample_parsed_model):
        """Number of nodes equals number of faces in the model."""
        builder = GraphBuilder()
        graph = builder.build_graph(sample_parsed_model)
        assert graph.num_nodes == len(sample_parsed_model.faces)

    def test_graph_has_edges(self, sample_parsed_model):
        """Graph contains at least one adjacency edge."""
        builder = GraphBuilder()
        graph = builder.build_graph(sample_parsed_model)
        assert graph.num_edges > 0

    def test_adjacency_lists_populated(self, sample_parsed_model):
        """Adjacency dict is populated for all nodes."""
        builder = GraphBuilder()
        graph = builder.build_graph(sample_parsed_model)
        assert len(graph.adjacency) == graph.num_nodes

    def test_brep_node_fields(self, sample_brep_graph):
        """BRepNode has all required fields."""
        node = next(iter(sample_brep_graph.nodes.values()))
        assert hasattr(node, "surface_type")
        assert hasattr(node, "area")
        assert hasattr(node, "normal")
        assert hasattr(node, "curvature")
        assert hasattr(node, "bounding_box")

    def test_brep_edge_fields(self, sample_brep_graph):
        """BRepEdge has all required fields."""
        edge = next(iter(sample_brep_graph.edges.values()))
        assert hasattr(edge, "source_node_id")
        assert hasattr(edge, "target_node_id")
        assert hasattr(edge, "dihedral_angle")

    def test_node_list_ordered(self, sample_brep_graph):
        """node_list() returns nodes in id order."""
        nodes = sample_brep_graph.node_list()
        ids = [n.id for n in nodes]
        assert ids == sorted(ids)

    def test_metadata_populated(self, sample_parsed_model):
        """Graph metadata is populated with source info."""
        builder = GraphBuilder()
        graph = builder.build_graph(sample_parsed_model)
        assert "num_faces" in graph.metadata

    def test_graph_num_nodes_property(self, sample_brep_graph):
        """num_nodes property returns the correct count."""
        assert sample_brep_graph.num_nodes == 4

    def test_graph_num_edges_property(self, sample_brep_graph):
        """num_edges property returns the correct count."""
        assert sample_brep_graph.num_edges == 4


class TestGraphFeatureExtractor:
    """Tests for GraphFeatureExtractor."""

    def test_node_features_shape(self, sample_brep_graph):
        """Node feature matrix has shape (num_nodes, 9)."""
        extractor = GraphFeatureExtractor(sample_brep_graph)
        feats = extractor.extract_node_features()
        assert feats.shape == (4, GraphFeatureExtractor.NODE_FEATURE_DIM)

    def test_node_features_dtype(self, sample_brep_graph):
        """Node features are float32."""
        extractor = GraphFeatureExtractor(sample_brep_graph)
        feats = extractor.extract_node_features()
        assert feats.dtype == np.float32

    def test_edge_features_shape(self, sample_brep_graph):
        """Edge feature matrix has shape (num_edges, 2)."""
        extractor = GraphFeatureExtractor(sample_brep_graph)
        feats = extractor.extract_edge_features()
        assert feats.shape == (4, GraphFeatureExtractor.EDGE_FEATURE_DIM)

    def test_edge_index_shape(self, sample_brep_graph):
        """Edge index has shape (2, num_edges)."""
        extractor = GraphFeatureExtractor(sample_brep_graph)
        idx = extractor.extract_edge_index()
        # Undirected: 2 * num_edges
        assert idx.shape[0] == 2
        assert idx.shape[1] == 2 * sample_brep_graph.num_edges

    def test_graph_statistics_keys(self, sample_brep_graph):
        """Graph statistics dict contains expected keys."""
        extractor = GraphFeatureExtractor(sample_brep_graph)
        stats = extractor.compute_graph_statistics()
        for key in ("num_nodes", "num_edges", "avg_degree", "mean_area"):
            assert key in stats

    def test_empty_graph_node_features(self):
        """Empty graph returns zero-row feature array."""
        graph = BRepGraph()
        extractor = GraphFeatureExtractor(graph)
        feats = extractor.extract_node_features()
        assert feats.shape == (0, GraphFeatureExtractor.NODE_FEATURE_DIM)

    def test_node_feature_values_bounded(self, sample_brep_graph):
        """Surface type encoding is in [0, 1]."""
        extractor = GraphFeatureExtractor(sample_brep_graph)
        feats = extractor.extract_node_features()
        # First column is surface type encoding in [0, 1]
        assert (feats[:, 0] >= 0.0).all()
        assert (feats[:, 0] <= 1.0).all()
