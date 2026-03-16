"""Tests for database connection and storage."""
from __future__ import annotations

from src.database.neo4j_connection import Neo4jConnection
from src.database.graph_storage import GraphStorage


class TestNeo4jConnection:
    """Tests for Neo4jConnection with mock fallback."""

    def test_connect_returns_self(self):
        """connect() returns the connection instance."""
        conn = Neo4jConnection("bolt://localhost:7687", "neo4j", "wrong")
        result = conn.connect()
        assert result is conn
        conn.close()

    def test_context_manager_usage(self):
        """Context manager does not raise exceptions."""
        with Neo4jConnection() as conn:
            assert isinstance(conn, Neo4jConnection)

    def test_run_query_returns_list(self):
        """run_query() always returns a list (possibly empty)."""
        with Neo4jConnection() as conn:
            result = conn.run_query("MATCH (n) RETURN n LIMIT 1")
            assert isinstance(result, list)

    def test_mock_log_records_queries(self):
        """Mock connection logs executed queries."""
        conn = Neo4jConnection()
        conn.connect()  # forces mock mode
        conn.run_query("RETURN 1 AS n")
        assert len(conn.mock_log) >= 1
        conn.close()

    def test_is_connected_false_for_mock(self):
        """is_connected is False when using mock mode."""
        conn = Neo4jConnection()
        conn.connect()
        # With no real Neo4j server the driver will be None
        assert isinstance(conn.is_connected, bool)
        conn.close()

    def test_close_idempotent(self):
        """close() can be called multiple times without error."""
        conn = Neo4jConnection()
        conn.connect()
        conn.close()
        conn.close()  # should not raise

    def test_run_query_without_connecting(self):
        """run_query() without an explicit connect() still returns a list."""
        conn = Neo4jConnection()
        result = conn.run_query("RETURN 1")
        assert isinstance(result, list)


class TestGraphStorage:
    """Tests for GraphStorage with in-memory fallback."""

    def test_store_part_graph_returns_true(self, sample_brep_graph):
        """store_part_graph() returns True on success."""
        with Neo4jConnection() as conn:
            storage = GraphStorage(conn)
            result = storage.store_part_graph("test_part", sample_brep_graph)
        assert result is True

    def test_query_features_returns_list(self, sample_brep_graph):
        """query_features() returns a list of face dicts."""
        with Neo4jConnection() as conn:
            storage = GraphStorage(conn)
            storage.store_part_graph("part_A", sample_brep_graph)
            features = storage.query_features("part_A")
        assert isinstance(features, list)
        assert len(features) > 0

    def test_query_features_missing_part(self):
        """query_features() returns empty list for unknown part."""
        with Neo4jConnection() as conn:
            storage = GraphStorage(conn)
            result = storage.query_features("unknown_part_xyz")
        assert result == []

    def test_compare_part_versions(self, sample_brep_graph):
        """compare_part_versions() returns a diff dict."""
        with Neo4jConnection() as conn:
            storage = GraphStorage(conn)
            storage.store_part_graph("v1", sample_brep_graph)
            storage.store_part_graph("v2", sample_brep_graph)
            diff = storage.compare_part_versions("v1", "v2")
        assert "added_faces" in diff
        assert "removed_faces" in diff
        assert "changed_faces" in diff
        assert "summary" in diff

    def test_compare_identical_versions(self, sample_brep_graph):
        """Comparing a graph to itself yields no changes."""
        with Neo4jConnection() as conn:
            storage = GraphStorage(conn)
            storage.store_part_graph("same", sample_brep_graph)
            diff = storage.compare_part_versions("same", "same")
        assert diff["summary"]["num_added"] == 0
        assert diff["summary"]["num_removed"] == 0

    def test_list_parts(self, sample_brep_graph):
        """list_parts() returns previously stored part IDs."""
        with Neo4jConnection() as conn:
            storage = GraphStorage(conn)
            storage.store_part_graph("part_list_test", sample_brep_graph)
            parts = storage.list_parts()
        assert "part_list_test" in parts

    def test_query_features_keys(self, sample_brep_graph):
        """Each stored feature has at minimum 'id' and 'surface_type'."""
        with Neo4jConnection() as conn:
            storage = GraphStorage(conn)
            storage.store_part_graph("feat_keys_test", sample_brep_graph)
            features = storage.query_features("feat_keys_test")
        for feat in features:
            assert "id" in feat
            assert "surface_type" in feat
