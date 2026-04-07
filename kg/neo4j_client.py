"""
kg/neo4j_client.py
==================
Robust Neo4j driver wrapper for the CAD Knowledge Graph.

Schema overview
---------------
Node types:
    (:CADModel   {model_id, source_file, num_faces, num_edges})
    (:Face       {face_id, model_id, surface_type, area})
    (:Edge       {edge_id, model_id, convexity})
    (:Feature    {feature_id, model_id, face_id, feature_type,
                  confidence, params_json})

Relationship types:
    (CADModel)-[:HAS_FACE]      →  (Face)
    (CADModel)-[:HAS_EDGE]      →  (Edge)
    (Face)    -[:ADJACENT_TO]   →  (Face)   {via: edge_id, convexity}
    (Face)    -[:PREDICTED_AS]  →  (Feature) {confidence, model_type}

Indices are created automatically on first connection.
"""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from typing import Any, Generator
from utils.logger import logger

try:
    from neo4j import GraphDatabase, Driver, Session
    _NEO4J_AVAILABLE = True
except ImportError:
    _NEO4J_AVAILABLE = False
    logger.warning(
        "neo4j driver not found ('pip install neo4j'). "
        "Neo4j features will be disabled."
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class Neo4jClient:
    """
    Thread-safe Neo4j session manager with schema bootstrap.

    Usage
    -----
    >>> client = Neo4jClient(uri="bolt://localhost:7687",
    ...                      user="neo4j", password="password")
    >>> with client.session() as s:
    ...     s.run("MATCH (n) RETURN count(n) AS cnt").single()
    >>> client.close()
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        **driver_kwargs: Any,
    ) -> None:
        self._uri = uri
        self._database = database
        self._driver: Any = None

        if not _NEO4J_AVAILABLE:
            logger.warning("Neo4jClient: driver unavailable – stub mode.")
            return

        try:
            self._driver: Driver = GraphDatabase.driver(
                uri,
                auth=(user, password),
                **driver_kwargs,
            )
            self._driver.verify_connectivity()
            logger.info(f"Neo4j connected: {uri}")
            self._create_indices()
        except Exception as exc:
            logger.error(f"Neo4j connection failed: {exc}")
            self._driver = None

    # ------------------------------------------------------------------
    # Context manager helpers
    # ------------------------------------------------------------------

    @contextmanager
    def session(self) -> Generator[Any, None, None]:
        """Yield a managed Neo4j session (no-op stub if driver unavailable)."""
        if self._driver is None:
            yield _StubSession()
            return
        with self._driver.session(database=self._database) as sess:
            yield sess

    def close(self) -> None:
        """Close the driver connection pool."""
        if self._driver is not None:
            self._driver.close()
            logger.info("Neo4j driver closed.")

    # ------------------------------------------------------------------
    # Schema / index creation
    # ------------------------------------------------------------------

    def _create_indices(self) -> None:
        """Idempotently create node label indices."""
        index_statements = [
            "CREATE INDEX cad_model_id IF NOT EXISTS FOR (n:CADModel) ON (n.model_id)",
            "CREATE INDEX face_face_id  IF NOT EXISTS FOR (n:Face)     ON (n.face_id, n.model_id)",
            "CREATE INDEX edge_edge_id  IF NOT EXISTS FOR (n:Edge)     ON (n.edge_id, n.model_id)",
            "CREATE INDEX feat_feat_id  IF NOT EXISTS FOR (n:Feature)  ON (n.feature_id)",
        ]
        with self.session() as sess:
            for stmt in index_statements:
                try:
                    sess.run(stmt)
                except Exception as exc:
                    logger.debug(f"Index creation skipped ({exc})")
        logger.debug("Neo4j indices ready.")

    # ------------------------------------------------------------------
    # High-level write helpers
    # ------------------------------------------------------------------

    def upsert_model(self, model_data: dict[str, Any]) -> None:
        """Create or update a CADModel node."""
        cypher = """
        MERGE (m:CADModel {model_id: $model_id})
        SET m.source_file = $source_file,
            m.num_faces   = $num_faces,
            m.num_edges   = $num_edges,
            m.num_vertices= $num_vertices
        """
        with self.session() as sess:
            sess.run(cypher, **model_data)

    def upsert_faces(self, model_id: str, faces: list[dict]) -> None:
        """Bulk upsert Face nodes and HAS_FACE relationships."""
        cypher = """
        UNWIND $rows AS row
        MERGE (f:Face {face_id: row.face_id, model_id: $model_id})
        SET f.surface_type = row.surface_type,
            f.area         = row.area
        WITH f, row
        MATCH (m:CADModel {model_id: $model_id})
        MERGE (m)-[:HAS_FACE]->(f)
        """
        rows = [
            {"face_id": f["id"],
             "surface_type": f.get("surface_type", "Unknown"),
             "area": f.get("area", 0.0)}
            for f in faces
        ]
        with self.session() as sess:
            sess.run(cypher, rows=rows, model_id=model_id)

    def upsert_edges(self, model_id: str, edges: list[dict]) -> None:
        """Bulk upsert Edge nodes."""
        cypher = """
        UNWIND $rows AS row
        MERGE (e:Edge {edge_id: row.edge_id, model_id: $model_id})
        SET e.convexity = row.convexity
        WITH e
        MATCH (m:CADModel {model_id: $model_id})
        MERGE (m)-[:HAS_EDGE]->(e)
        """
        rows = [
            {"edge_id": eg["id"],
             "convexity": eg.get("convexity", "unknown")}
            for eg in edges
        ]
        with self.session() as sess:
            sess.run(cypher, rows=rows, model_id=model_id)

    def upsert_adjacency(
        self,
        model_id: str,
        adjacency: list[tuple[int, int, str]],
    ) -> None:
        """
        Create ADJACENT_TO relationships between Face nodes.

        Parameters
        ----------
        adjacency : list of (face_id_src, face_id_dst, convexity_str)
        """
        cypher = """
        UNWIND $rows AS row
        MATCH (fa:Face {face_id: row.src, model_id: $model_id})
        MATCH (fb:Face {face_id: row.dst, model_id: $model_id})
        MERGE (fa)-[r:ADJACENT_TO {convexity: row.conv}]->(fb)
        """
        rows = [
            {"src": src, "dst": dst, "conv": conv}
            for src, dst, conv in adjacency
        ]
        with self.session() as sess:
            sess.run(cypher, rows=rows, model_id=model_id)

    def write_predictions(
        self,
        model_id: str,
        predictions: list[dict[str, Any]],
        model_type: str = "hgt",
    ) -> None:
        """
        Write GNN predictions as Feature nodes connected via PREDICTED_AS.

        Parameters
        ----------
        predictions : list of dicts with keys:
            face_id, feature_type, confidence, params (dict)
        """
        cypher = """
        UNWIND $rows AS row
        MATCH (f:Face {face_id: row.face_id, model_id: $model_id})
        MERGE (feat:Feature {feature_id: row.feature_id})
        SET feat.model_id     = $model_id,
            feat.face_id      = row.face_id,
            feat.feature_type = row.feature_type,
            feat.confidence   = row.confidence,
            feat.params_json  = row.params_json,
            feat.model_type   = $model_type
        MERGE (f)-[:PREDICTED_AS {confidence: row.confidence,
                                   model_type: $model_type}]->(feat)
        """
        rows = [
            {
                "feature_id": str(uuid.uuid4()),
                "face_id": p["face_id"],
                "feature_type": p["feature_type"],
                "confidence": float(p["confidence"]),
                "params_json": json.dumps(p.get("params", {})),
            }
            for p in predictions
        ]
        with self.session() as sess:
            sess.run(cypher, rows=rows, model_id=model_id,
                     model_type=model_type)
        logger.info(
            f"Wrote {len(predictions)} predictions to Neo4j "
            f"(model_id={model_id}, model_type={model_type})."
        )

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def get_features(self, model_id: str) -> list[dict]:
        """Retrieve all Feature nodes for a model."""
        cypher = """
        MATCH (feat:Feature {model_id: $model_id})
        RETURN feat.face_id      AS face_id,
               feat.feature_type AS feature_type,
               feat.confidence   AS confidence,
               feat.params_json  AS params_json
        ORDER BY feat.confidence DESC
        """
        with self.session() as sess:
            result = sess.run(cypher, model_id=model_id)
            return [
                {
                    "face_id": r["face_id"],
                    "feature_type": r["feature_type"],
                    "confidence": r["confidence"],
                    "params": json.loads(r["params_json"] or "{}"),
                }
                for r in result
            ]

    @staticmethod
    def from_config(cfg: dict) -> "Neo4jClient":
        """Construct a client from the ``neo4j`` section of default.yaml."""
        nc = cfg["neo4j"]
        return Neo4jClient(
            uri=nc["uri"],
            user=nc["user"],
            password=nc["password"],
            database=nc.get("database", "neo4j"),
            max_connection_lifetime=nc.get("max_connection_lifetime", 3600),
            max_connection_pool_size=nc.get("max_connection_pool_size", 50),
            connection_acquisition_timeout=nc.get(
                "connection_acquisition_timeout", 60
            ),
            encrypted=nc.get("encrypted", False),
        )


# ---------------------------------------------------------------------------
# Stub for when Neo4j is unavailable
# ---------------------------------------------------------------------------

class _StubSession:
    """No-op session used when Neo4j driver is not installed."""

    def run(self, *args: Any, **kwargs: Any) -> "_StubResult":  # noqa: ANN401
        logger.debug("[StubSession] run() called – ignoring (Neo4j unavailable).")
        return _StubResult()


class _StubResult:
    def single(self) -> None:
        return None

    def __iter__(self):  # noqa: ANN204
        return iter([])
