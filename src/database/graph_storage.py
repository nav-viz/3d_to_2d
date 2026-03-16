"""Graph storage layer with Neo4j backend and in-memory fallback."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from src.brep_graph.graph_builder import BRepGraph
from src.database.neo4j_connection import Neo4jConnection
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GraphStorage:
    """
    Persists and retrieves BRep graphs in Neo4j (or an in-memory dict when
    Neo4j is not available).

    Parameters
    ----------
    connection:
        An active or mock :class:`Neo4jConnection` instance.
    """

    def __init__(self, connection: Neo4jConnection) -> None:
        self._conn = connection
        self._memory: Dict[str, Dict] = {}  # fallback store
        self._log = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def store_part_graph(self, part_id: str, brep_graph: BRepGraph) -> bool:
        """
        Persist a BRep graph for a given part identifier.

        When the Neo4j connection is live the graph is written as a Cypher
        ``MERGE`` statement; otherwise it is stored in the in-memory dict.

        Args:
            part_id: Unique identifier for the part (e.g., filename stem).
            brep_graph: The BRep graph to store.

        Returns:
            ``True`` if the operation succeeded.
        """
        serialised = self._serialise_graph(brep_graph)
        self._memory[part_id] = serialised  # always cache in memory

        if self._conn.is_connected:
            try:
                self._write_to_neo4j(part_id, serialised)
                self._log.info("Stored graph for part '%s' in Neo4j.", part_id)
            except Exception as exc:
                self._log.warning("Neo4j write failed (%s); data in memory.", exc)
        else:
            self._log.info("Stored graph for part '%s' in memory.", part_id)
        return True

    def _write_to_neo4j(self, part_id: str, data: Dict) -> None:
        """Write serialised graph data to Neo4j."""
        self._conn.run_query(
            "MERGE (p:Part {id: $part_id}) SET p.data = $data",
            {"part_id": part_id, "data": json.dumps(data)},
        )
        for node in data.get("nodes", {}).values():
            self._conn.run_query(
                "MATCH (p:Part {id: $part_id}) "
                "MERGE (f:Face {id: $fid, part_id: $part_id}) "
                "SET f.surface_type = $stype, f.area = $area "
                "MERGE (p)-[:HAS_FACE]->(f)",
                {
                    "part_id": part_id,
                    "fid": node["id"],
                    "stype": node["surface_type"],
                    "area": node["area"],
                },
            )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def query_features(self, part_id: str) -> List[Dict]:
        """
        Retrieve stored face feature records for a part.

        Args:
            part_id: The part identifier used at storage time.

        Returns:
            List of face feature dictionaries.
        """
        if part_id in self._memory:
            nodes = self._memory[part_id].get("nodes", {})
            return list(nodes.values())

        if self._conn.is_connected:
            rows = self._conn.run_query(
                "MATCH (p:Part {id: $part_id})-[:HAS_FACE]->(f:Face) RETURN f",
                {"part_id": part_id},
            )
            return [r.get("f", {}) for r in rows]

        self._log.warning("No data found for part '%s'.", part_id)
        return []

    def compare_part_versions(
        self, part_id_1: str, part_id_2: str
    ) -> Dict[str, Any]:
        """
        Compare two stored part versions.

        Args:
            part_id_1: First part identifier.
            part_id_2: Second part identifier.

        Returns:
            Dictionary with keys ``added_faces``, ``removed_faces``,
            ``changed_faces``, and ``summary``.
        """
        features_1 = {f["id"]: f for f in self.query_features(part_id_1)}
        features_2 = {f["id"]: f for f in self.query_features(part_id_2)}

        ids_1 = set(features_1)
        ids_2 = set(features_2)

        added = list(ids_2 - ids_1)
        removed = list(ids_1 - ids_2)
        changed = [
            fid
            for fid in ids_1 & ids_2
            if features_1[fid].get("surface_type") != features_2[fid].get("surface_type")
            or abs(features_1[fid].get("area", 0) - features_2[fid].get("area", 0)) > 1e-3
        ]

        result = {
            "added_faces": added,
            "removed_faces": removed,
            "changed_faces": changed,
            "summary": {
                "num_added": len(added),
                "num_removed": len(removed),
                "num_changed": len(changed),
            },
        }
        self._log.info(
            "Comparison %s vs %s: +%d -%d ~%d faces.",
            part_id_1,
            part_id_2,
            len(added),
            len(removed),
            len(changed),
        )
        return result

    def list_parts(self) -> List[str]:
        """Return all stored part identifiers."""
        return list(self._memory.keys())

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialise_graph(graph: BRepGraph) -> Dict:
        """Convert a :class:`BRepGraph` to a JSON-serialisable dictionary."""
        nodes = {}
        for nid, node in graph.nodes.items():
            nodes[str(nid)] = {
                "id": node.id,
                "surface_type": node.surface_type,
                "area": node.area,
                "normal": list(node.normal),
                "center": list(node.center),
                "curvature": node.curvature,
                "feature_label": node.feature_label,
            }

        edges = {}
        for eid, edge in graph.edges.items():
            edges[str(eid)] = {
                "id": edge.id,
                "source": edge.source_node_id,
                "target": edge.target_node_id,
                "dihedral_angle": edge.dihedral_angle,
            }

        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": graph.metadata,
        }
