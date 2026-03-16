"""BRep graph builder."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.step_parser.parse_step import ParsedCADModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BRepNode:
    """A single node in the BRep graph, corresponding to one face."""
    id: int
    surface_type: str
    area: float
    normal: Tuple[float, float, float]
    center: Tuple[float, float, float]
    curvature: float
    bounding_box: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    feature_label: str = "Unknown"
    feature_params: dict = field(default_factory=dict)


@dataclass
class BRepEdge:
    """A graph edge representing face–face adjacency."""
    id: int
    source_node_id: int
    target_node_id: int
    shared_edge_ids: List[int] = field(default_factory=list)
    dihedral_angle: float = 0.0
    edge_type: str = "ADJACENCY"


@dataclass
class BRepGraph:
    """
    Graph representation of a solid's boundary representation topology.

    Attributes:
        nodes: Face nodes keyed by node id.
        edges: Adjacency edges (face–face) keyed by edge id.
        adjacency: Adjacency list mapping node id to neighbour node ids.
    """
    nodes: Dict[int, BRepNode] = field(default_factory=dict)
    edges: Dict[int, BRepEdge] = field(default_factory=dict)
    adjacency: Dict[int, List[int]] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    # Convenience properties ---------------------------------------------------

    @property
    def num_nodes(self) -> int:
        """Number of face nodes in the graph."""
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        """Number of adjacency edges in the graph."""
        return len(self.edges)

    def node_list(self) -> List[BRepNode]:
        """Return nodes as an ordered list."""
        return [self.nodes[k] for k in sorted(self.nodes)]


class GraphBuilder:
    """
    Builds a :class:`BRepGraph` from a :class:`ParsedCADModel`.

    The graph encodes face–face adjacency by detecting shared edge indices.
    """

    def __init__(self) -> None:
        self._log = get_logger(self.__class__.__name__)

    def build_graph(self, parsed_model: ParsedCADModel) -> BRepGraph:
        """
        Construct a BRep graph from a parsed CAD model.

        Args:
            parsed_model: Output of :class:`~src.step_parser.StepParser`.

        Returns:
            A :class:`BRepGraph` with nodes, edges, and adjacency list.
        """
        self._log.info("Building BRep graph …")
        nodes = self._build_nodes(parsed_model)
        graph_edges, adjacency = self._build_adjacency(parsed_model, nodes)

        graph = BRepGraph(
            nodes=nodes,
            edges=graph_edges,
            adjacency=adjacency,
            metadata={
                "source_file": parsed_model.source_file,
                "num_faces": len(parsed_model.faces),
                "num_edges": len(parsed_model.edges),
            },
        )
        self._log.info(
            "BRep graph: %d nodes, %d edges.",
            graph.num_nodes,
            graph.num_edges,
        )
        return graph

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_nodes(model: ParsedCADModel) -> Dict[int, BRepNode]:
        """Convert model faces to graph nodes."""
        nodes: Dict[int, BRepNode] = {}
        for face in model.faces:
            nodes[face.id] = BRepNode(
                id=face.id,
                surface_type=face.surface_type,
                area=face.area,
                normal=face.normal,
                center=face.center,
                curvature=face.curvature,
                bounding_box=face.bounding_box,
            )
        return nodes

    @staticmethod
    def _build_adjacency(
        model: ParsedCADModel,
        nodes: Dict[int, BRepNode],
    ) -> Tuple[Dict[int, BRepEdge], Dict[int, List[int]]]:
        """
        Detect face–face adjacency through shared edge indices.

        Returns:
            Tuple of (graph_edges dict, adjacency list dict).
        """
        # Map each edge id to the faces that reference it
        edge_to_faces: Dict[int, List[int]] = {}
        for face in model.faces:
            for eid in face.edges:
                edge_to_faces.setdefault(eid, []).append(face.id)

        graph_edges: Dict[int, BRepEdge] = {}
        adjacency: Dict[int, List[int]] = {nid: [] for nid in nodes}
        brep_eid = 0

        seen_pairs: set = set()
        for eid, face_ids in edge_to_faces.items():
            if len(face_ids) < 2:
                continue
            # All pairs of faces sharing this edge
            for i in range(len(face_ids)):
                for j in range(i + 1, len(face_ids)):
                    src, tgt = face_ids[i], face_ids[j]
                    pair = (min(src, tgt), max(src, tgt))
                    if pair in seen_pairs:
                        # Just append the shared edge
                        for ge in graph_edges.values():
                            if (ge.source_node_id, ge.target_node_id) == pair or \
                               (ge.target_node_id, ge.source_node_id) == pair:
                                ge.shared_edge_ids.append(eid)
                                break
                        continue
                    seen_pairs.add(pair)

                    dihedral = GraphBuilder._dihedral_angle(
                        nodes[src].normal, nodes[tgt].normal
                    )
                    brep_edge = BRepEdge(
                        id=brep_eid,
                        source_node_id=src,
                        target_node_id=tgt,
                        shared_edge_ids=[eid],
                        dihedral_angle=dihedral,
                    )
                    graph_edges[brep_eid] = brep_edge
                    brep_eid += 1

                    if tgt not in adjacency[src]:
                        adjacency[src].append(tgt)
                    if src not in adjacency[tgt]:
                        adjacency[tgt].append(src)

        # For models with no shared edges (e.g., pure synthetic), add spatial
        # adjacency based on proximity.
        if not graph_edges and len(nodes) > 1:
            graph_edges, adjacency = GraphBuilder._proximity_adjacency(nodes)

        return graph_edges, adjacency

    @staticmethod
    def _dihedral_angle(
        n1: Tuple[float, float, float],
        n2: Tuple[float, float, float],
    ) -> float:
        """Compute the dihedral angle (radians) between two face normals."""
        dot = sum(a * b for a, b in zip(n1, n2))
        mag1 = math.sqrt(sum(x ** 2 for x in n1))
        mag2 = math.sqrt(sum(x ** 2 for x in n2))
        if mag1 < 1e-10 or mag2 < 1e-10:
            return 0.0
        return math.acos(max(-1.0, min(1.0, dot / (mag1 * mag2))))

    @staticmethod
    def _proximity_adjacency(
        nodes: Dict[int, BRepNode],
    ) -> Tuple[Dict[int, BRepEdge], Dict[int, List[int]]]:
        """
        Build adjacency edges based on Euclidean proximity of face centres
        when no shared topological edges are available.
        """
        node_list = list(nodes.values())
        graph_edges: Dict[int, BRepEdge] = {}
        adjacency: Dict[int, List[int]] = {nid: [] for nid in nodes}
        eid = 0

        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                ni, nj = node_list[i], node_list[j]
                dist = math.sqrt(
                    sum((a - b) ** 2 for a, b in zip(ni.center, nj.center))
                )
                if dist < 120.0:  # heuristic threshold
                    dihedral = GraphBuilder._dihedral_angle(ni.normal, nj.normal)
                    graph_edges[eid] = BRepEdge(
                        id=eid,
                        source_node_id=ni.id,
                        target_node_id=nj.id,
                        dihedral_angle=dihedral,
                    )
                    adjacency[ni.id].append(nj.id)
                    adjacency[nj.id].append(ni.id)
                    eid += 1

        return graph_edges, adjacency
