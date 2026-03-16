"""Feature extraction from a BRep graph for AI model input."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from src.brep_graph.graph_builder import BRepGraph, BRepNode
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Optional PyTorch Geometric
try:
    import torch
    from torch_geometric.data import Data as PyGData
    PYG_AVAILABLE = True
    logger.info("PyTorch Geometric available – using PyG feature format.")
except ImportError:
    PYG_AVAILABLE = False
    logger.info("PyTorch Geometric not available – using NumPy feature format.")

SURFACE_TYPE_MAP: Dict[str, int] = {
    "PLANE": 0,
    "CYLINDER": 1,
    "CONE": 2,
    "SPHERE": 3,
    "TORUS": 4,
    "BSPLINE": 5,
    "UNKNOWN": 6,
}


class GraphFeatureExtractor:
    """
    Extracts numerical feature matrices from a :class:`BRepGraph`.

    Node features (per face):
        [surface_type_enc, area_norm, nx, ny, nz, curvature,
         cx_norm, cy_norm, cz_norm]  → 9 dimensions

    Edge features (per adjacency edge):
        [dihedral_angle_norm, edge_type_enc]  → 2 dimensions
    """

    NODE_FEATURE_DIM: int = 9
    EDGE_FEATURE_DIM: int = 2

    def __init__(self, graph: BRepGraph) -> None:
        """
        Initialise the extractor.

        Args:
            graph: The BRep graph to extract features from.
        """
        self._graph = graph
        self._log = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_node_features(self) -> np.ndarray:
        """
        Extract a node feature matrix of shape ``(N, 9)``.

        Normalisation:
        - area: divided by 10 000 (typical face area in mm²)
        - centre coordinates: divided by 1 000 (typical part size in mm)
        - dihedral angles: divided by π

        Returns:
            NumPy float32 array of shape ``(num_nodes, NODE_FEATURE_DIM)``.
        """
        nodes = self._graph.node_list()
        if not nodes:
            return np.zeros((0, self.NODE_FEATURE_DIM), dtype=np.float32)

        features = np.zeros((len(nodes), self.NODE_FEATURE_DIM), dtype=np.float32)
        for i, node in enumerate(nodes):
            features[i] = self._node_to_vector(node)
        self._log.debug("Node feature matrix: %s", features.shape)
        return features

    def extract_edge_features(self) -> np.ndarray:
        """
        Extract an edge feature matrix of shape ``(E, 2)``.

        Returns:
            NumPy float32 array of shape ``(num_edges, EDGE_FEATURE_DIM)``.
        """
        edges = list(self._graph.edges.values())
        if not edges:
            return np.zeros((0, self.EDGE_FEATURE_DIM), dtype=np.float32)

        features = np.zeros((len(edges), self.EDGE_FEATURE_DIM), dtype=np.float32)
        for i, edge in enumerate(edges):
            features[i, 0] = float(edge.dihedral_angle) / math.pi
            features[i, 1] = 0.0  # single edge type for now
        self._log.debug("Edge feature matrix: %s", features.shape)
        return features

    def extract_edge_index(self) -> np.ndarray:
        """
        Return an edge index array of shape ``(2, E)`` (COO format).

        Row 0 = source node indices, Row 1 = target node indices.
        Both indices reference positions in the sorted node list.
        """
        node_id_to_pos = {nid: pos for pos, nid in enumerate(sorted(self._graph.nodes))}
        edges = list(self._graph.edges.values())
        if not edges:
            return np.zeros((2, 0), dtype=np.int64)

        src_list = [node_id_to_pos[e.source_node_id] for e in edges]
        tgt_list = [node_id_to_pos[e.target_node_id] for e in edges]
        # Undirected: add reverse edges
        index = np.array(
            [src_list + tgt_list, tgt_list + src_list], dtype=np.int64
        )
        return index

    def to_pyg_data(self) -> "Optional[PyGData]":  # type: ignore[name-defined]
        """
        Wrap features in a PyTorch Geometric ``Data`` object if available.

        Returns:
            A ``torch_geometric.data.Data`` instance, or ``None`` when PyG is
            not installed.
        """
        if not PYG_AVAILABLE:
            self._log.debug("PyG not available; returning None from to_pyg_data().")
            return None
        x = torch.from_numpy(self.extract_node_features())
        edge_index = torch.from_numpy(self.extract_edge_index())
        edge_attr = torch.from_numpy(self.extract_edge_features())
        # Duplicate edge_attr for reverse edges
        edge_attr_full = torch.cat([edge_attr, edge_attr], dim=0)
        return PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr_full)

    def compute_graph_statistics(self) -> Dict[str, float]:
        """
        Return basic graph statistics useful for downstream models.

        Keys: ``num_nodes``, ``num_edges``, ``avg_degree``,
        ``surface_type_entropy``, ``mean_area``, ``mean_curvature``.
        """
        nodes = self._graph.node_list()
        num_nodes = len(nodes)
        num_edges = self._graph.num_edges

        avg_degree = (2 * num_edges / num_nodes) if num_nodes > 0 else 0.0

        type_counts: Dict[str, int] = {}
        areas: List[float] = []
        curvatures: List[float] = []
        for node in nodes:
            type_counts[node.surface_type] = type_counts.get(node.surface_type, 0) + 1
            areas.append(node.area)
            curvatures.append(node.curvature)

        total = sum(type_counts.values()) or 1
        entropy = -sum(
            (c / total) * math.log2(c / total) for c in type_counts.values() if c > 0
        )

        return {
            "num_nodes": float(num_nodes),
            "num_edges": float(num_edges),
            "avg_degree": avg_degree,
            "surface_type_entropy": entropy,
            "mean_area": float(np.mean(areas)) if areas else 0.0,
            "mean_curvature": float(np.mean(curvatures)) if curvatures else 0.0,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _node_to_vector(node: BRepNode) -> np.ndarray:
        """Encode a single node as a 9-D feature vector."""
        stype = SURFACE_TYPE_MAP.get(node.surface_type.upper(), 6)
        return np.array([
            float(stype) / len(SURFACE_TYPE_MAP),
            float(node.area) / 10000.0,
            float(node.normal[0]),
            float(node.normal[1]),
            float(node.normal[2]),
            float(node.curvature),
            float(node.center[0]) / 1000.0,
            float(node.center[1]) / 1000.0,
            float(node.center[2]) / 1000.0,
        ], dtype=np.float32)
