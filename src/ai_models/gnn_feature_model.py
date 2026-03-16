"""GNN feature model with rule-based fallback."""
from __future__ import annotations

import importlib.util
from typing import Dict, List

import numpy as np

from src.brep_graph.graph_builder import BRepGraph, BRepNode
from src.brep_graph.graph_features import GraphFeatureExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)

FEATURE_CLASSES = [
    "Hole", "Pocket", "Slot", "Boss", "Fillet", "Chamfer", "Rib", "Unknown"
]

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    PYG_AVAILABLE = importlib.util.find_spec("torch_geometric") is not None
    if PYG_AVAILABLE:
        logger.info("PyTorch Geometric available – GNN model active.")
    else:
        logger.info("torch_geometric not available – using rule-based fallback.")
else:
    PYG_AVAILABLE = False
    logger.info("PyTorch not available – using rule-based feature recognition.")


class GNNFeatureModel:
    """
    Graph Neural Network model for manufacturing feature recognition.

    Assigns one of the following labels to each face in the BRep graph:
    ``Hole``, ``Pocket``, ``Slot``, ``Boss``, ``Fillet``, ``Chamfer``,
    ``Rib``, ``Unknown``.

    When PyTorch Geometric is available a GIN-based architecture is used;
    otherwise a deterministic rule-based classifier runs on raw geometry.
    """

    def __init__(
        self,
        model_path: str = "",
        num_classes: int = len(FEATURE_CLASSES),
        hidden_dim: int = 64,
    ) -> None:
        self._log = get_logger(self.__class__.__name__)
        self._num_classes = num_classes
        self._model: object = None

        if PYG_AVAILABLE:
            self._model = self._build_gnn(hidden_dim, num_classes)
            if model_path:
                self._load_weights(model_path)
        else:
            self._log.info("Rule-based feature classifier active.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, brep_graph: BRepGraph) -> Dict:
        """
        Predict manufacturing feature labels for every face in the graph.

        Args:
            brep_graph: Input BRep graph.

        Returns:
            Dictionary with:
            - ``face_labels``: dict mapping face id → label string
            - ``face_probabilities``: dict mapping face id → probability vector
            - ``summary``: feature-class count dictionary
        """
        if PYG_AVAILABLE and self._model is not None:
            return self._predict_gnn(brep_graph)
        return self._predict_rule_based(brep_graph)

    # ------------------------------------------------------------------
    # GNN path
    # ------------------------------------------------------------------

    def _build_gnn(self, hidden_dim: int, num_classes: int):  # type: ignore[return]
        """Build a 2-layer GIN classifier."""
        import torch.nn as nn
        from torch_geometric.nn import GINConv

        class _GIN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                in_dim = GraphFeatureExtractor.NODE_FEATURE_DIM
                mlp1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
                mlp2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
                self.conv1 = GINConv(mlp1)
                self.conv2 = GINConv(mlp2)
                self.head = nn.Linear(hidden_dim, num_classes)

            def forward(self, x, edge_index, batch=None):
                x = self.conv1(x, edge_index)
                x = self.conv2(x, edge_index)
                return self.head(x)

        return _GIN()

    def _predict_gnn(self, brep_graph: BRepGraph) -> Dict:
        """Run inference with the GNN model."""
        import torch
        import torch.nn.functional as F
        extractor = GraphFeatureExtractor(brep_graph)
        data = extractor.to_pyg_data()
        if data is None:
            return self._predict_rule_based(brep_graph)

        self._model.eval()  # type: ignore[union-attr]
        with torch.no_grad():
            logits = self._model(data.x, data.edge_index)  # type: ignore[operator]
            probs = F.softmax(logits, dim=-1).numpy()

        return self._build_result(brep_graph, probs)

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------

    def _predict_rule_based(self, brep_graph: BRepGraph) -> Dict:
        """Classify faces using heuristic geometry rules."""
        nodes = brep_graph.node_list()
        num_nodes = len(nodes)
        probs = np.zeros((num_nodes, self._num_classes), dtype=np.float32)

        for i, node in enumerate(nodes):
            label_idx = self._classify_node(node)
            probs[i, label_idx] = 1.0

        return self._build_result(brep_graph, probs)

    @staticmethod
    def _classify_node(node: BRepNode) -> int:
        """Map geometry rules to a feature class index."""
        stype = node.surface_type.upper()
        if stype == "CYLINDER":
            if node.curvature > 0.05:
                return FEATURE_CLASSES.index("Hole")
            return FEATURE_CLASSES.index("Boss")
        if stype == "PLANE":
            if node.area < 200:
                if node.curvature > 0.001:
                    return FEATURE_CLASSES.index("Fillet")
                return FEATURE_CLASSES.index("Chamfer")
            nx, ny, nz = node.normal
            if abs(nz) > 0.9:
                return FEATURE_CLASSES.index("Pocket")
            return FEATURE_CLASSES.index("Slot")
        if stype in ("CONE",):
            return FEATURE_CLASSES.index("Chamfer")
        if stype in ("TORUS",):
            return FEATURE_CLASSES.index("Fillet")
        return FEATURE_CLASSES.index("Unknown")

    # ------------------------------------------------------------------
    # Result builder
    # ------------------------------------------------------------------

    def _build_result(self, brep_graph: BRepGraph, probs: np.ndarray) -> Dict:
        """Package prediction arrays into the standard result dict."""
        nodes = brep_graph.node_list()
        face_labels: Dict[int, str] = {}
        face_probs: Dict[int, List[float]] = {}
        summary: Dict[str, int] = {cls: 0 for cls in FEATURE_CLASSES}

        for i, node in enumerate(nodes):
            label_idx = int(np.argmax(probs[i]))
            label = FEATURE_CLASSES[label_idx]
            face_labels[node.id] = label
            face_probs[node.id] = probs[i].tolist()
            summary[label] = summary.get(label, 0) + 1
            node.feature_label = label

        return {
            "face_labels": face_labels,
            "face_probabilities": face_probs,
            "summary": summary,
        }

    def _load_weights(self, path: str) -> None:
        """Load pre-trained weights if the file exists."""
        import os
        import torch
        if not os.path.exists(path):
            self._log.warning("Model weights not found at '%s' – using random init.", path)
            return
        try:
            state = torch.load(path, map_location="cpu")
            self._model.load_state_dict(state)  # type: ignore[union-attr]
            self._log.info("Loaded GNN weights from '%s'.", path)
        except Exception as exc:
            self._log.warning("Could not load weights from '%s': %s", path, exc)
