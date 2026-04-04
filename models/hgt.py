"""
models/hgt.py
=============
Stable CADHGT – uses HeteroGraphConv + SAGEConv (recommended for MFCAD++)
Works reliably with single node type ("face") and variable-sized graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

try:
    import dgl
    from dgl.nn import HeteroGraphConv, SAGEConv
    _DGL_AVAILABLE = True
except ImportError:
    _DGL_AVAILABLE = False
    logger.warning("DGL not found")


class CADHGT(nn.Module):
    """
    Stable version of CADHGT using HeteroGraphConv + GraphSAGE convolution.
    This is the recommended architecture for your MFCAD++ dataset.
    """

    def __init__(
        self,
        in_feats: int = 5,
        hidden_dim: int = 256,
        out_feats: int = 25,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not _DGL_AVAILABLE:
            raise ImportError("DGL is required")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(in_feats, hidden_dim)

        # HeteroGraphConv with SAGEConv (very stable)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                HeteroGraphConv(
                    {
                        ("face", "convex", "face"): SAGEConv(hidden_dim, hidden_dim, "mean"),
                        ("face", "concave", "face"): SAGEConv(hidden_dim, hidden_dim, "mean"),
                        ("face", "smooth", "face"): SAGEConv(hidden_dim, hidden_dim, "mean"),
                        ("face", "adj", "face"): SAGEConv(hidden_dim, hidden_dim, "mean"),
                    },
                    aggregate="sum",
                )
            )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_feats),
        )

        logger.info(
            f"CADHGT (stable) initialized | layers={num_layers} | "
            f"hidden={hidden_dim} | in_feats={in_feats}"
        )

    def forward(self, g: "dgl.DGLHeteroGraph") -> torch.Tensor:
        # Get features
        h = g.nodes["face"].data["feat"]                    # (N, in_feats)
        h = F.relu(self.input_proj(h))
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Message passing
        for layer in self.layers:
            h_dict = layer(g, {"face": h})
            h = h_dict["face"]                              # extract face features
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Final prediction
        logits = self.classifier(h)
        return logits