"""
models/hgt.py
=============
Stable baseline + mild improvements: higher capacity, better residuals, mild attention option.
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
    Stable CADHGT (SAGE-based) with mild upgrades for better performance.
    """

    def __init__(
        self,
        in_feats: int = 13,      # updated
        hidden_dim: int = 640,
        out_feats: int = 25,
        num_layers: int = 8,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        if not _DGL_AVAILABLE:
            raise ImportError("DGL is required")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.input_proj = nn.Linear(in_feats, hidden_dim)

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

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, out_feats),
        )

        logger.info(f"CADHGT stable | layers={num_layers} | hidden={hidden_dim} | in_feats={in_feats}")
    
    def forward(self, g: "dgl.DGLHeteroGraph") -> torch.Tensor:
        h = g.nodes["face"].data["feat"]
        h = F.relu(self.input_proj(h))
        h = F.dropout(h, p=self.dropout, training=self.training)

        for i, layer in enumerate(self.layers):
            h_dict = layer(g, {"face": h})
            h_new = h_dict["face"]
            # Residual with scaling to prevent explosion
            h = h + 0.8 * h_new if h.shape == h_new.shape else h_new
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        logits = self.classifier(h)
        return logits