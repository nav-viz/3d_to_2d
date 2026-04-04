"""
models/hgt.py - Fixed for single node-type graphs (only "face")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

try:
    import dgl
    from dgl.nn import HGTConv
    _DGL_AVAILABLE = True
except ImportError:
    _DGL_AVAILABLE = False
    logger.warning("DGL not found")


class CADHGT(nn.Module):
    def __init__(
        self,
        in_feats: int = 5,
        hidden_dim: int = 256,
        out_feats: int = 25,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        if not _DGL_AVAILABLE:
            raise ImportError("DGL is required for CADHGT.")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(in_feats, hidden_dim)

        # HGT layers - simplified for single node type
        self.hgt_layers = nn.ModuleList([
            HGTConv(
                in_size=hidden_dim,
                head_size=hidden_dim // num_heads,
                num_heads=num_heads,
                num_ntypes=1,           # Only "face" node type
                num_etypes=4,           # convex, concave, smooth, adj
                dropout=dropout,
                use_norm=True,
            )
            for _ in range(num_layers)
        ])

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_feats),
        )

        logger.info(f"CADHGT initialized | layers={num_layers} | hidden={hidden_dim} | in_feats={in_feats}")

    def forward(self, g: "dgl.DGLHeteroGraph"):
        # Get features
        h = g.nodes["face"].data["feat"]                     # (N, in_feats)
        h = F.relu(self.input_proj(h))
        h = F.dropout(h, p=self.dropout, training=self.training)

        # HGT layers
        for layer in self.hgt_layers:
            # For single node type, we use the same h for src and dst
            h = layer(g, h, h, etype=None)   # etype=None works for single-type case in many versions
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        logits = self.classifier(h)
        return logits