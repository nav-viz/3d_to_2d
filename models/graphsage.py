"""
models/graphsage.py
===================
CADGraphSAGE – GraphSAGE-based GNN using DGL's ``HeteroGraphConv``
to handle the MFCAD++ heterogeneous graph (convex/concave/smooth/adj edges).

Architecture
------------
• Input projection:  Linear(in_feats → hidden_dim)
• HeteroGraphConv layers stacked (num_layers)
  – inner conv: dgl.nn.SAGEConv (mean aggregator)
• Mean-pooling across edge types at each layer
• Output: Linear(hidden_dim → num_classes)

Switching between HGT and GraphSAGE is controlled by
``configs/default.yaml`` → ``model.type`` or via ``--model_type`` CLI.
"""

from __future__ import annotations

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
    logger.warning("DGL not found – CADGraphSAGE will not function.")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CADGraphSAGE(nn.Module):
    """
    Heterogeneous GraphSAGE for node-level machining feature classification.

    Parameters
    ----------
    in_feats   : int  – input feature dimension
    hidden_dim : int  – hidden state size (same across all layers)
    out_feats  : int  – number of output classes
    num_layers : int  – number of message-passing iterations
    dropout    : float
    aggr       : str  – SAGEConv aggregator: "mean" | "gcn" | "pool" | "lstm"
    """

    def __init__(
        self,
        in_feats: int = 10,
        hidden_dim: int = 256,
        out_feats: int = 25,
        num_layers: int = 3,
        dropout: float = 0.2,
        aggr: str = "mean",
    ) -> None:
        super().__init__()
        if not _DGL_AVAILABLE:
            raise ImportError("DGL is required for CADGraphSAGE.")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout    = dropout

        # Input projection
        self.input_proj = nn.Linear(in_feats, hidden_dim)

        # Edge-type relation map used by HeteroGraphConv
        _edge_types = ["convex", "concave", "smooth", "adj"]
        self._rel_map = {
            ("face", rel, "face"): SAGEConv(
                in_feats=hidden_dim,
                out_feats=hidden_dim,
                aggregator_type=aggr,
            )
            for rel in _edge_types
        }

        # HeteroGraphConv layers
        self.hconv_layers: nn.ModuleList = nn.ModuleList([
            HeteroGraphConv(
                self._rel_map if i == 0 else {
                    ("face", rel, "face"): SAGEConv(
                        in_feats=hidden_dim,
                        out_feats=hidden_dim,
                        aggregator_type=aggr,
                    )
                    for rel in _edge_types
                },
                aggregate="mean",   # combine multiple edge-type outputs
            )
            for i in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_feats),
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"CADGraphSAGE | layers={num_layers} | "
            f"hidden={hidden_dim} | params={n_params:,}"
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        g: "dgl.DGLHeteroGraph",
        feat_dict: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        g         : DGL heterograph (face nodes with 'feat' data)
        feat_dict : optional override for node features

        Returns
        -------
        logits : FloatTensor, shape (N_face, num_classes)
        """
        if feat_dict is None:
            feat_dict = {"face": g.nodes["face"].data["feat"]}

        # Input projection
        h: dict[str, torch.Tensor] = {
            "face": F.relu(self.input_proj(feat_dict["face"]))
        }

        for i, conv in enumerate(self.hconv_layers):
            # Filter to existing edge types in the graph
            h_out = conv(g, h)
            # Apply layer norm + residual
            h = {
                ntype: self.norms[i](h_out[ntype] + h[ntype])
                for ntype in h_out
            }
            h = {
                ntype: F.dropout(F.relu(feat), p=self.dropout,
                                 training=self.training)
                for ntype, feat in h.items()
            }

        logits = self.classifier(h["face"])
        return logits


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> nn.Module:
    """
    Construct the GNN model from the ``model`` section of default.yaml.
    Supports ``model.type = "hgt"`` or ``"graphsage"``.
    """
    mc = cfg["model"]
    model_type = mc.get("type", "hgt").lower()

    if model_type == "graphsage":
        return CADGraphSAGE(
            in_feats=mc["in_feats"],
            hidden_dim=mc["hidden_dim"],
            out_feats=mc["out_feats"],
            num_layers=mc["num_layers"],
            dropout=mc["dropout"],
        )
    elif model_type == "hgt":
        from models.hgt import CADHGT
        return CADHGT(
            in_feats=mc["in_feats"],
            hidden_dim=mc["hidden_dim"],
            out_feats=mc["out_feats"],
            num_layers=mc["num_layers"],
            num_heads=mc.get("num_heads", 4),
            dropout=mc["dropout"],
            use_norm=mc.get("use_norm", True),
        )
    else:
        raise ValueError(
            f"Unknown model_type='{model_type}'. "
            "Choose 'graphsage' or 'hgt'."
        )
