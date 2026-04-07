"""
models/graphsage.py
===================
Unified model builder for CAD pipeline.

- Ensures consistency between training and inference
- Supports checkpoint-based feature dimension restoration
- Adds defensive logging for debugging weight mismatch issues
"""

from __future__ import annotations

import torch.nn as nn
from loguru import logger

# Import model
from models.hgt import CADHGT

# Import NUM_CLASSES safely
try:
    from graph.mfcad_h5_dataset import NUM_CLASSES
except ImportError:
    NUM_CLASSES = 25
    logger.warning("NUM_CLASSES fallback to 25 (mfcad_h5_dataset not found)")


# ---------------------------------------------------------------------------
# Model Builder
# ---------------------------------------------------------------------------

# models/graphsage.py
"""
Unified model builder – now always respects cfg (or checkpoint-saved cfg).
No more hard-coded 768/8.
"""

import torch.nn as nn
from loguru import logger

from models.hgt import CADHGT


def build_model(cfg: dict, checkpoint_in_feats: int = None) -> nn.Module:
    """Build CADHGT using cfg (or checkpoint-saved values)."""
    model_type = cfg["model"]["type"]

    # in_feats always comes from checkpoint first
    in_feats = (
        int(checkpoint_in_feats)
        if checkpoint_in_feats is not None
        else cfg["model"].get("in_feats", 13)
    )

    # All other params from cfg (or checkpoint cfg)
    hidden_dim = cfg["model"].get("hidden_dim", 768)
    num_layers = cfg["model"].get("num_layers", 8)
    dropout = cfg["model"].get("dropout", 0.12)

    logger.success(
        f"✅ Building CADHGT | in_feats={in_feats} | "
        f"hidden={hidden_dim} | layers={num_layers} | dropout={dropout}"
    )

    if model_type == "hgt":
        return CADHGT(
            in_feats=in_feats,
            hidden_dim=hidden_dim,
            out_feats=25,
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        logger.warning(f"Falling back to CADHGT for model_type={model_type}")
        return CADHGT(
            in_feats=in_feats,
            hidden_dim=hidden_dim,
            out_feats=25,
            num_layers=num_layers,
            dropout=dropout,
        )