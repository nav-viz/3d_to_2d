"""
models/graphsage.py
===================
Unified model builder that forwards to the stable CADHGT.
No longer passes deprecated 'num_heads' parameter.
"""

import torch.nn as nn
from loguru import logger

# Import the actual working model
from models.hgt import CADHGT


def build_model(cfg: dict, checkpoint_in_feats: int = None) -> nn.Module:
    """Build CADHGT using the trained checkpoint's feature dimension."""
    model_type = cfg["model"]["type"]

    # Force the dimension from checkpoint (highest priority)
    if checkpoint_in_feats is not None:
        in_feats = int(checkpoint_in_feats)
        logger.success(f"✅ Using checkpoint's original in_feats = {in_feats}")
    else:
        in_feats = cfg["model"].get("in_feats", 5)
        logger.warning(f"Using in_feats from config: {in_feats}")

    if model_type == "hgt":
        logger.info(f"Building CADHGT | in_feats={in_feats} | hidden=768 | layers=8")
        return CADHGT(
            in_feats=in_feats,
            hidden_dim=768,           # match your training
            out_feats=25,
            num_layers=8,             # match your training
            dropout=0.15,
        )
    else:
        logger.warning(f"Falling back to CADHGT for {model_type}")
        return CADHGT(
            in_feats=in_feats,
            hidden_dim=768,
            out_feats=25,
            num_layers=8,
            dropout=0.15,
        )