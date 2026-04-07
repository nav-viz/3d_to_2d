"""
models/graphsage.py
===================
Unified model builder for the CAD Automation Pipeline.

- Ensures consistency between training and inference
- Prioritizes checkpoint-saved feature dimension
- Adds clear defensive logging for debugging dimension/weight mismatches
- Removes duplicate code and hard-coded values
"""

from __future__ import annotations

import torch.nn as nn
from utils.logger import logger

# Import the actual model implementation
from models.hgt import CADHGT

# Safe import for NUM_CLASSES (fallback if dataset module unavailable)
try:
    from graph.mfcad_h5_dataset import NUM_CLASSES
except ImportError:
    NUM_CLASSES = 25
    logger.warning("NUM_CLASSES fallback to 25 (mfcad_h5_dataset not found)")


def build_model(cfg: dict, checkpoint_in_feats: int | None = None) -> nn.Module:
    """
    Build the CADHGT model using configuration, with priority to checkpoint values.

    Parameters
    ----------
    cfg : dict
        Pipeline configuration dictionary (from default.yaml or saved checkpoint cfg)
    checkpoint_in_feats : int | None
        Feature dimension saved in the checkpoint. Takes highest priority.

    Returns
    -------
    nn.Module
        Initialized CADHGT model
    """
    # --- Determine input feature dimension (checkpoint > config) ---
    if checkpoint_in_feats is not None:
        in_feats = int(checkpoint_in_feats)
        logger.success(f"✅ Using checkpoint's original in_feats = {in_feats}")
    else:
        in_feats = cfg["model"].get("in_feats", 13)
        logger.info(f"Using in_feats from config: {in_feats}")

    # --- Extract other architecture parameters from cfg ---
    hidden_dim = cfg["model"].get("hidden_dim", 768)
    num_layers = cfg["model"].get("num_layers", 8)
    dropout = cfg["model"].get("dropout", 0.12)

    # Defensive logging for debugging weight mismatches
    logger.success(
        f"✅ Building CADHGT | in_feats={in_feats} | "
        f"hidden={hidden_dim} | layers={num_layers} | dropout={dropout}"
    )

    # Get model type (with fallback)
    model_type = cfg["model"].get("type", "hgt")

    if model_type == "hgt":
        return CADHGT(
            in_feats=in_feats,
            hidden_dim=hidden_dim,
            out_feats=NUM_CLASSES,      # Use safely imported constant
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        logger.warning(f"Model type '{model_type}' not fully supported. Falling back to CADHGT.")
        return CADHGT(
            in_feats=in_feats,
            hidden_dim=hidden_dim,
            out_feats=NUM_CLASSES,
            num_layers=num_layers,
            dropout=dropout,
        )