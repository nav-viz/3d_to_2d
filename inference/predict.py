"""
inference/predict.py
====================
Run GNN inference on a single STEP file (end-to-end).

Flow
----
1. Load + parse STEP         (cad.parser)
2. Ingest to Neo4j           (kg.ingest)
3. Build DGL graph           (graph.build_dgl)
4. Load trained checkpoint   (checkpoints/best_*.pt)
5. Forward pass → softmax probabilities
6. Return list of per-face predictions

Usage (CLI)
-----------
    python -m inference.predict \
        --step_file path/to/part.step \
        --model_type hgt \
        --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml
from cad.parser import load_step
from graph.build_dgl import build_from_neo4j
from graph.extract import extract_face_features
from kg.ingest import ingest_step
from kg.neo4j_client import Neo4jClient
from models.graphsage import build_model

try:
    import dgl
    _DGL_AVAILABLE = True
except ImportError:
    _DGL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(
    step_path: str | Path,
    cfg: dict,
    model_type: str | None = None,
    checkpoint_path: str | Path | None = None,
) -> list[dict]:
    """
    Run end-to-end inference on a STEP file.

    Parameters
    ----------
    step_path       : path to the ``.step`` file
    cfg             : full pipeline config dict
    model_type      : "hgt" | "graphsage" (default: config value)
    checkpoint_path : explicit ckpt path (default: checkpoints/best_<type>.pt)

    Returns
    -------
    list of dicts:
        {face_id, feature_type, confidence, class_id, logits, params}
    """
    step_path = Path(step_path)
    model_type = model_type or cfg["model"]["type"]

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt_path = checkpoint_path or (
        Path(cfg["paths"]["checkpoints"]) / f"best_{model_type}.pt"
    )
    checkpoint = _load_checkpoint(ckpt_path, cfg, model_type)
    model = checkpoint["model"]
    device = next(model.parameters()).device

    # ── Parse STEP ────────────────────────────────────────────────────────────
    topology = load_step(step_path)

    # ── Ingest to Neo4j ───────────────────────────────────────────────────────
    client = Neo4jClient.from_config(cfg)
    model_id = ingest_step(step_path, client, cfg)

    # ── Build graph ───────────────────────────────────────────────────────────
    g = build_from_neo4j(model_id, client, topology, cfg)
    g = g.to(device)

    # ── Inference ─────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits = model(g)                      # (N, num_classes)
        probs  = F.softmax(logits, dim=-1)     # (N, num_classes)
        class_ids  = probs.argmax(dim=-1).cpu().tolist()
        confidences = probs.max(dim=-1).values.cpu().tolist()

    # ── Decode predictions ────────────────────────────────────────────────────
    from graph.encode import encode_labels  # label map constant
    CLASS_NAMES = _class_id_to_name()

    conf_thresh = cfg["inference"]["confidence_threshold"]
    predictions = []
    for i, (cid, conf) in enumerate(zip(class_ids, confidences)):
        feature_type = (
            CLASS_NAMES.get(cid, "Unknown")
            if conf >= conf_thresh
            else "Low_Confidence"
        )
        predictions.append({
            "face_id":      i,
            "feature_type": feature_type,
            "class_id":     cid,
            "confidence":   round(conf, 4),
            "logits":       logits[i].cpu().tolist(),
            "params":       {},   # extend with geometric params
        })
        logger.debug(
            f"Face {i}: {feature_type}  conf={conf:.3f}"
        )

    logger.info(
        f"Inference done: {len(predictions)} faces  "
        f"(threshold={conf_thresh})"
    )
    return predictions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(
    ckpt_path: str | Path,
    cfg: dict,
    model_type: str,
) -> dict:
    """Load trained checkpoint with correct feature dimension."""
    ckpt_path = Path(ckpt_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    # Extract saved feature dimension
    saved_in_feats = ckpt.get("in_feats")
    if saved_in_feats is None:
        saved_in_feats = ckpt.get("cfg", {}).get("model", {}).get("in_feats", 5)

    logger.success(f"Checkpoint trained with in_feats = {saved_in_feats}")

    # Build model with correct dimension
    model = build_model(cfg, checkpoint_in_feats=saved_in_feats)

    # Load weights
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    logger.success(
        f"Model loaded successfully from {ckpt_path} "
        f"(epoch={ckpt.get('epoch', '?')}, val_f1={ckpt.get('val_f1', 0):.4f})"
    )
    return {"model": model, "ckpt": ckpt}

def _class_id_to_name() -> dict[int, str]:
    """Return the MFCAD++ class id → name mapping."""
    return {
        0: "Chamfer",
        1: "Through_Hole",
        2: "Triangular_Passage",
        3: "Rectangular_Passage",
        4: "6Sides_Passage",
        5: "Triangular_Through_Slot",
        6: "Rectangular_Through_Slot",
        7: "Circular_Through_Slot",
        8: "Rectangular_Through_Step",
        9: "2Sides_Through_Step",
        10: "Slanted_Through_Step",
        11: "Oring",
        12: "Blind_Hole",
        13: "Triangular_Pocket",
        14: "Rectangular_Pocket",
        15: "6Sides_Pocket",
        16: "Circular_End_Pocket",
        17: "Rectangular_Blind_Step",
        18: "2Sides_Blind_Step",
        19: "Slanted_Blind_Step",
        20: "Rectangular_Blind_Slot",
        21: "Vertical_Circular_End_Blind_Slot",
        22: "Horizontal_Circular_End_Blind_Slot",
        23: "Triangular_Blind_Step",
        24: "Stock",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GNN inference on a STEP file")
    p.add_argument("--step_file",   required=True)
    p.add_argument("--config",      default="configs/default.yaml")
    p.add_argument("--model_type",  default=None,
                   choices=["hgt", "graphsage"])
    p.add_argument("--checkpoint",  default=None)
    p.add_argument("--output_json", default=None,
                   help="Save predictions to JSON file")
    return p.parse_args()


def main() -> None:
    import json
    args = _parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    logger.remove()
    logger.add(sys.stderr, level=cfg["logging"]["level"],
               colorize=cfg["logging"]["colorize"])

    preds = predict(
        step_path=args.step_file,
        cfg=cfg,
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
    )

    if args.output_json:
        with open(args.output_json, "w") as fh:
            json.dump(preds, fh, indent=2)
        logger.info(f"Predictions saved → {args.output_json}")
    else:
        for p in preds[:10]:
            print(f"Face {p['face_id']:>4}  {p['feature_type']:<35}  "
                  f"conf={p['confidence']:.3f}")


if __name__ == "__main__":
    main()
