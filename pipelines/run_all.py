"""
pipelines/run_all.py
====================
End-to-end orchestrator for the CAD automation pipeline.

CLI usage
---------
    # Stage 1: Ingest STEP → Neo4j
    python pipelines/run_all.py ingest --step_file path/to/part.step

    # Stage 2: Train GNN on MFCAD++ H5 files
    python pipelines/run_all.py train --model_type hgt --epochs 100

    # Stage 3: Infer features for a STEP file
    python pipelines/run_all.py infer --step_file path/to/part.step

    # Stage 4: Export DXF drawing
    python pipelines/run_all.py export --step_file path/to/part.step \
                                        --output out/part.dxf

    # Run all stages for a STEP file (ingest → infer → export)
    python pipelines/run_all.py all --step_file path/to/part.step \
                                     --model_type hgt

Global options
--------------
    --config        Path to YAML config (default: configs/default.yaml)
    --model_type    "hgt" | "graphsage"
    --log_level     DEBUG | INFO | WARNING
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from loguru import logger
import yaml

# Ensure project root on path when run directly
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

def stage_ingest(step_path: Path, client, cfg: dict) -> str:
    """Parse STEP and write topology + rule-based features to Neo4j."""
    from kg.ingest import ingest_step
    model_id = ingest_step(step_path, client, cfg)
    logger.success(f"[INGEST] Done. model_id={model_id}")
    return model_id


def stage_train(cfg: dict, model_type: str, **kwargs) -> None:
    """Train GNN on MFCAD++ H5 datasets."""
    from training.train import main as train_main
    import sys as _sys
    # Build argv for the training sub-module
    extra = []
    if model_type:
        extra += ["--model_type", model_type]
    if kwargs.get("epochs"):
        extra += ["--epochs", str(kwargs["epochs"])]
    if kwargs.get("max_samples"):
        extra += ["--max_samples", str(kwargs["max_samples"])]
    if kwargs.get("batch_size"):
        extra += ["--batch_size", str(kwargs["batch_size"])]

    old_argv = _sys.argv
    _sys.argv = ["train.py", "--config", str(kwargs.get("config", "configs/default.yaml"))] + extra
    try:
        train_main()
    finally:
        _sys.argv = old_argv


def stage_infer(
    step_path: Path,
    model_type: str,
    cfg: dict,
    checkpoint: str | None = None,
) -> list[dict]:
    """Run GNN inference → return predictions."""
    from inference.predict import predict
    preds = predict(
        step_path=step_path,
        cfg=cfg,
        model_type=model_type,
        checkpoint_path=checkpoint,
    )
    logger.success(f"[INFER] Done. {len(preds)} predictions.")
    return preds


def stage_exportdxf(
    step_path: Path,
    predictions: list[dict],
    model_id: str,
    client,
    output_path: Path,
    cfg: dict,
) -> None:
    """Project views, place dimensions, export DXF."""
    from cad.parser import load_step
    from reasoning.view_selector import select_views
    from drawing.projection import project_view
    from drawing.dimension_placer import extract_dimensions, place_and_resolve
    from drawing.dxf_writer import export_dxf

    topology = load_step(step_path)
    shape    = topology.get("shape")

    # ── View selection ────────────────────────────────────────────────────────
    views = select_views(predictions, cfg, num_faces=topology["num_faces"])
    logger.info(f"[EXPORT] Selected {len(views)} views: "
                f"{[v.name for v in views]}")

    # ── HLR Projection ────────────────────────────────────────────────────────
    projections = []
    for v in views:
        proj = project_view(
            shape=shape,
            view_name=v.name,
            direction=v.direction,
            scale=1.0,
        )
        projections.append(proj)

    # ── Dimension extraction + placement ─────────────────────────────────────
    raw_dims = extract_dimensions(predictions, topology)
    if projections:
        bbox = projections[0].bounding_box  # use front view bbox for placement
    else:
        bbox = (0, 0, 100, 80)
    placed_dims = place_and_resolve(raw_dims, bbox, cfg)

    # ── Export ────────────────────────────────────────────────────────────────
    export_dxf(
        projections=projections,
        dimensions=placed_dims,
        predictions=predictions,
        output_path=output_path,
        cfg=cfg,
        part_name=step_path.stem.upper(),
    )
    logger.success(f"[EXPORT] DXF saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cad-run",
        description="CAD Automation Pipeline – hybrid KG + GNN",
    )
    p.add_argument("--config",     default="configs/default.yaml",
                   help="Path to YAML config file")
    p.add_argument("--log_level",  default=None,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    sub = p.add_subparsers(dest="stage", required=True)

    # ── ingest ────────────────────────────────────────────────────────────────
    p_ingest = sub.add_parser("ingest", help="Parse STEP and ingest to Neo4j")
    p_ingest.add_argument("--step_file", required=True)

    # ── train ─────────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Train GNN on MFCAD++ H5 files")
    p_train.add_argument("--model_type", default=None,
                         choices=["hgt", "graphsage"])
    p_train.add_argument("--epochs",      type=int, default=None)
    p_train.add_argument("--max_samples", type=int, default=None,
                         help="Limit dataset size (debug)")
    p_train.add_argument("--batch_size", type=int, default=None)
    # ── infer ─────────────────────────────────────────────────────────────────
    p_infer = sub.add_parser("infer", help="Run GNN inference on a STEP file")
    p_infer.add_argument("--step_file",   required=True)
    p_infer.add_argument("--model_type",  default=None,
                         choices=["hgt", "graphsage"])
    p_infer.add_argument("--checkpoint",  default=None)
    p_infer.add_argument("--output_json", default=None)

    # ── export ────────────────────────────────────────────────────────────────
    p_export = sub.add_parser("export", help="Full DXF export from STEP")
    p_export.add_argument("--step_file",  required=True)
    p_export.add_argument("--model_type", default=None,
                          choices=["hgt", "graphsage"])
    p_export.add_argument("--checkpoint", default=None)
    p_export.add_argument("--output",     default=None,
                          help="Output .dxf path (default: output/<name>.dxf)")

    # ── all ───────────────────────────────────────────────────────────────────
    p_all = sub.add_parser("all", help="Run ingest → infer → export")
    p_all.add_argument("--step_file",  required=True)
    p_all.add_argument("--model_type", default=None,
                       choices=["hgt", "graphsage"])
    p_all.add_argument("--checkpoint", default=None)
    p_all.add_argument("--output",     default=None)

    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    config_path = getattr(args, "config", "configs/default.yaml")
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level = args.log_level or cfg["logging"]["level"]
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        colorize=cfg["logging"].get("colorize", True),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> – {message}"
        ),
    )
    log_dir = Path(cfg["paths"]["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "pipeline.log",
        level="DEBUG",
        rotation=cfg["logging"]["rotation"],
        retention=cfg["logging"]["retention"],
    )

    # ── Model type resolution ─────────────────────────────────────────────────
    model_type = getattr(args, "model_type", None) or cfg["model"]["type"]
    cfg["model"]["type"] = model_type

    # ── Neo4j client ─────────────────────────────────────────────────────────
    from kg.neo4j_client import Neo4jClient
    client = Neo4jClient.from_config(cfg)

    t_start = time.perf_counter()
    stage   = args.stage

    # ─────────────────────────────────────────────────────────────────────────
    if stage == "ingest":
        step_path = Path(args.step_file)
        stage_ingest(step_path, client, cfg)

    # ─────────────────────────────────────────────────────────────────────────
    elif stage == "train":
        stage_train(
            cfg=cfg,
            model_type=model_type,
            config=config_path,
            epochs=getattr(args, "epochs", None),
            max_samples=getattr(args, "max_samples", None),
            batch_size=getattr(args, "batch_size", None),   # ← Add this
        )

    # ─────────────────────────────────────────────────────────────────────────
    elif stage == "infer":
        step_path = Path(args.step_file)
        preds = stage_infer(
            step_path=step_path,
            model_type=model_type,
            cfg=cfg,
            checkpoint=getattr(args, "checkpoint", None),
        )
        # Write-back to KG
        from kg.ingest import _model_id_from_path
        from inference.writeback import write_predictions_to_kg
        model_id = _model_id_from_path(step_path)
        write_predictions_to_kg(model_id, preds, client, model_type)

        if getattr(args, "output_json", None):
            with open(args.output_json, "w") as fh:
                json.dump(preds, fh, indent=2)
            logger.info(f"Predictions → {args.output_json}")

    # ─────────────────────────────────────────────────────────────────────────
    elif stage == "export":
        step_path = Path(args.step_file)

        # Ingest first (idempotent)
        model_id = stage_ingest(step_path, client, cfg)

        # Infer
        preds = stage_infer(
            step_path=step_path,
            model_type=model_type,
            cfg=cfg,
            checkpoint=getattr(args, "checkpoint", None),
        )

        # Write-back
        from inference.writeback import write_predictions_to_kg
        write_predictions_to_kg(model_id, preds, client, model_type)

        # Export
        out_dir = Path(cfg["paths"].get("output_dxf", "output"))
        output = Path(args.output) if args.output else (
            out_dir / f"{step_path.stem}.dxf"
        )
        stage_exportdxf(step_path, preds, model_id, client, output, cfg)

    # ─────────────────────────────────────────────────────────────────────────
    elif stage == "all":
        step_path = Path(args.step_file)
        model_id  = stage_ingest(step_path, client, cfg)
        preds     = stage_infer(
            step_path=step_path,
            model_type=model_type,
            cfg=cfg,
            checkpoint=getattr(args, "checkpoint", None),
        )
        from inference.writeback import write_predictions_to_kg
        write_predictions_to_kg(model_id, preds, client, model_type)
        out_dir = Path(cfg["paths"].get("output_dxf", "output"))
        output  = Path(args.output) if args.output else (
            out_dir / f"{step_path.stem}.dxf"
        )
        stage_exportdxf(step_path, preds, model_id, client, output, cfg)

    elapsed = time.perf_counter() - t_start
    logger.info(f"Pipeline stage '{stage}' completed in {elapsed:.2f}s")
    client.close()


if __name__ == "__main__":
    main()
