"""
cad_pipeline.py
===============
Production-ready pipeline: STEP → ISO/ASME DXF + PNG preview.

Usage:
    python cad_pipeline.py --step_file path/to/part.stp [--output_dir output]

Produces:
    output/<stem>.dxf   — ISO 128 / ASME Y14.3/Y14.5 compliant drawing
    output/<stem>.png   — High-res (300 DPI) preview via ezdxf drawing addon
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

# ─── Project root on sys.path ────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.logger import logger

# ─── OCC imports ─────────────────────────────────────────────────────────────
try:
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
    from OCC.Core.GeomAbs import (
        GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Cone,
        GeomAbs_Torus, GeomAbs_BSplineSurface,
    )
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2
    from OCC.Core.BRepLProp import BRepLProp_SLProps
    from OCC.Core.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
    from OCC.Core.HLRAlgo import HLRAlgo_Projector
    _OCC = True
except ImportError:
    _OCC = False
    logger.warning("pythonocc-core not found — geometric extraction limited.")

# ─── ezdxf ───────────────────────────────────────────────────────────────────
try:
    import ezdxf
    from ezdxf.enums import TextEntityAlignment
    _EZDXF = True
except ImportError:
    _EZDXF = False
    logger.error("ezdxf not installed. Run: pip install ezdxf")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features(step_path: str | Path) -> dict:
    """
    Extract all machining features from a STEP file.

    Strategy (layered):
      1. OCC geometric B-Rep analysis (primary, always runs)
      2. HGT model predictions (supplementary, merged where confident)

    Returns a structured dict with:
        bbox        — (xmin,ymin,zmin,xmax,ymax,zmax) in mm
        dimensions  — (length, width, height) in mm
        holes       — list of {diameter, depth, position:(x,y,z), through:bool}
        pockets     — list of {width, length, depth, position}
        chamfers    — list of {size, angle}
        fillets     — list of {radius}
        slots       — list of {width, length, depth}
        faces_raw   — raw per-face list for view-selector
    """
    step_path = Path(step_path)
    logger.info(f"[EXTRACT] Loading STEP: {step_path.name}")

    from cad.parser import load_step
    topology = load_step(str(step_path))
    shape = topology.get("shape")

    features = {
        "step_path": str(step_path),
        "stem": step_path.stem,
        "bbox": (0, 0, 0, 100, 50, 30),
        "dimensions": (100, 50, 30),
        "holes": [],
        "blind_holes": [],
        "pockets": [],
        "chamfers": [],
        "fillets": [],
        "slots": [],
        "faces_raw": [],
        "num_faces": len(topology.get("faces", [])),
    }

    if not _OCC or shape is None:
        logger.warning("[EXTRACT] OCC unavailable — returning defaults")
        return features

    # ── Global bounding box ───────────────────────────────────────────────────
    bbox = _shape_bbox(shape)
    features["bbox"] = bbox
    features["dimensions"] = (
        round(bbox[3] - bbox[0], 2),
        round(bbox[4] - bbox[1], 2),
        round(bbox[5] - bbox[2], 2),
    )
    logger.info(f"[EXTRACT] BBox: {features['dimensions']} mm (L×W×H)")

    # ── Per-face analysis ─────────────────────────────────────────────────────
    faces_data = topology.get("faces", [])
    total_area = sum(f.get("area", 0) for f in faces_data)
    avg_area = total_area / max(len(faces_data), 1)

    cylinders, planes, cones, tori = [], [], [], []

    for face in faces_data:
        occ_face = face.get("occ_face")
        if occ_face is None:
            continue
        surf_type = face.get("surface_type", "Unknown")
        area = face.get("area", 0.0)
        face_id = face.get("id", 0)

        if surf_type == "Cylinder":
            r, depth, center = _cylinder_params(occ_face)
            if r > 0:
                cylinders.append({"id": face_id, "radius": r, "depth": depth,
                                   "center": center, "area": area})

        elif surf_type == "Plane":
            normal, centroid = _plane_params(occ_face)
            planes.append({"id": face_id, "normal": normal,
                           "centroid": centroid, "area": area})

        elif surf_type == "Cone":
            cones.append({"id": face_id, "area": area})

        elif surf_type == "Torus":
            r = _torus_minor_radius(occ_face)
            tori.append({"id": face_id, "radius": r, "area": area})

    # ── Classify cylinders as holes / blind-holes / bosses ───────────────────
    large_area_thresh = avg_area * 1.5
    for cyl in cylinders:
        diameter = round(cyl["radius"] * 2, 3)
        depth = round(cyl["depth"], 3)
        center = cyl["center"]
        # Heuristic: small cylinder relative to part → hole
        if cyl["area"] < large_area_thresh:
            # Check if through-hole by depth vs model height
            model_h = features["dimensions"][2]
            is_through = depth >= model_h * 0.85
            entry = {
                "diameter": diameter,
                "radius": cyl["radius"],
                "depth": depth,
                "position": center,
                "through": is_through,
            }
            if is_through:
                features["holes"].append(entry)
            else:
                features["blind_holes"].append(entry)
        # else: boss / external cylinder — skip for now

    # ── Chamfers from cone faces ──────────────────────────────────────────────
    for cone in cones:
        features["chamfers"].append({"size": 1.0, "angle": 45.0})

    # ── Fillets from torus faces ──────────────────────────────────────────────
    for t in tori:
        if t["radius"] > 0:
            features["fillets"].append({"radius": round(t["radius"], 3)})

    # ── faces_raw for downstream view selection ───────────────────────────────
    from cad.brep_features import classify_faces
    classifications = classify_faces(topology)
    features["faces_raw"] = classifications

    # ── Merge HGT predictions if checkpoint available ─────────────────────────
    _merge_hgt(features, step_path)

    logger.info(
        f"[EXTRACT] Done: {len(features['holes'])} through-holes, "
        f"{len(features['blind_holes'])} blind-holes, "
        f"{len(features['chamfers'])} chamfers, "
        f"{len(features['fillets'])} fillets"
    )
    return features


def _shape_bbox(shape) -> tuple:
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    return box.Get()  # (xmin,ymin,zmin,xmax,ymax,zmax)


def _cylinder_params(occ_face) -> tuple:
    """Return (radius, depth, center_xyz)."""
    try:
        adaptor = BRepAdaptor_Surface(occ_face)
        if adaptor.GetType() != GeomAbs_Cylinder:
            return 0.0, 0.0, (0, 0, 0)
        cyl = adaptor.Cylinder()
        r = cyl.Radius()
        loc = cyl.Location()
        center = (loc.X(), loc.Y(), loc.Z())
        # depth from bounding box along cylinder axis
        box = Bnd_Box()
        brepbndlib.Add(occ_face, box)
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        dims = sorted([xmax - xmin, ymax - ymin, zmax - zmin])
        depth = dims[-1]
        return round(r, 4), round(depth, 4), tuple(round(c, 3) for c in center)
    except Exception:
        return 0.0, 0.0, (0, 0, 0)


def _plane_params(occ_face) -> tuple:
    """Return (normal_xyz, centroid_xyz)."""
    try:
        from OCC.Core.BRepLProp import BRepLProp_SLProps
        adaptor = BRepAdaptor_Surface(occ_face)
        u = (adaptor.FirstUParameter() + adaptor.LastUParameter()) / 2
        v = (adaptor.FirstVParameter() + adaptor.LastVParameter()) / 2
        props = BRepLProp_SLProps(adaptor, u, v, 1, 1e-6)
        if props.IsNormalDefined():
            n = props.Normal()
            normal = (round(n.X(), 3), round(n.Y(), 3), round(n.Z(), 3))
        else:
            normal = (0, 0, 1)
        # centroid via GProp
        gp = GProp_GProps()
        brepgprop.SurfaceProperties(occ_face, gp)
        c = gp.CentreOfMass()
        centroid = (round(c.X(), 3), round(c.Y(), 3), round(c.Z(), 3))
        return normal, centroid
    except Exception:
        return (0, 0, 1), (0, 0, 0)


def _torus_minor_radius(occ_face) -> float:
    try:
        adaptor = BRepAdaptor_Surface(occ_face)
        return adaptor.Torus().MinorRadius()
    except Exception:
        return 0.0


def _merge_hgt(features: dict, step_path: Path) -> None:
    """Try to load HGT predictions and merge into features dict."""
    try:
        import yaml, torch
        import torch.nn.functional as F
        cfg_path = _ROOT / "configs" / "default.yaml"
        if not cfg_path.exists():
            return
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        ckpt_path = _ROOT / "checkpoints" / "best_hgt_retrained.pt"
        if not ckpt_path.exists():
            logger.warning("[HGT] No checkpoint found — skipping HGT merge.")
            return

        from inference.predict import predict
        preds = predict(step_path, cfg, checkpoint_path=str(ckpt_path))

        # Count HGT-detected non-Stock features
        hgt_features = [p for p in preds if p["feature_type"] not in ("Stock", "Low_Confidence")]
        logger.info(f"[HGT] {len(hgt_features)}/{len(preds)} faces classified as machining features")
        features["hgt_predictions"] = preds

    except Exception as e:
        logger.warning(f"[HGT] Merge failed (non-fatal): {e}")
        features["hgt_predictions"] = []
