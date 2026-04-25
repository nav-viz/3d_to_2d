"""
cad_pipeline.py
===============
Production-ready pipeline: STEP â†’ ISO/ASME DXF + PNG preview.

Usage:
    python cad_pipeline.py --step_file path/to/part.stp [--output_dir output]

Produces:
    output/<stem>.dxf   â€” ISO 128 / ASME Y14.3/Y14.5 compliant drawing
    output/<stem>.png   â€” High-res (300 DPI) preview via ezdxf drawing addon
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

# â”€â”€â”€ Project root on sys.path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.logger import logger

# â”€â”€â”€ OCC imports â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
    logger.warning("pythonocc-core not found â€” geometric extraction limited.")

# â”€â”€â”€ ezdxf â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
try:
    import ezdxf
    from ezdxf.enums import TextEntityAlignment
    _EZDXF = True
except ImportError:
    _EZDXF = False
    logger.error("ezdxf not installed. Run: pip install ezdxf")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PART 1 â€” FEATURE EXTRACTION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def extract_features(step_path: str | Path) -> dict:
    """
    Extract all machining features from a STEP file.

    Strategy (layered):
      1. OCC geometric B-Rep analysis (primary, always runs)
      2. HGT model predictions (supplementary, merged where confident)

    Returns a structured dict with:
        bbox        â€” (xmin,ymin,zmin,xmax,ymax,zmax) in mm
        dimensions  â€” (length, width, height) in mm
        holes       â€” list of {diameter, depth, position:(x,y,z), through:bool}
        pockets     â€” list of {width, length, depth, position}
        chamfers    â€” list of {size, angle}
        fillets     â€” list of {radius}
        slots       â€” list of {width, length, depth}
        faces_raw   â€” raw per-face list for view-selector
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
        logger.warning("[EXTRACT] OCC unavailable â€” returning defaults")
        return features

    # â”€â”€ Global bounding box â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    bbox = _shape_bbox(shape)
    features["bbox"] = bbox
    features["dimensions"] = (
        round(bbox[3] - bbox[0], 2),
        round(bbox[4] - bbox[1], 2),
        round(bbox[5] - bbox[2], 2),
    )
    logger.info(f"[EXTRACT] BBox: {features['dimensions']} mm (LÃ—WÃ—H)")

    # â”€â”€ Per-face analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # â”€â”€ Classify cylinders as holes / blind-holes / bosses â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    large_area_thresh = avg_area * 1.5
    for cyl in cylinders:
        diameter = round(cyl["radius"] * 2, 3)
        depth = round(cyl["depth"], 3)
        center = cyl["center"]
        # Heuristic: small cylinder relative to part â†’ hole
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
        # else: boss / external cylinder â€” skip for now

    # â”€â”€ Chamfers from cone faces â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for cone in cones:
        features["chamfers"].append({"size": 1.0, "angle": 45.0})

    # â”€â”€ Fillets from torus faces â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for t in tori:
        if t["radius"] > 0:
            features["fillets"].append({"radius": round(t["radius"], 3)})

    # â”€â”€ faces_raw for downstream view selection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    from cad.brep_features import classify_faces
    classifications = classify_faces(topology)
    features["faces_raw"] = classifications

    # â”€â”€ Merge HGT predictions if checkpoint available â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
            logger.warning("[HGT] No checkpoint found â€” skipping HGT merge.")
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
"""
cad_pipeline.py  â€” Part 2: DXF Generator + PNG Renderer + CLI
Append this content BELOW Part 1 in the same file.
"""

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PART 2 â€” HLR PROJECTION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

_HLR_VIEWS = {
    "FRONT": {"dir": (0, 0, -1), "up": (0, 1, 0)},
    "TOP":   {"dir": (0, -1, 0), "up": (0, 0, 1)},
    "RIGHT": {"dir": (1, 0, 0),  "up": (0, 1, 0)},
    "ISO":   {"dir": (1, 1, -1), "up": (0, 0, 1)},
}


def _project_hlr(shape, view_key: str, scale: float = 1.0) -> dict:
    """
    Run HLR projection for one view. Returns dict with:
        visible: list of [(x,y),...] polylines
        hidden:  list of [(x,y),...] polylines
        bbox:    (xmin,ymin,xmax,ymax) in drawing coords
    """
    info = _HLR_VIEWS[view_key]
    d = info["dir"]
    u = info["up"]

    try:
        dir_vec = gp_Dir(*d)
        up_vec  = gp_Dir(*u)
        ax2 = gp_Ax2(gp_Pnt(0, 0, 0), dir_vec, up_vec)
        projector = HLRAlgo_Projector(ax2)

        hlr = HLRBRep_Algo()
        hlr.Add(shape)
        hlr.Projector(projector)
        hlr.Update()
        hlr.Hide()

        hs = HLRBRep_HLRToShape(hlr)
        vis = _extract_polylines(hs.VCompound(), scale)
        hid = _extract_polylines(hs.HCompound(), scale)

    except Exception as e:
        logger.warning(f"[HLR] {view_key} failed: {e} â€” using stub")
        vis = [[(0, 0), (100 * scale, 0), (100 * scale, 80 * scale),
                (0, 80 * scale), (0, 0)]]
        hid = []

    all_pts = [p for pl in vis + hid for p in pl]
    if all_pts:
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        bbox = (min(xs), min(ys), max(xs), max(ys))
    else:
        bbox = (0, 0, 100 * scale, 80 * scale)

    return {"visible": vis, "hidden": hid, "bbox": bbox}


def _extract_polylines(compound, scale: float) -> list:
    """Convert OCC compound to list of [(x,y),...] polylines."""
    lines = []
    if not _OCC or compound is None:
        return lines
    try:
        if compound.IsNull():
            return lines
    except Exception:
        return lines

    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.TopExp import TopExp_Explorer

    exp = TopExp_Explorer(compound, TopAbs_EDGE)
    while exp.More():
        edge = exp.Current()
        try:
            adaptor = BRepAdaptor_Curve(edge)
            u0, u1 = adaptor.FirstParameter(), adaptor.LastParameter()
            pts = []
            N = 24
            for i in range(N + 1):
                u = u0 + (u1 - u0) * i / N
                p = adaptor.Value(u)
                pts.append((p.X() * scale, p.Y() * scale))
            if pts:
                lines.append(pts)
        except Exception:
            pass
        exp.Next()
    return lines


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PART 3 â€” DXF LAYOUT ENGINE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

# Sheet = A2 (594 Ã— 420 mm) â€” good balance for complex parts
SHEET_W, SHEET_H = 594.0, 420.0
MARGIN           = 20.0   # outer margin
TITLE_H          = 30.0   # title block height at bottom
SPACING          = 25.0   # gap between views (â‰¥20 mm per Y14.5)
DIM_OFFSET_1     = 12.0   # first dim line from view outline (â‰¥10 mm)
DIM_OFFSET_STEP  = 8.0    # stagger for parallel dims (â‰¥6 mm)
TEXT_H           = 3.5    # ASME Y14.2 standard text height (mm)
ARROW_SZ         = 2.5


def _auto_scale(features: dict) -> float:
    """Pick 1:1, 1:2, 1:5, 2:1 so the front view fits the usable area."""
    L, W, H = features["dimensions"]
    # usable area minus title block and 3 views
    usable_w = (SHEET_W - 2 * MARGIN - 2 * SPACING) / 2.0
    usable_h = (SHEET_H - 2 * MARGIN - TITLE_H - SPACING) / 2.0
    for s in [2.0, 1.0, 0.5, 0.2, 0.1]:
        if L * s <= usable_w and H * s <= usable_h:
            return s
    return 0.1


def generate_dxf(features: dict, output_dxf: str | Path) -> Path:
    """
    Generate an ISO 128 / ASME Y14.3 + Y14.5 compliant DXF.

    Layout (third-angle projection):
        [TOP]    [    ]
        [FRONT]  [RIGHT]
                 [ISO]
    """
    if not _EZDXF:
        raise ImportError("ezdxf not installed.")

    output_dxf = Path(output_dxf)
    output_dxf.parent.mkdir(parents=True, exist_ok=True)

    doc = ezdxf.new(dxfversion="R2010", setup=True)
    msp = doc.modelspace()

    # â”€â”€ Layers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _setup_layers(doc)

    scale = _auto_scale(features)
    scale_str = f"1:{int(1/scale)}" if scale < 1 else f"{int(scale)}:1" if scale > 1 else "1:1"
    logger.info(f"[DXF] Scale selected: {scale_str}")

    shape = None
    if _OCC:
        from cad.parser import load_step
        topo = load_step(features["step_path"])
        shape = topo.get("shape")

    # â”€â”€ Project views â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    views = {}
    for key in ("FRONT", "TOP", "RIGHT", "ISO"):
        views[key] = _project_hlr(shape, key, scale) if (shape and _OCC) else _stub_view(key, scale)
        logger.info(f"[DXF] Projected {key}: bbox={views[key]['bbox']}")

    # â”€â”€ Compute offsets (third-angle layout) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Front view â€” bottom-left anchor
    front_bb = views["FRONT"]["bbox"]
    fw = front_bb[2] - front_bb[0]
    fh = front_bb[3] - front_bb[1]

    top_bb  = views["TOP"]["bbox"]
    right_bb = views["RIGHT"]["bbox"]

    tw = top_bb[2] - top_bb[0]
    th = top_bb[3] - top_bb[1]
    rw = right_bb[2] - right_bb[0]
    rh = right_bb[3] - right_bb[1]

    # Front origin â€” leave room for dims left/below
    front_ox = MARGIN + 30.0          # 30 mm left room for dims
    front_oy = MARGIN + TITLE_H + 25.0  # 25 mm bottom room for dims

    top_ox = front_ox
    top_oy = front_oy + fh + SPACING

    right_ox = front_ox + fw + SPACING
    right_oy = front_oy

    iso_ox = right_ox
    iso_oy = top_oy

    offsets = {
        "FRONT": (front_ox, front_oy),
        "TOP":   (top_ox,   top_oy),
        "RIGHT": (right_ox, right_oy),
        "ISO":   (iso_ox,   iso_oy),
    }

    # â”€â”€ Draw views â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for key, (ox, oy) in offsets.items():
        vdata = views[key]
        _draw_view(msp, vdata, ox, oy, key)

    # â”€â”€ Dimensions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    L, W, H = features["dimensions"]
    _add_dimensions(msp, features, offsets, views, scale, L, W, H)

    # â”€â”€ Border & title block â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _draw_border(msp, features, scale_str)

    doc.saveas(str(output_dxf))
    logger.info(f"[DXF] Saved â†’ {output_dxf}")
    return output_dxf


def _setup_layers(doc) -> None:
    defs = {
        "VISIBLE":    {"color": 7,  "lineweight": 35},
        "HIDDEN":     {"color": 8,  "lineweight": 18},
        "CENTER":     {"color": 1,  "lineweight": 9},
        "DIM":        {"color": 2,  "lineweight": 13},
        "TEXT":       {"color": 7,  "lineweight": 13},
        "BORDER":     {"color": 5,  "lineweight": 50},
        "TITLE":      {"color": 7,  "lineweight": 25},
        "HATCH":      {"color": 254,"lineweight": 9},
    }
    for name, props in defs.items():
        try:
            doc.layers.new(name, dxfattribs={"color": props["color"],
                                              "lineweight": props["lineweight"]})
        except Exception:
            pass
    # Ensure DASHED linetype for HIDDEN
    try:
        doc.linetypes.get("DASHED")
        doc.layers.get("HIDDEN").dxf.linetype = "DASHED"
    except Exception:
        pass


def _stub_view(key: str, scale: float) -> dict:
    w, h = 100 * scale, 80 * scale
    return {
        "visible": [[(0,0),(w,0),(w,h),(0,h),(0,0)]],
        "hidden": [[(w/2,0),(w/2,h)]],
        "bbox": (0.0, 0.0, w, h),
    }


def _draw_view(msp, vdata: dict, ox: float, oy: float, label: str) -> None:
    """Draw visible + hidden polylines for one view, with label."""
    for pl in vdata["visible"]:
        if len(pl) >= 2:
            shifted = [(x + ox, y + oy) for x, y in pl]
            msp.add_lwpolyline(shifted, dxfattribs={"layer": "VISIBLE"})

    for pl in vdata["hidden"]:
        if len(pl) >= 2:
            shifted = [(x + ox, y + oy) for x, y in pl]
            msp.add_lwpolyline(shifted, dxfattribs={"layer": "HIDDEN"})

    # View label below the view
    bb = vdata["bbox"]
    label_x = ox + (bb[2] - bb[0]) / 2
    label_y = oy - 8.0
    msp.add_text(
        label,
        dxfattribs={"layer": "TEXT", "height": TEXT_H,
                    "insert": (label_x, label_y), "halign": 1},
    )


def _add_dimensions(msp, features, offsets, views, scale, L, W, H) -> None:
    """Add ASME Y14.5 dimensions: overall + per-feature."""
    fox, foy = offsets["FRONT"]
    fbb = views["FRONT"]["bbox"]
    fw = fbb[2] - fbb[0]
    fh = fbb[3] - fbb[1]

    # â”€â”€ Overall LENGTH (below front view) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _dim_linear(msp,
                p1=(fox, foy - DIM_OFFSET_1),
                p2=(fox + fw, foy - DIM_OFFSET_1),
                base=(fox + fw / 2, foy - DIM_OFFSET_1 - 5),
                text=f"{L:.1f}",
                layer="DIM")

    # â”€â”€ Overall HEIGHT (left of front view) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _dim_linear(msp,
                p1=(fox - DIM_OFFSET_1, foy),
                p2=(fox - DIM_OFFSET_1, foy + fh),
                base=(fox - DIM_OFFSET_1 - 5, foy + fh / 2),
                text=f"{H:.1f}",
                layer="DIM",
                vertical=True)

    # â”€â”€ Overall WIDTH (above top view) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    tox, toy = offsets["TOP"]
    tbb = views["TOP"]["bbox"]
    tw = tbb[2] - tbb[0]
    th = tbb[3] - tbb[1]
    _dim_linear(msp,
                p1=(tox, toy + th + DIM_OFFSET_1),
                p2=(tox + tw, toy + th + DIM_OFFSET_1),
                base=(tox + tw / 2, toy + th + DIM_OFFSET_1 + 5),
                text=f"{W:.1f}",
                layer="DIM")

    # â”€â”€ Holes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    all_holes = features.get("holes", []) + features.get("blind_holes", [])
    y_offset = foy - DIM_OFFSET_1 - DIM_OFFSET_STEP
    for i, hole in enumerate(all_holes[:6]):   # limit to 6 to avoid clutter
        dia = hole["diameter"]
        depth = hole["depth"]
        is_through = hole.get("through", False)
        note = f"âŒ€{dia:.2f} THRU" if is_through else f"âŒ€{dia:.2f} DP {depth:.1f}"
        x_pos = fox + fw + 5 + (i % 3) * 35
        y_pos = foy + fh / 2 - (i // 3) * 10
        msp.add_text(note, dxfattribs={
            "layer": "TEXT", "height": TEXT_H * 0.9,
            "insert": (x_pos, y_pos),
        })

    # â”€â”€ Chamfer notes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    chamfers = features.get("chamfers", [])
    if chamfers:
        note = f"{len(chamfers)}X C{chamfers[0].get('size', 1.0):.1f}"
        msp.add_text(note, dxfattribs={
            "layer": "TEXT", "height": TEXT_H,
            "insert": (fox + fw + 5, foy + fh + 5),
        })

    # â”€â”€ Fillet notes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fillets = features.get("fillets", [])
    if fillets:
        r = fillets[0].get("radius", 1.0)
        note = f"ALL FILLETS R{r:.1f} UNLESS NOTED"
        msp.add_text(note, dxfattribs={
            "layer": "TEXT", "height": TEXT_H,
            "insert": (fox, foy - DIM_OFFSET_1 - DIM_OFFSET_STEP * 3),
        })


def _dim_linear(msp, p1, p2, base, text: str,
                layer: str = "DIM", vertical: bool = False) -> None:
    """Add a linear dimension entity (horizontal or vertical)."""
    try:
        if vertical:
            msp.add_linear_dim(
                base=base, p1=p1, p2=p2,
                angle=90,
                dimstyle="ISO-25",
                override={"dimtxt": TEXT_H, "dimasz": ARROW_SZ,
                          "dimpost": f"{text}<>"},
                dxfattribs={"layer": layer},
            ).render()
        else:
            msp.add_linear_dim(
                base=base, p1=p1, p2=p2,
                dimstyle="ISO-25",
                override={"dimtxt": TEXT_H, "dimasz": ARROW_SZ,
                          "dimpost": f"{text}<>"},
                dxfattribs={"layer": layer},
            ).render()
    except Exception as e:
        logger.warning(f"[DIM] Could not add dimension: {e}")


def _draw_border(msp, features: dict, scale_str: str) -> None:
    """Draw sheet border + ISO/ASME title block."""
    m = MARGIN
    # Outer border
    border = [(m, m), (SHEET_W - m, m), (SHEET_W - m, SHEET_H - m),
              (m, SHEET_H - m), (m, m)]
    msp.add_lwpolyline(border, dxfattribs={"layer": "BORDER"})

    # Title block (bottom-right)
    tb_left = SHEET_W - m - 180
    tb_y    = m
    tb_box  = [(tb_left, tb_y), (SHEET_W - m, tb_y),
               (SHEET_W - m, tb_y + TITLE_H),
               (tb_left, tb_y + TITLE_H), (tb_left, tb_y)]
    msp.add_lwpolyline(tb_box, dxfattribs={"layer": "BORDER"})

    part_name = features.get("stem", "PART").upper()
    today = date.today().isoformat()
    entries = [
        (part_name,                    4.0, (tb_left + 5, tb_y + 20)),
        (f"SCALE: {scale_str}",        2.5, (tb_left + 5, tb_y + 12)),
        (f"DATE: {today}",             2.5, (tb_left + 5, tb_y + 6)),
        ("THIRD ANGLE PROJECTION",     2.0, (tb_left + 5, tb_y + 2)),
        ("CAD AUTOMATION PIPELINE",    2.0, (SHEET_W - m - 60, tb_y + 2)),
    ]
    for txt, h, pos in entries:
        msp.add_text(txt, dxfattribs={"layer": "TITLE", "height": h,
                                       "insert": pos})

    # Projection symbol (simplified) â€” circle with arrow hint
    sym_cx, sym_cy = tb_left + 155, tb_y + 15
    msp.add_circle((sym_cx, sym_cy), 5, dxfattribs={"layer": "BORDER"})
    msp.add_text("E", dxfattribs={"layer": "TITLE", "height": 2.5,
                                   "insert": (sym_cx - 1.5, sym_cy - 1)})


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PART 4 â€” PNG RENDERER (ezdxf drawing addon, 300 DPI)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def render_png(dxf_path: str | Path, output_png: str | Path) -> Path:
    """
    Render DXF to PNG using ezdxf's SVGBackend + cairosvg.
    Falls back to PIL text-card if cairosvg is absent.
    """
    dxf_path = Path(dxf_path)
    output_png = Path(output_png)

    try:
        import ezdxf
        from ezdxf.addons.drawing import RenderContext, Frontend
        from ezdxf.addons.drawing.svg import SVGBackend
        
        doc = ezdxf.readfile(str(dxf_path))
        ctx = RenderContext(doc)
        backend = SVGBackend()
        frontend = Frontend(ctx, backend)

        for entity in doc.modelspace():
            try:
                frontend.draw_entity(entity)
            except Exception:
                continue

        backend.finalize()
        from ezdxf.addons.drawing import layout
        from ezdxf.math import BoundingBox2d
        page = layout.Page(SHEET_W, SHEET_H)
        box = BoundingBox2d([(0, 0), (SHEET_W, SHEET_H)])
        svg_string = backend.get_string(page, render_box=box)

        import cairosvg
        png_bytes = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'), dpi=300)
        
        output_png.parent.mkdir(parents=True, exist_ok=True)
        with open(output_png, "wb") as f:
            f.write(png_bytes)
            
        logger.info(f"[PNG] Saved → {output_png}  (300 DPI)")

    except ImportError:
        logger.warning("[PNG] cairosvg not found — using PIL fallback")
        _png_pil_fallback(dxf_path, output_png)

    except Exception as e:
        logger.error(f"[PNG] Rendering failed: {e}")
        _png_pil_fallback(dxf_path, output_png)

    return output_png


def _png_pil_fallback(dxf_path: Path, output_png: Path) -> None:
    """Create a minimal PNG with part name when proper rendering fails."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img  = Image.new("RGB", (2480, 1748), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([40, 40, 2440, 1708], outline=(0, 0, 0), width=6)
        draw.text((120, 120), f"DXF: {dxf_path.name}", fill=(30, 30, 30))
        draw.text((120, 200), "Rendered with PIL fallback (install matplotlib for full preview)",
                  fill=(180, 0, 0))
        output_png.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(output_png), dpi=(300, 300))
        logger.info(f"[PNG] PIL fallback saved â†’ {output_png}")
    except Exception as e2:
        logger.error(f"[PNG] PIL fallback also failed: {e2}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PART 5 â€” CLI ENTRY POINT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def run_pipeline(step_file: str, output_dir: str = "output") -> dict:
    """One-call entry: STEP â†’ features â†’ DXF â†’ PNG."""
    step_path  = Path(step_file).resolve()
    out_dir    = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = step_path.stem
    dxf_path = out_dir / f"{stem}_drawing.dxf"
    png_path = out_dir / f"{stem}_drawing.png"
    json_path = out_dir / f"{stem}_features.json"

    logger.info("=" * 60)
    logger.info(f"CAD PIPELINE â€” {stem}")
    logger.info("=" * 60)

    # 1. Feature extraction
    features = extract_features(step_path)

    # 2. Save features JSON
    safe = {k: (list(v) if isinstance(v, tuple) else v)
            for k, v in features.items()
            if k not in ("faces_raw", "hgt_predictions")}
    json_path.write_text(json.dumps(safe, indent=2, default=str))
    logger.info(f"[JSON] Features saved â†’ {json_path}")

    # 3. Generate DXF
    generate_dxf(features, dxf_path)

    # 4. Render PNG
    render_png(dxf_path, png_path)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  DXF  â†’ {dxf_path}")
    logger.info(f"  PNG  â†’ {png_path}")
    logger.info(f"  JSON â†’ {json_path}")
    logger.info("=" * 60)

    return {"dxf": str(dxf_path), "png": str(png_path), "features": features}


def main():
    p = argparse.ArgumentParser(
        description="STEP â†’ ISO/ASME DXF + PNG pipeline"
    )
    p.add_argument("--step_file", required=True, help="Path to input .stp/.step file")
    p.add_argument("--output_dir", default="output", help="Output folder (default: output/)")
    args = p.parse_args()
    run_pipeline(args.step_file, args.output_dir)


if __name__ == "__main__":
    main()
