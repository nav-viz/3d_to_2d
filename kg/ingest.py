"""
kg/ingest.py
============
STEP ingestion pipeline with fixed OCC calls for pythonocc-core 7.7.2+
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from utils.logger import logger

from cad.parser import load_step
from cad.brep_features import classify_faces
from kg.neo4j_client import Neo4jClient


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_step(
    step_path: str | Path,
    client: Neo4jClient,
    cfg: dict[str, Any],
) -> str:
    step_path = Path(step_path)
    model_id = _model_id_from_path(step_path)
    logger.info(f"Ingesting model_id={model_id}  source={step_path.name}")

    topology = load_step(step_path)

    brep_features = classify_faces(topology)
    logger.info(f"Rule-based MFR: {len(brep_features)} face classifications.")

    client.upsert_model({
        "model_id": model_id,
        "source_file": str(step_path),
        "num_faces": topology["num_faces"],
        "num_edges": topology["num_edges"],
        "num_vertices": topology["num_vertices"],
    })

    client.upsert_faces(model_id, topology["faces"])
    client.upsert_edges(model_id, topology["edges"])

    adjacency = _build_adjacency(topology)
    client.upsert_adjacency(model_id, adjacency)
    logger.info(f"Adjacency pairs written: {len(adjacency)}")

    predictions = [
        {
            "face_id": f["face_id"],
            "feature_type": f["feature_type"],
            "confidence": f["confidence"],
            "params": f["params"],
        }
        for f in brep_features
    ]
    client.write_predictions(model_id, predictions, model_type="rule_based")

    logger.success(f"Ingest complete: model_id={model_id}")
    return model_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_id_from_path(path: Path) -> str:
    canonical = str(path.resolve())
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _build_adjacency(topology: dict[str, Any]) -> list[tuple[int, int, str]]:
    """OCC adjacency first, fallback to edge.face_ids"""
    try:
        occ_pairs = build_adjacency_from_occ(topology)
        if occ_pairs:
            logger.success(f"✅ OCC adjacency built: {len(occ_pairs)} directed pairs")
            return occ_pairs
    except Exception as e:
        logger.warning(f"OCC adjacency failed: {e}. Using fallback.")

    logger.info("Using fallback adjacency from edge.face_ids")
    pairs: list[tuple[int, int, str]] = []
    for edge in topology.get("edges", []):
        fids = edge.get("face_ids", [])
        conv = str(edge.get("convexity", "unknown")).lower()
        if len(fids) >= 2:
            a, b = fids[0], fids[1]
            pairs.append((a, b, conv))
            if a != b:
                pairs.append((b, a, conv))
    return pairs


def build_adjacency_from_occ(topology: dict[str, Any]) -> list[tuple[int, int, str]]:
    """Correct OCC-based face-to-face adjacency using TopExp_Explorer per edge."""
    try:
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
        from OCC.Core.TopTools import (
            TopTools_IndexedDataMapOfShapeListOfShape,
            TopTools_IndexedMapOfShape,
            TopTools_ListIteratorOfListOfShape,
        )
        from OCC.Core.TopoDS import topods
    except ImportError:
        logger.warning("OCC not available – adjacency skipped.")
        return []

    shape = topology.get("shape")
    if shape is None:
        logger.warning("No OCC shape – adjacency skipped.")
        return []

    # Build edge → list_of_faces map (the reliable way)
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    from OCC.Core.TopExp import topexp_MapShapesAndAncestors
    topexp_MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    # Build a robust face-shape index instead of relying on HashCode.
    face_map = TopTools_IndexedMapOfShape()
    face_index_to_id: dict[int, int] = {}
    for f in topology.get("faces", []):
        occ_face = f.get("occ_face")
        if occ_face is not None:
            try:
                occ_face = topods.Face(occ_face)
            except Exception:
                # Keep original shape if conversion is not needed.
                pass
            if not face_map.Contains(occ_face):
                face_map.Add(occ_face)
            idx = face_map.FindIndex(occ_face)
            if idx > 0:
                face_index_to_id[idx] = f["id"]

    pairs: list[tuple[int, int, str]] = []

    for i in range(1, edge_face_map.Extent() + 1):
        faces_of_edge = edge_face_map.FindFromIndex(i)
        fid_list = []
        # pythonocc wrappers vary by version: support both .begin() and explicit iterator class.
        try:
            it = faces_of_edge.begin()
        except AttributeError:
            it = TopTools_ListIteratorOfListOfShape(faces_of_edge)

        while it.More():
            face_shape = it.Value()
            try:
                face_shape = topods.Face(face_shape)
            except Exception:
                pass
            try:
                idx = face_map.FindIndex(face_shape)
            except Exception:
                idx = -1
            if idx > 0 and idx in face_index_to_id:
                fid = face_index_to_id[idx]
                if fid not in fid_list:
                    fid_list.append(fid)
            it.Next()

        if len(fid_list) >= 2:
            # For now use "unknown" convexity; you can compute dihedral angle later
            fa, fb = fid_list[0], fid_list[1]
            pairs.append((fa, fb, "unknown"))
            if fa != fb:                     # avoid self-pairs if any
                pairs.append((fb, fa, "unknown"))

    logger.info(f"✅ OCC adjacency built: {len(pairs)//2} unique shared edges → {len(pairs)} directed pairs")
    return pairs
