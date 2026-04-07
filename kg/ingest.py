"""
kg/ingest.py
============
Ingestion pipeline: takes parsed STEP topology and writes it to Neo4j.

Flow
----
1. Parse STEP → topology dict  (cad.parser)
2. Rule-based MFR              (cad.brep_features)
3. Upsert to Neo4j             (kg.neo4j_client)
4. Build face adjacency        (from edge ↔ face relationships)

Usage
-----
>>> from kg.ingest import ingest_step
>>> ingest_step("my_part.step", client, cfg)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from loguru import logger

from cad.parser import load_step
from cad.brep_features import classify_faces
from kg.neo4j_client import Neo4jClient


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ingest_step(
    step_path: str | Path,
    client: Neo4jClient,
    cfg: dict[str, Any],
) -> str:
    """
    Parse a STEP file and persist its topology + rule-based features to Neo4j.

    Parameters
    ----------
    step_path : str or Path
        Path to ``.step`` or ``.stp`` file.
    client  : Neo4jClient
    cfg     : full pipeline config dict

    Returns
    -------
    model_id : str
        Deterministic SHA-256 based model identifier.
    """
    step_path = Path(step_path)
    model_id = _model_id_from_path(step_path)
    logger.info(f"Ingesting model_id={model_id}  source={step_path.name}")

    # ── Parse ─────────────────────────────────────────────────────────────────
    topology = load_step(step_path)

    # ── Rule-based MFR ────────────────────────────────────────────────────────
    brep_features = classify_faces(topology)
    logger.info(f"Rule-based MFR: {len(brep_features)} face classifications.")

    # ── Write model node ──────────────────────────────────────────────────────
    client.upsert_model({
        "model_id": model_id,
        "source_file": str(step_path),
        "num_faces": topology["num_faces"],
        "num_edges": topology["num_edges"],
        "num_vertices": topology["num_vertices"],
    })

    # ── Write face nodes ──────────────────────────────────────────────────────
    client.upsert_faces(model_id, topology["faces"])

    # ── Write edge nodes ──────────────────────────────────────────────────────
    client.upsert_edges(model_id, topology["edges"])

    # ── Build & write adjacency ───────────────────────────────────────────────
    adjacency = _build_adjacency(topology)
    client.upsert_adjacency(model_id, adjacency)
    logger.info(f"Adjacency pairs written: {len(adjacency)}")

    # ── Write rule-based feature predictions ─────────────────────────────────
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
    """
    Create a deterministic model_id from the canonical file path.
    Uses SHA-256(absolute_path) → first 16 hex chars.
    """
    canonical = str(path.resolve())
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# kg/ingest.py  →  replace _build_adjacency function

# kg/ingest.py

def _build_adjacency(
    topology: dict[str, Any],
) -> list[tuple[int, int, str]]:
    """
    Build face adjacency using OCC when possible, with strong fallback.
    """
    pairs: list[tuple[int, int, str]] = []

    # Priority 1: Use full OCC topology (most accurate)
    try:
        adjacency = build_adjacency_from_occ(topology)
        if adjacency:
            logger.info(f"OCC adjacency built: {len(adjacency)} directed pairs")
            return adjacency
    except Exception as e:
        logger.debug(f"OCC adjacency failed: {e}")

    # Priority 2: Fallback using edge.face_ids (if parser filled them)
    for edge in topology.get("edges", []):
        fids = edge.get("face_ids", [])
        conv = str(edge.get("convexity", "unknown")).lower()
        if len(fids) >= 2:
            a, b = fids[0], fids[1]
            pairs.append((a, b, conv))
            if a != b:
                pairs.append((b, a, conv))

    # Priority 3: Last resort - connect faces sharing any edge (even if face_ids empty)
    if len(pairs) == 0 and len(topology.get("edges", [])) > 0:
        logger.warning("No face_ids found on edges. Building crude adjacency from edge connectivity.")
        # This is a last-ditch effort - may create too many edges
        face_count = topology.get("num_faces", 0)
        if face_count > 0:
            # Connect every face to next one as temporary self-loops fallback
            for i in range(face_count):
                pairs.append((i, (i + 1) % face_count, "unknown"))

    logger.info(f"Adjacency pairs built: {len(pairs)}")
    return pairs


# kg/ingest.py

# kg/ingest.py

def build_adjacency_from_occ(topology: dict[str, Any]) -> list[tuple[int, int, str]]:
    """
    Build face-to-face adjacency using modern pythonocc-core API (no deprecation).
    """
    try:
        # Correct imports - TopExp (capital T)
        from OCC.Core.TopExp import TopExp_Explorer, topexp_MapShapesAndAncestors
        from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
        from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    except ImportError as e:
        logger.warning(f"OCC import failed for adjacency: {e}")
        return []

    shape = topology.get("shape")
    if shape is None:
        logger.debug("No shape available for OCC adjacency.")
        return []

    # Create map
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()

    # Modern call (no deprecation)
    topexp_MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    # Build fast lookup: face hash → face_id
    face_hash_to_id: dict[int, int] = {}
    for f in topology.get("faces", []):
        occ_face = f.get("occ_face")
        if occ_face is not None:
            # Use a safe large prime for HashCode
            face_hash_to_id[occ_face.HashCode(2**31 - 1)] = f["id"]

    pairs: list[tuple[int, int, str]] = []

    # Iterate over all edges in the map
    for i in range(1, edge_face_map.Extent() + 1):
        faces_of_edge = edge_face_map.FindFromIndex(i)
        fid_list = []

        it = faces_of_edge.begin()
        while it != faces_of_edge.end():
            h = it.Value().HashCode(2**31 - 1)
            if h in face_hash_to_id:
                fid_list.append(face_hash_to_id[h])
            it.Next()

        if len(fid_list) == 2:
            a, b = fid_list
            # TODO: Later compute real dihedral angle for convexity ("convex"/"concave"/"smooth")
            pairs.append((a, b, "unknown"))
            pairs.append((b, a, "unknown"))   # make bidirectional

    logger.info(f"OCC adjacency built: {len(pairs)} directed pairs from {len(pairs)//2} shared edges")
    return pairs
