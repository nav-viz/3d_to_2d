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


def _build_adjacency(
    topology: dict[str, Any],
) -> list[tuple[int, int, str]]:
    """
    Derive face-to-face adjacency from shared edges.

    For each edge, if it has ≥2 face_ids we emit an ADJACENT_TO pair.
    When OCC is unavailable (stub mode) we emit no adjacency.

    Returns
    -------
    list of (face_id_a, face_id_b, convexity_str)
    """
    pairs: list[tuple[int, int, str]] = []
    for edge in topology.get("edges", []):
        fids = edge.get("face_ids", [])
        conv = edge.get("convexity", "unknown")
        if len(fids) >= 2:
            # Emit both directions for undirected adjacency
            pairs.append((fids[0], fids[1], conv))
            pairs.append((fids[1], fids[0], conv))
    return pairs


def build_adjacency_from_occ(topology: dict[str, Any]) -> list[tuple[int, int, str]]:
    """
    Use OCC TopExp to map each edge to its adjacent faces and determine
    convexity (convex / concave / smooth).

    Extension point: integrate OCC face-edge-face walker here for full BRep
    adjacency rather than the stub version above.
    """
    try:
        from OCC.Core.TopExp import TopExp_Explorer, topexp_MapShapesAndAncestors
        from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
        from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    except ImportError:
        logger.warning("OCC not available – adjacency skipped.")
        return []

    shape = topology.get("shape")
    if shape is None:
        return []

    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp_MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    # Build hash → face_id lookup
    face_hash_to_id: dict[int, int] = {
        f["occ_face"].HashCode(2**31): f["id"]
        for f in topology["faces"]
        if f.get("occ_face") is not None
    }

    pairs: list[tuple[int, int, str]] = []
    for i in range(1, edge_face_map.Extent() + 1):
        faces_of_edge = edge_face_map.FindFromIndex(i)
        fid_list = []
        it = faces_of_edge.begin()
        while it != faces_of_edge.end():
            h = it.Value().HashCode(2**31)
            if h in face_hash_to_id:
                fid_list.append(face_hash_to_id[h])
            it.Next()
        if len(fid_list) == 2:
            # TODO: determine convexity from dihedral angle
            pairs.append((fid_list[0], fid_list[1], "unknown"))
            pairs.append((fid_list[1], fid_list[0], "unknown"))

    return pairs
