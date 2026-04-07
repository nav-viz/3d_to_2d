"""
inference/writeback.py
======================
Write GNN predictions back to the Neo4j Knowledge Graph as
Feature nodes linked via PREDICTED_AS relationships.

Also supports updating confidence scores for existing Feature nodes
when re-running inference after retraining.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from utils.logger import logger

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kg.neo4j_client import Neo4jClient


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_predictions_to_kg(
    model_id: str,
    predictions: list[dict[str, Any]],
    client: Neo4jClient,
    model_type: str = "hgt",
    overwrite: bool = True,
) -> None:
    """
    Persist GNN predictions to Neo4j.

    Parameters
    ----------
    model_id    : str – SHA-16 model identifier from ingest
    predictions : list of prediction dicts from ``inference.predict.predict``
    client      : Neo4jClient
    model_type  : str  ("hgt" | "graphsage" | "rule_based")
    overwrite   : bool – if True, delete existing PREDICTED_AS relations
                  for this model_id + model_type before writing new ones.
    """
    if overwrite:
        _delete_existing(model_id, model_type, client)

    # Filter out low-confidence predictions before write-back
    filtered = [
        p for p in predictions
        if p.get("feature_type", "Low_Confidence") != "Low_Confidence"
    ]
    skipped = len(predictions) - len(filtered)
    if skipped:
        logger.info(
            f"Skipping {skipped} low-confidence predictions (not written to KG)."
        )

    client.write_predictions(model_id, filtered, model_type=model_type)
    logger.success(
        f"Write-back complete: {len(filtered)} predictions → Neo4j  "
        f"(model_id={model_id}, model_type={model_type})"
    )


def _delete_existing(
    model_id: str,
    model_type: str,
    client: Neo4jClient,
) -> None:
    """Remove previous PREDICTED_AS relationships for the given model."""
    cypher = """
    MATCH (f:Face {model_id: $model_id})-[r:PREDICTED_AS {model_type: $model_type}]->(feat:Feature)
    DETACH DELETE feat
    """
    with client.session() as sess:
        sess.run(cypher, model_id=model_id, model_type=model_type)
    logger.debug(
        f"Cleared existing PREDICTED_AS ({model_type}) for model_id={model_id}"
    )


def read_back_features(
    model_id: str,
    client: Neo4jClient,
) -> list[dict]:
    """
    Retrieve and return all Feature nodes for a model from Neo4j.
    Useful for validation / debugging.
    """
    features = client.get_features(model_id)
    logger.info(
        f"Read-back: {len(features)} features for model_id={model_id}"
    )
    return features
