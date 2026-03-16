"""Configuration settings for the CAD Drawing Generator."""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Settings:
    """
    Central configuration store for the CAD Drawing Generator pipeline.

    All attributes have sensible defaults so the system runs without
    any environment variables.
    """

    # --- Neo4j ----------------------------------------------------------------
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # --- Paths ----------------------------------------------------------------
    OUTPUT_DIR: str = "output"
    STEP_FILE_PATH: str = ""
    MODEL_DIR: str = "models"

    # --- Drawing sheet sizes (mm) --------------------------------------------
    SHEET_SIZES: Dict[str, tuple] = field(
        default_factory=lambda: {
            "A0": (1189, 841),
            "A1": (841, 594),
            "A2": (594, 420),
            "A3": (420, 297),
            "A4": (297, 210),
        }
    )

    # --- Recognised manufacturing feature classes ----------------------------
    FEATURE_CLASSES: List[str] = field(
        default_factory=lambda: [
            "Hole",
            "Pocket",
            "Slot",
            "Boss",
            "Fillet",
            "Chamfer",
            "Rib",
            "Unknown",
        ]
    )

    # --- Orthographic views ---------------------------------------------------
    VIEWS: List[str] = field(
        default_factory=lambda: ["FRONT", "TOP", "RIGHT", "ISOMETRIC"]
    )

    # --- AI model paths -------------------------------------------------------
    GNN_MODEL_PATH: str = "models/gnn_feature_model.pt"
    REGRESSION_MODEL_PATH: str = "models/feature_regression_model.pt"
    VIEW_MODEL_PATH: str = "models/view_prediction_model.json"
    RANKING_MODEL_PATH: str = "models/dimension_ranking_model.json"

    # --- Pipeline tweaks ------------------------------------------------------
    MAX_DIMENSIONS: int = 100
    OVERLAP_THRESHOLD: float = 5.0  # mm – minimum gap between dimension lines
    DEFAULT_SCALE: float = 1.0
    FONT_SIZE: float = 2.5  # DXF text height in mm


# Module-level singleton accessed by the rest of the package.
settings = Settings()
