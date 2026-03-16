"""
CAD Drawing Generator Pipeline.

Orchestrates all pipeline stages from STEP parsing to DXF/PNG output.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict

from src.ai_models.dimension_ranking_model import DimensionRankingModel
from src.ai_models.feature_regression_model import FeatureRegressionModel
from src.ai_models.gnn_feature_model import GNNFeatureModel
from src.ai_models.view_prediction_model import ViewPredictionModel
from src.brep_graph.graph_builder import GraphBuilder
from src.brep_graph.graph_features import GraphFeatureExtractor
from src.config.settings import settings
from src.database.graph_storage import GraphStorage
from src.database.neo4j_connection import Neo4jConnection
from src.dimensioning.dimension_generator import DimensionGenerator
from src.dimensioning.dimension_optimizer import DimensionOptimizer
from src.drawing.dxf_writer import DXFWriter
from src.drawing.png_renderer import PNGRenderer
from src.drawing.view_projection import ViewProjector
from src.step_parser.geometry_extractor import GeometryExtractor
from src.step_parser.parse_step import StepParser
from src.utils.logger import get_logger


class CADPipeline:
    """
    End-to-end CAD drawing generation pipeline.

    Steps
    -----
    1.  Parse STEP file (or generate synthetic model)
    2.  Build BRep graph
    3.  Store graph in Neo4j (or in-memory fallback)
    4.  AI feature recognition (GNN / rule-based)
    5.  Feature parameter prediction (MLP / geometric)
    6.  View selection (XGBoost / rule-based)
    7.  Generate dimension candidates
    8.  Rank and optimise dimensions
    9.  Generate DXF drawing
    10. Render PNG image
    """

    def __init__(self) -> None:
        self._log = get_logger(self.__class__.__name__)

        # Initialise sub-components
        self._parser = StepParser()
        self._graph_builder = GraphBuilder()
        self._gnn = GNNFeatureModel(model_path=settings.GNN_MODEL_PATH)
        self._regressor = FeatureRegressionModel(
            model_path=settings.REGRESSION_MODEL_PATH
        )
        self._view_predictor = ViewPredictionModel(
            model_path=settings.VIEW_MODEL_PATH
        )
        self._dim_gen = DimensionGenerator()
        self._dim_ranker = DimensionRankingModel(
            model_path=settings.RANKING_MODEL_PATH
        )
        self._dim_optimizer = DimensionOptimizer()
        self._projector = ViewProjector()
        self._dxf_writer = DXFWriter()
        self._png_renderer = PNGRenderer()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, step_file_path: str = "") -> Dict:
        """
        Execute the full pipeline.

        Args:
            step_file_path: Path to the STEP file.  When empty a synthetic
                            demo model is used.

        Returns:
            Dictionary with output file paths and pipeline metadata.
        """
        start_time = datetime.now()
        self._log.info("=" * 60)
        self._log.info("Pipeline START  |  input: '%s'", step_file_path or "<synthetic>")
        self._log.info("=" * 60)

        os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
        part_id = self._derive_part_id(step_file_path)

        # ----------------------------------------------------------
        # Step 1: Parse STEP
        # ----------------------------------------------------------
        self._log.info("[1/10] Parsing STEP file …")
        model = self._parser.parse(step_file_path)
        self._log.info("       %d faces, %d edges, %d vertices.",
                       len(model.faces), len(model.edges), len(model.vertices))

        # ----------------------------------------------------------
        # Step 2: Build BRep graph
        # ----------------------------------------------------------
        self._log.info("[2/10] Building BRep graph …")
        graph = self._graph_builder.build_graph(model)
        extractor = GeometryExtractor(model)
        feat_extractor = GraphFeatureExtractor(graph)
        stats = feat_extractor.compute_graph_statistics()
        complexity = extractor.compute_complexity_score()
        bounding_box = extractor.extract_bounding_box()
        self._log.info("       Nodes=%d  Edges=%d  Complexity=%.2f",
                       graph.num_nodes, graph.num_edges, complexity)

        # ----------------------------------------------------------
        # Step 3: Store in Neo4j
        # ----------------------------------------------------------
        self._log.info("[3/10] Storing graph …")
        with Neo4jConnection(
            settings.NEO4J_URI, settings.NEO4J_USER, settings.NEO4J_PASSWORD
        ) as conn:
            storage = GraphStorage(conn)
            storage.store_part_graph(part_id, graph)

        # ----------------------------------------------------------
        # Step 4: AI feature recognition
        # ----------------------------------------------------------
        self._log.info("[4/10] Running AI feature recognition …")
        feat_result = self._gnn.predict(graph)
        self._log.info("       Feature summary: %s", feat_result["summary"])

        # ----------------------------------------------------------
        # Step 5: Feature parameter prediction
        # ----------------------------------------------------------
        self._log.info("[5/10] Predicting feature parameters …")
        feature_params: Dict = {}
        for node in graph.node_list():
            label = feat_result["face_labels"].get(node.id, "Unknown")
            face_data = {
                "surface_type": node.surface_type,
                "area": node.area,
                "normal": node.normal,
                "center": node.center,
                "bounding_box": node.bounding_box,
                "curvature": node.curvature,
            }
            feature_params[node.id] = self._regressor.predict(face_data, label)

        feat_result["feature_params"] = feature_params

        # ----------------------------------------------------------
        # Step 6: View prediction
        # ----------------------------------------------------------
        self._log.info("[6/10] Predicting drawing views …")
        view_config = self._view_predictor.predict(
            bounding_box,
            sum(feat_result["summary"].values()),
            complexity,
        )
        self._log.info(
            "       Views=%s  Sheet=%s  Scale=%.2f",
            view_config["views"], view_config["sheet_size"], view_config["scale"],
        )

        # ----------------------------------------------------------
        # Step 7: Generate dimensions
        # ----------------------------------------------------------
        self._log.info("[7/10] Generating dimension candidates …")
        candidates = self._dim_gen.generate(graph, feat_result)

        # ----------------------------------------------------------
        # Step 8: Rank and optimise dimensions
        # ----------------------------------------------------------
        self._log.info("[8/10] Ranking & optimising dimensions …")
        candidate_dicts = [
            {
                "id": c.id,
                "dim_type": c.dim_type,
                "value": c.value,
                "start_point": c.start_point,
                "end_point": c.end_point,
                "feature_label": c.feature_label,
            }
            for c in candidates
        ]
        ranked = self._dim_ranker.rank(candidate_dicts)
        selected_dims = self._dim_optimizer.optimize(candidates, ranked)
        self._log.info(
            "       %d → %d dimensions selected.", len(candidates), len(selected_dims)
        )

        # ----------------------------------------------------------
        # Step 9: Generate DXF
        # ----------------------------------------------------------
        self._log.info("[9/10] Generating DXF drawing …")
        views = self._projector.project_all(
            graph,
            view_config["views"],
            sheet_size=view_config["sheet_size"],
            scale=view_config["scale"],
        )
        metadata = {
            "title": os.path.basename(step_file_path) if step_file_path else "DEMO PART",
            "drawing_number": f"DRW-{part_id[:6].upper()}",
            "revision": "A",
            "date": start_time.strftime("%Y-%m-%d"),
        }
        dxf_doc = self._dxf_writer.create_drawing(
            views,
            selected_dims,
            metadata,
            sheet_size=view_config["sheet_size"],
        )
        dxf_path = os.path.join(settings.OUTPUT_DIR, f"{part_id}.dxf")
        self._dxf_writer.save(dxf_doc, dxf_path)

        # ----------------------------------------------------------
        # Step 10: Render PNG
        # ----------------------------------------------------------
        self._log.info("[10/10] Rendering PNG …")
        png_path = os.path.join(settings.OUTPUT_DIR, f"{part_id}.png")
        self._png_renderer.render(dxf_path, png_path)

        # ----------------------------------------------------------
        # Save JSON outputs
        # ----------------------------------------------------------
        features_json = os.path.join(settings.OUTPUT_DIR, f"{part_id}_features.json")
        graph_json = os.path.join(settings.OUTPUT_DIR, f"{part_id}_graph.json")
        self._save_json(
            {
                "part_id": part_id,
                "face_labels": {str(k): v for k, v in feat_result["face_labels"].items()},
                "feature_params": {
                    str(k): v for k, v in feature_params.items()
                },
                "summary": feat_result["summary"],
                "view_config": view_config,
            },
            features_json,
        )
        self._save_json(stats, graph_json)

        elapsed = (datetime.now() - start_time).total_seconds()
        result = {
            "part_id": part_id,
            "dxf_path": dxf_path,
            "png_path": png_path,
            "features_json": features_json,
            "graph_json": graph_json,
            "num_faces": len(model.faces),
            "num_dimensions": len(selected_dims),
            "elapsed_seconds": round(elapsed, 2),
        }
        self._log.info("=" * 60)
        self._log.info("Pipeline DONE  |  %.2f s", elapsed)
        self._log.info("  DXF  → %s", dxf_path)
        self._log.info("  PNG  → %s", png_path)
        self._log.info("=" * 60)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_part_id(step_file_path: str) -> str:
        """Derive a safe part identifier from the file path."""
        if step_file_path:
            stem = os.path.splitext(os.path.basename(step_file_path))[0]
            return "".join(c if c.isalnum() else "_" for c in stem)[:40]
        return "demo_part"

    def _save_json(self, data: Dict, path: str) -> None:
        """Write *data* to *path* as JSON."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            self._log.debug("JSON saved → '%s'", path)
        except Exception as exc:
            self._log.warning("Could not save JSON to '%s': %s", path, exc)
