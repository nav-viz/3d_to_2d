# Hybrid AI CAD Drawing Generator

A production-grade system that automatically generates 2-D engineering drawings
(DXF + PNG) from 3-D CAD models (STEP files) using a hybrid AI pipeline
combining Graph Neural Networks, feature regression, and ISO-compliant
dimension placement.

## Features

- **STEP File Parsing** – pythonOCC integration with synthetic fallback
- **BRep Graph** – face-adjacency graph for topology analysis
- **AI Feature Recognition** – GNN (GIN/GAT) with rule-based fallback
- **Feature Parameter Regression** – MLP or geometry-based estimation
- **View Prediction** – XGBoost or rule-based sheet & scale selection
- **Dimension Generation** – ISO 129 / ASME Y14.5 rules
- **Dimension Optimisation** – ILP (OR-Tools) or greedy conflict resolution
- **DXF Output** – ezdxf-based drawing with layers, title block, annotations
- **PNG Rendering** – matplotlib backend or pure-NumPy fallback
- **Neo4j Graph Storage** – in-memory fallback when server unavailable

## Quick Start

```bash
# Install core dependencies
pip install -r requirements.txt

# Run the demo pipeline (no STEP file needed)
python run_pipeline.py

# Or with a real STEP file
python run_pipeline.py
# > Enter STEP file path (or press Enter for demo): my_part.step

# Run as module
python -m src.main --step-file my_part.step
```

## Repository Structure

```
src/               – Pipeline source code
  config/          – Settings / configuration
  step_parser/     – STEP file parser & geometry extractor
  brep_graph/      – BRep graph builder & feature extractor
  database/        – Neo4j connection & graph storage
  ai_models/       – GNN, regression, view & ranking models
  dimensioning/    – Dimension generation & optimisation
  drawing/         – View projection, DXF writer, PNG renderer
  utils/           – Logger, geometry utils, graph utils
dataset/           – Synthetic CAD model generator
tests/             – pytest test suite
docker/            – Dockerfile
output/            – Generated DXF/PNG/JSON artefacts
models/            – Trained model weight files
```

## Running Tests

```bash
pytest tests/ -v --tb=short
```

## Architecture

```
STEP File → StepParser → BRepGraph → Neo4j Storage
                    ↓
              GNNFeatureModel → FeatureRegressionModel
                    ↓
              ViewPredictionModel → DimensionGenerator
                    ↓
              DimensionRankingModel → DimensionOptimizer
                    ↓
              ViewProjector → DXFWriter → PNGRenderer
```

## Dependencies

**Required:** `numpy`, `ezdxf`, `pytest`, `flake8`

**Optional (enables AI/DB features):**
`torch`, `torch-geometric`, `neo4j`, `xgboost`, `ortools`, `opencv-python`, `pythonOCC-core`

All optional dependencies have graceful fallbacks so the system works end-to-end without them.