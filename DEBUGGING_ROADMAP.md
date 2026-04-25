# CAD Automation Pipeline: Diagnostic Report & Fix Roadmap

## Executive Summary
The pipeline suffered from **data disconnection** between three major layers: the CAD Parser (Phase 1), the Knowledge Graph (Phase 2), and the GNN Inference (Phase 4). While individual modules ran without crashing, the flow of "Adjacency" (how faces connect) was broken at every handoff, rendering the GNN's topological reasoning ineffective.

**Status: All critical fixes applied.** Remaining action: retrain the GNN model with 13-dim features.

---

## ✅ Fixes Applied

### Stage 1: Parser Connectivity — `cad/parser.py` ✅
- **Root cause**: `_extract_topology` never populated `face_ids` for edges.
- **Fix**: Added `topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, ...)` to resolve each edge's parent faces to sequential 0-indexed IDs.
- **Also fixed**: Completed `_stub_topology()` (was empty), removed unused `yaml` import.

### Stage 2: KG Adjacency — `kg/ingest.py` ✅
- **Root cause**: `build_adjacency_from_occ` used `hash(str(face))` producing random IDs that didn't match Face nodes.
- **Fix**: Deleted `build_adjacency_from_occ` entirely. `_build_adjacency` now reads `face_ids` directly from the topology dict (populated by Stage 1). Also cleaned up 6 duplicate/conflicting imports.

### Stage 3: Feature Documentation — `graph/extract.py`, `graph/build_dgl.py` ✅
- **Issue**: Stale docstrings said "10 dims" but code produces 13.
- **Fix**: Updated all documentation to describe the actual 13-feature layout.

### Stage 4: Model & Inference — `models/hgt.py`, `inference/predict.py`, `training/train.py` ✅
- **Issues**: 
  - `hgt.py` defaulted to `in_feats=5`, `hidden_dim=640` (config says 13/768)
  - `predict.py` hardcoded `runtime_in_feats = 13`
  - `train.py` had a duplicate `build_model()` with divergent defaults
- **Fixes**:
  - Aligned `hgt.py` defaults to `in_feats=13`, `hidden_dim=768`
  - `predict.py` now reads from `cfg["model"]["in_feats"]`
  - `train.py` uses single `build_model` from `models.graphsage`
  - Added loud retraining warning when resize hack activates

### Stage 5: Projection — `drawing/projection.py` ✅
- **Root cause**: Two competing `project_view` functions; the active one returned a dict (not ProjectionResult); bounding box was 6 values (3D) instead of 4 (2D).
- **Fix**: Complete rewrite — single `project_view` always returns `ProjectionResult` with 2D bbox computed from projected polylines.

### Stage 6: Orchestration — `pipelines/run_all.py` ✅
- **Issue**: `output_path=None` when `--output` omitted → `Path(None)` crash.
- **Fix**: Auto-generates `output/{stem}_{timestamp}.dxf` when no output specified. Removed dead dict→dataclass safety net.

### Stage 7: Tests ✅
- Fixed `test_kg_ingest_full_793.py`: updated imports, fixed `ingest_step` signature
- Fixed `test_drawing_projection_793.py`: fixed `load_step` args, dict access, OmegaConf syntax
- All files pass `python -m py_compile`

---

## ⚠️ Remaining: Model Retraining

The checkpoint `best_hgt.pt` was trained with `in_feats=5`. The resize hack zero-pads features 5–12, meaning they have **no learned weights**.

```bash
python training/train.py --config configs/default.yaml --model_type hgt
```

After retraining, the resize hack will no longer activate and all 13 features will contribute to inference.
