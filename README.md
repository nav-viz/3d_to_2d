# CAD Automation Pipeline

> **Hybrid KG + GNN → ISO-compliant DXF Technical Drawings**
> STEP File → Neo4j Knowledge Graph → DGL Heterograph → GNN (HGT / GraphSAGE) → View Selection → CSP Dimension Placement → DXF Export

---

## 📁 Project Structure

```
cad_automation/
├── cad/               STEP parsing (pythonocc-core) + rule-based MFR
├── kg/                Neo4j Knowledge Graph client + STEP ingestion
├── graph/             DGL heterograph construction, MFCAD++ H5 loader
├── models/            CADHGT + CADGraphSAGE GNN architectures
├── training/          Full training script (cosine LR, early stopping)
├── inference/         End-to-end STEP→prediction + KG write-back
├── reasoning/         Feature-confidence view selector (adaptive routing)
├── drawing/           HLR projection, CSP dimension placer, ezdxf writer
├── pipelines/         run_all.py – CLI orchestrator
├── configs/           default.yaml (all hyperparameters)
├── checkpoints/       Saved model weights
├── logs/              Training JSONL logs + pipeline log
├── data/mfcad_plus/hierarchical_graphs/   ← place H5 files here
└── tests/             pytest unit tests
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
# Create conda env (recommended)
conda create -n cad_auto python=3.10
conda activate cad_auto

# pythonocc-core (OCC) – conda is the easiest route
conda install -c conda-forge pythonocc-core=7.7.2

# Core packages
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

> **Note:** PyTorch + DGL are CPU-only by default. Edit `requirements.txt` to change the torch index URL for CUDA builds.

---

### 2. Place MFCAD++ H5 files
```
data/mfcad_plus/hierarchical_graphs/
├── training_MFCAD++.h5
├── val_MFCAD++.h5
└── test_MFCAD++.h5
```

---

### 3. Configure Neo4j
Edit `configs/default.yaml`:
```yaml
neo4j:
  uri: "bolt://localhost:7687"
  user: "neo4j"
  password: "your_password"
```
Or run a local Docker instance:
```bash
docker run --name neo4j-cad -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password neo4j:5
```

---

### 4. Train the GNN
```bash
# Train HGT (default)
python pipelines/run_all.py train --model_type hgt --epochs 100

# Train GraphSAGE
python pipelines/run_all.py train --model_type graphsage --epochs 100

# Quick test with 100 samples
python pipelines/run_all.py train --model_type hgt --max_samples 100
```

---

### 5. Ingest a STEP file
```bash
python pipelines/run_all.py ingest --step_file path/to/part.step
```

---

### 6. Run inference
```bash
python pipelines/run_all.py infer \
    --step_file path/to/part.step \
    --model_type hgt \
    --output_json predictions.json
```

---

### 7. Export DXF
```bash
python pipelines/run_all.py export \
    --step_file path/to/part.step \
    --model_type hgt \
    --output output/part.dxf
```

---

### 8. Run everything at once
```bash
python pipelines/run_all.py all \
    --step_file path/to/part.step \
    --model_type hgt \
    --output output/part.dxf
```

---

## 🧪 Run Tests
```bash
pytest tests/ -v
pytest tests/ --cov=graph --cov=drawing --cov=reasoning --cov-report=term-missing
```

---

## ⚙️ Configuration Reference

All settings are in `configs/default.yaml`. Key sections:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `model` | `type` | `hgt` | GNN model: `hgt` or `graphsage` |
| `model` | `hidden_dim` | `256` | GNN hidden dimension |
| `model` | `num_layers` | `3` | Message-passing layers |
| `training` | `epochs` | `100` | Max training epochs |
| `training` | `patience` | `15` | Early stopping patience |
| `drawing` | `sheet_size` | `A2` | ISO sheet size |
| `drawing` | `placement_strategy` | `csp` | `csp` or `greedy` |
| `inference` | `confidence_threshold` | `0.6` | Min confidence to accept prediction |

---

## 📐 Neo4j Knowledge Graph Schema

```
(:CADModel) -[:HAS_FACE]→     (:Face)
(:CADModel) -[:HAS_EDGE]→     (:Edge)
(:Face)     -[:ADJACENT_TO]→  (:Face)   {convexity}
(:Face)     -[:PREDICTED_AS]→ (:Feature) {confidence, model_type}
```

Indices are auto-created on first connection.

---

## 🧩 Architecture

```
STEP File
   │
   ▼
cad/parser.py          ← OCC STEPControl_Reader + ShapeFix
cad/brep_features.py   ← Rule-based MFR (Hole/Pocket/Boss/…)
   │
   ▼
kg/ingest.py           ← Upsert topology → Neo4j
kg/neo4j_client.py     ← Driver wrapper + schema
   │
   ▼
graph/build_dgl.py     ← Heterograph from Neo4j or H5
graph/mfcad_h5_dataset.py ← MFCAD++ H5 loader (E_1/E_2/E_3)
   │
   ▼
models/hgt.py          ← CADHGT (Heterogeneous Graph Transformer)
models/graphsage.py    ← CADGraphSAGE (HeteroGraphConv)
training/train.py      ← Full train loop
   │
   ▼
inference/predict.py   ← Softmax predictions per face
inference/writeback.py ← PREDICTED_AS → Neo4j
   │
   ▼
reasoning/view_selector.py ← Feature-confidence view planning
drawing/projection.py      ← OCC HLR / stub wireframe
drawing/dimension_placer.py← CSP + greedy placement
drawing/dxf_writer.py      ← ezdxf DXF export + GD&T
   │
   ▼
output/part.dxf
```

---

## 🔧 Extension Points

| Location | What to Extend |
|----------|----------------|
| `graph/extract.py` | Add V_2 (vertex) and V_3 (edge) features |
| `graph/mfcad_h5_dataset.py` | Support H_2 hierarchical level graphs |
| `models/hgt.py` | Add full ntype/etype metadata from dataset |
| `cad/brep_features.py` | Add new rule-based feature classes |
| `reasoning/view_selector.py` | Add more feature → view affinities |
| `drawing/dimension_placer.py` | Full ISO-1101 GD&T frame generation |
| `drawing/dxf_writer.py` | Surface finish symbols, welding annotations |

---

## 📦 Requirements Summary

- Python 3.10
- PyTorch 2.1.2 (CPU)
- DGL 1.1.2
- pythonocc-core 7.7.2 (OCC STEP parsing + HLR)
- Neo4j 5.x + neo4j Python driver 5.14.1
- ezdxf 1.1.4
- h5py 3.10+
- python-constraint 1.4.0
- loguru, rich, pyyaml, click

---

## 📄 License
MIT – M.Tech Research Project
