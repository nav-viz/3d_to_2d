import streamlit as st
import os
import yaml
import sys
import time
from pathlib import Path
import tempfile
import json

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loguru import logger
from pipelines.run_all import stage_ingest, stage_infer, stage_exportdxf, get_timestamp
from kg.neo4j_client import Neo4jClient

# --- Configuration ---
DEFAULT_CONFIG = "configs/default.yaml"
CHECKPOINT_PATH = "checkpoints/best_hgt_retrained.pt"

# --- Page Setup ---
st.set_page_config(
    page_title="Antigravity CAD Automation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("🛠 Settings")
config_file = st.sidebar.text_input("Config Path", DEFAULT_CONFIG)
model_type = st.sidebar.selectbox("Model Type", ["hgt", "graphsage"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.75)

st.sidebar.markdown("---")
st.sidebar.info("Antigravity CAD v0.1\n\nAutomated Technical Drawing Generation via KG + GNN")

# --- App Logic ---
def load_config(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)

def _prediction_key(pred, idx):
    """Build a stable per-face key across possible payload shapes."""
    for key in ("face_id", "face_idx", "face_index", "id", "node_id"):
        value = pred.get(key)
        if value is not None:
            return f"{key}:{value}"
    return f"idx:{idx}"

def summarize_predictions(preds, threshold):
    """
    Return (faces_detected, machining_features, avg_confidence) from predictions.
    Deduplicates repeated per-face entries by keeping the highest-confidence one.
    """
    if not isinstance(preds, list) or not preds:
        return 0, 0, 0.0

    best_per_face = {}
    for idx, pred in enumerate(preds):
        if not isinstance(pred, dict):
            continue
        key = _prediction_key(pred, idx)
        conf = float(pred.get("confidence", 0.0) or 0.0)
        existing = best_per_face.get(key)
        if existing is None or conf > existing.get("_confidence", 0.0):
            best_per_face[key] = {"prediction": pred, "_confidence": conf}

    unique_preds = [v["prediction"] for v in best_per_face.values()]
    faces_detected = len(unique_preds)
    machining_features = sum(
        1 for p in unique_preds
        if p.get("feature_type") not in {"Stock", "Low_Confidence"}
        and float(p.get("confidence", 0.0) or 0.0) >= threshold
    )
    avg_confidence = (
        sum(float(p.get("confidence", 0.0) or 0.0) for p in unique_preds) / faces_detected
        if faces_detected else 0.0
    )
    return faces_detected, machining_features, avg_confidence

def run_pipeline(step_path, config_path):
    cfg = load_config(config_path)
    cfg["inference"]["confidence_threshold"] = confidence_threshold
    cfg["model"]["type"] = model_type
    
    client = Neo4jClient.from_config(cfg)
    
    try:
        with st.status("Processing CAD Model...", expanded=True) as status:
            st.write("Step 1: Ingesting to Knowledge Graph...")
            model_id = stage_ingest(Path(step_path), client, cfg)
            
            st.write("Step 2: AI Feature Recognition...")
            preds = stage_infer(
                step_path=Path(step_path),
                model_type=model_type,
                cfg=cfg,
                checkpoint=CHECKPOINT_PATH
            )
            
            # Write predictions back to KG
            from inference.writeback import write_predictions_to_kg
            write_predictions_to_kg(model_id, preds, client, model_type)
            
            st.write("Step 3: Exporting Technical Drawing...")
            # Generate a specific output path
            output_dir = Path(cfg["paths"].get("output_dxf", "output"))
            output_dir.mkdir(parents=True, exist_ok=True)
            output_filename = f"{Path(step_path).stem}_{get_timestamp()}.dxf"
            output_path = output_dir / output_filename
            
            stage_exportdxf(
                step_path=Path(step_path),
                predictions=preds,
                model_id=model_id,
                client=client,
                output_path=output_path,
                cfg=cfg
            )
            
            st.write("Step 4: Generating Preview...")
            png_path = output_path.with_suffix(".png")
            from dxf_to_png_working import dxf_to_png
            dxf_to_png(output_path, png_path)
            
            status.update(label="Processing Complete!", state="complete", expanded=False)
            
        return output_path, png_path, preds
        
    except Exception as e:
        st.error(f"Pipeline Failed: {e}")
        logger.exception(e)
        return None, None, None
    finally:
        client.close()

# --- Main UI ---
st.title("🤖 Antigravity CAD Automation")
st.markdown("Upload a 3D STEP file to generate ISO-compliant technical drawings automatically.")

uploaded_file = st.file_uploader("Choose a STEP file", type=['stp', 'step'])

if uploaded_file is not None:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stp") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    if st.button("🚀 Process Drawing"):
        out_path, png_path, preds = run_pipeline(tmp_path, config_file)
        
        if out_path and out_path.exists():
            st.success(f"Drawing generated successfully!")
            
            # Show Metrics
            col1, col2, col3 = st.columns(3)
            faces_detected, machining_features, avg_conf = summarize_predictions(
                preds, confidence_threshold
            )
            with col1:
                st.metric("Faces Detected", faces_detected)
            with col2:
                st.metric("Machining Features", machining_features)
            with col3:
                st.metric("Avg Confidence", f"{avg_conf:.2%}")

            # Show Preview
            if png_path and png_path.exists():
                st.markdown("### 🖼 Drawing Preview")
                st.image(str(png_path), use_container_width=True)

            # Download Section
            st.markdown("### 📥 Downloads")
            col_dxf, col_png = st.columns(2)
            with col_dxf:
                with open(out_path, "rb") as f:
                    st.download_button(
                        label="Download DXF",
                        data=f,
                        file_name=out_path.name,
                        mime="application/dxf"
                    )
            with col_png:
                if png_path and png_path.exists():
                    with open(png_path, "rb") as f:
                        st.download_button(
                            label="Download PNG",
                            data=f,
                            file_name=png_path.name,
                            mime="image/png"
                        )
            
            # Clean up
            os.remove(tmp_path)
        else:
            st.error("Failed to generate output.")

else:
    st.info("Please upload a STEP file to begin.")
    # Show example image if available
    example_img = Path("output/nist_ctc_01_asme1_rd.png")
    if example_img.exists():
        st.image(str(example_img), caption="Sample Output Rendering", use_container_width=True)
