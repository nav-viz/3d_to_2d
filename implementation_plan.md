# Streamlit CAD Automation Dashboard

Create a modern, web-based interface to interact with the CAD-to-DXF pipeline. This dashboard will allow users to upload STEP files, trigger the AI feature recognition and drawing generation, and download the resulting engineering documents.

## User Review Required

> [!IMPORTANT]
> **Streamlit Installation**: Streamlit is not currently installed in the `cad_auto` environment. You will need to run `pip install streamlit` once the code is generated.

## Proposed Changes

### Dashboard Application
#### [NEW] [dashboard.py](file:///c:/Users/navee/Downloads/08_04_26_v1-20260424T105823Z-3-001/08_04_26_v1/dashboard.py)
- **UI Design**: 
    - Dark mode theme with custom CSS for a premium "Antigravity" look.
    - Sidebar for configuration (Model selection, Confidence threshold).
    - Main stage: File uploader, Progress bars, and Results gallery.
- **Backend Logic**:
    - Integration with `pipelines.run_all` modules.
    - Temporary file management for uploaded STEP files.
    - Result extraction: Capturing PNG and DXF paths for display and download.

### Pipeline Refactoring (Minor)
#### [MODIFY] [run_all.py](file:///c:/Users/navee/Downloads/08_04_26_v1-20260424T105823Z-3-001/08_04_26_v1/pipelines/run_all.py)
- Ensure functions are cleanly exportable for use in the dashboard without triggering CLI logic.

## Verification Plan

### Manual Verification
1.  Run the dashboard locally: `streamlit run dashboard.py`.
2.  Upload the test file `nist_ctc_01_asme1_rd.stp`.
3.  Observe progress logs in the UI.
4.  Verify the PNG drawing is displayed.
5.  Download the DXF and verify it contains the technical drawing.

### Automated Tests
- None (Visual UI verification required).
