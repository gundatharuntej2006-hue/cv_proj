# P1 Clinician Triage Dashboard (Streamlit)
import sys
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import streamlit as st
    from tbcore.enums import TriageCategory, QualityStatus
    from tbcore.schemas import InputCXR
    from p4_pipeline.triage_orchestrator import TriagePipelineOrchestrator
    from p3_ops.pdf_generator import TriagePDFReportGenerator

    st.set_page_config(
        page_title="AI TB Triage CAD - BMSIT&M",
        page_icon="🫁",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🫁 Explainable TB Screening & Triage System")
    st.caption("Autonomous Dept. of AI & ML, BMSIT&M — Mini Project Phase-0 & 1 (BAI506)")
    st.markdown("Compliant with **WHO Target Product Profile** (Sensitivity >= 90%, Specificity >= 70%, Conformal Miss Rate <= 10%)")

    st.sidebar.header("Triage Settings")
    alpha_target = st.sidebar.slider("Conformal Miss Rate Target (α)", min_value=0.01, max_value=0.20, value=0.10, step=0.01)
    tau_low = st.sidebar.slider("Tau Low (NOT TB threshold)", min_value=0.05, max_value=0.40, value=0.15, step=0.01)
    tau_high = st.sidebar.slider("Tau High (TB threshold)", min_value=0.45, max_value=0.90, value=0.65, step=0.01)

    uploaded_file = st.sidebar.file_uploader("Upload Chest Radiograph (PNG / JPEG / DICOM)", type=["png", "jpg", "jpeg", "dcm"])

    col1, col2, col3 = st.columns([1, 1, 1])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        
        with col1:
            st.subheader("1. Input Radiograph")
            st.image(img, use_column_width=True, caption=f"Uploaded: {uploaded_file.name}")

        orch = TriagePipelineOrchestrator()
        orch.triage_engine.alpha_target = alpha_target
        orch.triage_engine.tau_low = tau_low
        orch.triage_engine.tau_high = tau_high

        meta = InputCXR(
            image_id=uploaded_file.name,
            file_path=uploaded_file.name,
            width=img.width,
            height=img.height
        )

        with st.spinner("Processing through Gate -> Crop -> RAD-DINO -> D-FINE -> Conformal Triage..."):
            result = orch.process(arr, meta)

        with col2:
            st.subheader("2. Diagnostic Localization")
            st.image(img, use_column_width=True, caption="Lesion Localization Overlay")
            if result.detection and result.detection.boxes:
                st.write(f"Detected {result.detection.total_lesions_found} lesion(s)")
                for b in result.detection.boxes:
                    st.info(f"Type: {b.lesion_type} | Conf: {b.confidence:.2f}")

        with col3:
            st.subheader("3. Conformal Triage Verdict")
            triage = result.triage
            if triage:
                if triage.triage_category == TriageCategory.TB:
                    st.error("🚨 TRIAGE VERDICT: TB (HIGH RISK)")
                elif triage.triage_category == TriageCategory.REFER:
                    st.warning("⚠️ TRIAGE VERDICT: REFER (INDETERMINATE)")
                else:
                    st.success("✅ TRIAGE VERDICT: NOT TB (SCREENED OUT)")

                st.metric("Calibrated TB Probability", f"{triage.calibrated_tb_prob * 100:.1f}%")
                st.write(f"**Recommendation:** {triage.clinical_recommendation}")
                st.write(f"**Latency:** {result.latency_ms:.1f} ms (< 2000 ms target)")

                pdf_gen = TriagePDFReportGenerator()
                pdf_path = pdf_gen.generate_report(result, f"report/{uploaded_file.name}_triage.pdf")
                with open(pdf_path, "rb") as f:
                    st.download_button("📄 Download Clinical PDF Report", f, file_name=f"{uploaded_file.name}_triage.pdf")
    else:
        st.info("👈 Please upload a radiograph from the sidebar to evaluate triage.")

except ImportError as e:
    print(f"Streamlit or submodules not loaded: {e}")
