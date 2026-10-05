# Streamlit Clinician Triage Dashboard
try:
    import streamlit as st
    st.set_page_config(page_title="Explainable TB Screening CAD", layout="wide")
    st.title("Explainable TB Screening & Triage")
    st.markdown("Automated triage compliant with WHO Target Product Profile.")
    st.info("System Ready. Upload radiograph to run triage.")
except ImportError:
    pass
