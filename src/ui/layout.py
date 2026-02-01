import streamlit as st

def setup_page():
    st.set_page_config(page_title="Memory Agent", layout="wide")
    st.title("🧠 Memory-Aware Agent")
    st.markdown("Live RAG Demo with **Vector Visualization** and **Confidence Scoring**.")