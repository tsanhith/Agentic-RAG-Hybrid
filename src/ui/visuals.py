import streamlit as st
import pandas as pd
import numpy as np

def render_header():
    """Renders the main title and a 'How it Works' guide."""
    st.title("🧠 Memory-Aware Agent")
    
    # Simple "How it Works" Expander
    with st.expander("ℹ️ How does this AI think? (Click to learn)"):
        st.markdown("""
        1. **Reading:** The AI reads your uploaded files and breaks them into small "chunks" (paragraphs).
        2. **Memorizing:** It converts every chunk into a list of **384 numbers** (a "Vector Fingerprint").
        3. **Retrieving:** When you ask a question, the AI converts your question into numbers and finds the chunk that mathematically matches best.
        4. **Answering:** It reads that specific chunk and uses it to answer you.
        """)

def render_sidebar_stats(chunk_count):
    """Shows the 'Memory Size' stats with plain English tooltips."""
    st.sidebar.markdown("### 📊 Memory Stats")
    col1, col2 = st.sidebar.columns(2)
    
    col1.metric(
        "Knowledge Chunks", 
        chunk_count,
        help="The number of small text pieces (paragraphs) currently stored in the AI's brain."
    )
    
    col2.metric(
        "Brain 'Resolution'", 
        "384 Dims",
        help="This is how detailed the AI's understanding is. It looks at 384 distinct features (topic, tone, keywords, context) for every sentence."
    )
    
    st.sidebar.caption(f"Total Knowledge: {chunk_count} paragraphs loaded.")

def render_comparison_chart(chunk_text, chunk_score, doc_vector, query_vector, source_info):
    """
    Visualizes the 'Match' with simplified explanations.
    """
    # Color Coding for the Score
    score_color = "green" if chunk_score > 0.4 else "orange" if chunk_score > 0.2 else "red"
    
    # Simple Explanation of the Score
    score_meaning = ""
    if chunk_score > 0.5: score_meaning = "(Strong Match)"
    elif chunk_score > 0.3: score_meaning = "(Decent Match)"
    else: score_meaning = "(Weak Match - Might be irrelevant)"

    st.markdown(f"#### 🔗 Match Confidence: :{score_color}[**{chunk_score:.2f}**] {score_meaning}")
    st.caption(f"Source: **{source_info}**")
    
    # Vector X-Ray with Guide
    with st.expander("🧬 X-Ray Vision: See the Match", expanded=True):
        st.markdown("""
        **How to read this graph:**
        * **🟦 Blue Line:** Your Question's "Thought Pattern".
        * **🟥 Red Line:** The Document's "Thought Pattern".
        * **✅ Good Match:** When the Blue and Red lines **move together** (overlap).
        * **❌ Bad Match:** When the lines move in opposite directions.
        """)
        
        # Data Setup
        chart_data = pd.DataFrame({
            "The Document (Red)": doc_vector[:50],
            "Your Question (Blue)": query_vector[:50]
        })
        
        # Render Chart
        st.line_chart(
            chart_data, 
            color=["#FF4B4B", "#1C83E1"]
        )
        
    # Matched Text Display
    st.info(f"**📖 Content Found in {source_info}:**\n\n{chunk_text}")