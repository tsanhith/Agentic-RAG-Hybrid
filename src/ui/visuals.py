import streamlit as st
import pandas as pd
import altair as alt

def render_sidebar_stats(chunk_count):
    st.sidebar.markdown(f"### 📊 Chunks: {chunk_count}")
    st.sidebar.progress(min(chunk_count / 100, 1.0))

def render_source_badges(results):
    if not results: return
    st.markdown("### 📚 Sources Used")
    cols = st.columns(len(results))
    for i, (doc, score) in enumerate(results):
        confidence = (1.0 / (1.0 + score)) * 100
        with cols[i]:
            st.metric(label=f"Page {doc.metadata.get('page','?')}", value=f"{confidence:.0f}%", delta="Match")

def render_comparison_chart(doc_content, distance_score, doc_vector, query_vector, label):
    similarity_score = 1.0 / (1.0 + distance_score)
    percentage = similarity_score * 100
    
    if percentage >= 70: color, label_text = "green", "Strong Match"
    elif percentage >= 50: color, label_text = "orange", "Moderate Match"
    else: color, label_text = "red", "Weak Match"

    with st.expander(f"📊 Match Analysis: {label} ({percentage:.1f}%)", expanded=False):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f":{color}[**{percentage:.1f}%**] ({label_text})")
            st.caption(f"(Raw Distance: {distance_score:.4f})")
        with c2:
            st.markdown("**📖 Full Content:**")
            st.info(doc_content[:2000]) # Shows 2000 chars

        # Chart
        dims = min(len(query_vector), 50)
        df = pd.DataFrame({
            'Dimension': list(range(dims)) * 2,
            'Value': query_vector[:dims] + doc_vector[:dims],
            'Type': ['Your Question'] * dims + ['The Document'] * dims
        })
        chart = alt.Chart(df).mark_line().encode(
            x='Dimension', y='Value', color='Type'
        ).properties(height=200).interactive()
        st.altair_chart(chart, use_container_width=True)