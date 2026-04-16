import streamlit as st
import pandas as pd
import altair as alt


def normalize_source_results(results):
    normalized = []
    for item in results or []:
        doc = None
        score = None

        if isinstance(item, tuple) and len(item) >= 2:
            doc, score = item[0], item[1]
        else:
            doc = item

        if doc is None or not hasattr(doc, "page_content"):
            continue

        if score is not None:
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = None

        normalized.append((doc, score))

    return normalized


def render_sidebar_stats(chunk_count):
    st.sidebar.markdown(f"### Indexed Chunks: {chunk_count}")
    st.sidebar.progress(min(chunk_count / 100, 1.0))


def render_source_badges(results):
    normalized = normalize_source_results(results)
    if not normalized:
        return
    st.markdown("### Sources Used")
    cols = st.columns(len(normalized))
    for i, (doc, score) in enumerate(normalized):
        source_name = doc.metadata.get("source") or doc.metadata.get("file_name") or "Document"
        with cols[i]:
            if score is None:
                st.metric(label=f"Source {i + 1}", value="Context", delta=f"Page {doc.metadata.get('page', '?')}")
            else:
                confidence = (1.0 / (1.0 + score)) * 100
                st.metric(
                    label=f"Source {i + 1}",
                    value=f"{confidence:.0f}% match",
                    delta=f"Page {doc.metadata.get('page', '?')}",
                )
            st.caption(source_name[:48])


def render_comparison_chart(doc_content, distance_score, doc_vector, query_vector, label):
    similarity_score = 1.0 / (1.0 + distance_score)
    percentage = similarity_score * 100

    if percentage >= 70:
        color, label_text = "green", "Strong Match"
    elif percentage >= 50:
        color, label_text = "orange", "Moderate Match"
    else:
        color, label_text = "red", "Weak Match"

    with st.expander(f"Match Analysis: {label} ({percentage:.1f}%)", expanded=False):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f":{color}[**{percentage:.1f}%**] ({label_text})")
            st.caption(f"(Raw Distance: {distance_score:.4f})")
        with c2:
            st.markdown("**Context Preview**")
            st.info(doc_content[:1700] + ("..." if len(doc_content) > 1700 else ""))

        # Chart
        dims = min(len(query_vector), 50)
        df = pd.DataFrame({
            "Dimension": list(range(dims)) * 2,
            "Value": query_vector[:dims] + doc_vector[:dims],
            "Type": ["Query"] * dims + ["Source"] * dims,
        })
        chart = (
            alt.Chart(df)
            .mark_line(strokeWidth=2.4)
            .encode(
                x=alt.X("Dimension:Q", axis=alt.Axis(title="Embedding Dimension")),
                y=alt.Y("Value:Q", axis=alt.Axis(title="Value")),
                color=alt.Color(
                    "Type:N",
                    scale=alt.Scale(domain=["Query", "Source"], range=["#0e7490", "#0f766e"]),
                ),
            )
            .properties(height=220)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
