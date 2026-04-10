import streamlit as st
import os
import tempfile

from src.ui.layout import setup_page
from src.ui.visuals import render_sidebar_stats, render_comparison_chart, render_source_badges
from src.core.memory import MemoryManager
from src.core.processing import DocumentProcessor
from src.core.agent import AgentBrain

# 1. Setup
uploaded_files, groq_api_key, tavily_api_key, retrieval_k, chunk_size = setup_page()

if not groq_api_key:
    st.markdown(
        """
        <section class="hero-shell">
            <p class="hero-kicker">Agentic RAG Workspace</p>
            <h1 class="hero-title">Query your docs with a clean hybrid assistant</h1>
            <p class="hero-subtitle">Add your Groq key from the sidebar to start chatting with document + web intelligence.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.info("Please enter your Groq API key in the sidebar to begin.")
    st.stop()

# 2. State Init
if "memory_manager" not in st.session_state: st.session_state.memory_manager = MemoryManager()
if "messages" not in st.session_state: st.session_state.messages = []
if "processed_state" not in st.session_state: st.session_state.processed_state = None

# --- HOT RELOAD FIX ---
# Check if Agent exists. If NOT, create it.
# If it DOES exist, check if the API keys have changed. If so, recreate it.
force_reinit = False
if "agent" in st.session_state:
    # Check if the existing agent has the current Tavily Key
    current_agent_has_tavily = st.session_state.agent.tavily is not None
    user_provided_tavily = bool(tavily_api_key)
    
    # If user provided a key but agent doesn't have it -> REINIT
    if user_provided_tavily and not current_agent_has_tavily:
        force_reinit = True

if "agent" not in st.session_state or force_reinit:
    if force_reinit: st.toast("🔄 Updating Agent with new Keys...", icon="🔑")
    st.session_state.agent = AgentBrain(groq_api_key, tavily_api_key, st.session_state.memory_manager)

# 3. Smart Ingestion Logic
if uploaded_files:
    current_state = {"files": frozenset({f.name for f in uploaded_files}), "chunk_size": chunk_size}
    
    if st.session_state.processed_state != current_state:
        st.sidebar.warning("⚠️ Pending Changes")
        # WARNING FIX: Removed 'use_container_width'
        if st.sidebar.button("⚡ Process Files", type="primary"):
            processor = DocumentProcessor(chunk_size=chunk_size)
            with st.status("🏗️ Building Knowledge Base...", expanded=True) as status:
                splits = []
                for f in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.getbuffer())
                        tmp_path = tmp.name
                    try: splits.extend(processor.process_files([tmp_path]))
                    finally: os.remove(tmp_path)
                
                st.session_state.memory_manager.clear()
                st.session_state.memory_manager.ingest_docs(splits, status_container=status)
                st.session_state.processed_state = current_state
                status.update(label="✅ Ready!", state="complete", expanded=False)
                st.rerun()
    else:
        st.sidebar.success("✅ System Ready")

# 4. UI Stats
if st.session_state.memory_manager.vector_store:
    render_sidebar_stats(st.session_state.memory_manager.vector_store.index.ntotal)

# 5. Chat Loop
st.markdown(
    """
    <section class="hero-shell">
        <p class="hero-kicker">Agentic RAG Workspace</p>
        <h1 class="hero-title">Agentic Brain</h1>
        <p class="hero-subtitle">Hybrid retrieval across your PDFs and live web search when needed.</p>
    </section>
    """,
    unsafe_allow_html=True,
)
st.caption("Model: Llama-3.3-70B via Groq")

if not st.session_state.messages:
    st.markdown("**Try prompts like:**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.code("Summarize the main argument from my uploaded documents.")
    with c2:
        st.code("What are the latest updates on this topic?")
    with c3:
        st.code("Compare source evidence and highlight conflicts.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🧠"):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"): st.markdown(prompt)

    with st.chat_message("assistant", avatar="🧠"):
        if not st.session_state.memory_manager.vector_store and not tavily_api_key:
            st.warning("No documents uploaded and no Tavily key provided.")
            response = "I do not have knowledge access yet. Upload PDFs or add Tavily for web search."
            tool = "CHAT"
            results = []
        else:
            with st.status("🤔 Thinking...", expanded=True) as status:
                response, results, tool = st.session_state.agent.ask(
                    prompt,
                    chat_history=st.session_state.messages,
                    k=retrieval_k,
                    status_container=status,
                )
                labels = {
                    "RAG": "Documents",
                    "WEB": "Internet",
                    "CHAT": "Logic",
                    "MIXED": "Mixed",
                }
                status.update(label=f"Used Tool: {labels.get(tool, '🔧 Other')}", state="complete", expanded=False)

            st.markdown(response)

        labels = {
            "RAG": "Routed via Documents",
            "WEB": "Routed via Internet",
            "CHAT": "Routed via Logic",
            "MIXED": "Routed via Mixed Mode",
        }
        st.markdown(f"<div class='tool-pill'>{labels.get(tool, 'Routed via Other')}</div>", unsafe_allow_html=True)

        if tool in {"RAG", "MIXED"} and results:
            render_source_badges(results)
            for i, (doc, score) in enumerate(results):
                # Quick embedding for graph
                model = st.session_state.memory_manager.get_embedding_model()
                render_comparison_chart(
                    doc.page_content,
                    score,
                    model.embed_query(doc.page_content),
                    model.embed_query(prompt),
                    f"Source {i+1}",
                )

    st.session_state.messages.append({"role": "assistant", "content": response})
