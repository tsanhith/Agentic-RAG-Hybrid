import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# Import our Modular Backend & Frontend
from src.ui.layout import setup_page
from src.ui.visuals import render_sidebar_stats, render_comparison_chart
from src.core.memory import MemoryManager
from src.core.processing import DocumentProcessor
from src.core.agent import AgentBrain

# 1. SETUP
load_dotenv()
setup_page()

# 2. SIDEBAR: Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # A. API Keys
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        groq_api_key = st.text_input("Groq API Key (Required)", type="password")

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    # If env var is missing, let user input it
    if not tavily_api_key:
        tavily_api_key = st.text_input(
            "Tavily API Key (Optional)", 
            type="password", 
            help="Paste a key to unlock Web Search. Leave empty for Document-Only mode."
        )
        
    if not groq_api_key:
        st.warning("⚠️ Groq Key required to continue.")
        st.stop()

    st.divider()

    # B. Tuning Knobs
    st.subheader("🔧 Tuning")
    chunk_size = st.slider("Chunk Size", 200, 2000, 700, 100)
    
    # C. Initialize Backend (With Hot-Swap Support)
    if "memory" not in st.session_state:
        st.session_state.memory = MemoryManager()
        st.session_state.processor = DocumentProcessor(chunk_size=chunk_size)
        st.session_state.chunk_count = 0
        st.session_state.active_tavily_key = None # Track which key is currently loaded
        
        # Initial Agent Creation
        st.session_state.agent = AgentBrain(groq_api_key, tavily_api_key, st.session_state.memory)
        st.session_state.active_tavily_key = tavily_api_key

    # --- THE FIX: DETECT KEY CHANGE ---
    # If the user pasted a new key, but the Agent is still using the old one (or None), UPDATE IT.
    current_key_input = tavily_api_key if tavily_api_key else None
    
    if current_key_input != st.session_state.active_tavily_key:
        with st.spinner("🔄 Updating Agent Capabilities..."):
            # Re-initialize the Agent with the new Key
            st.session_state.agent = AgentBrain(groq_api_key, current_key_input, st.session_state.memory)
            st.session_state.active_tavily_key = current_key_input
            st.success("✅ Agent upgraded with Web Search!")
            # Rerun to apply changes immediately
            st.rerun()
    # ----------------------------------

    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0

    render_sidebar_stats(st.session_state.chunk_count)

    # E. Document Ingestion
    st.divider()
    st.subheader("📂 Knowledge Base")
    uploaded_files = st.file_uploader("Upload PDFs or Text", accept_multiple_files=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Ingest"):
            if uploaded_files:
                with st.spinner("Processing..."):
                    temp_paths = []
                    for f in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{f.name.split('.')[-1]}") as tmp:
                            tmp.write(f.getvalue())
                            temp_paths.append(tmp.name)
                    
                    st.session_state.processor.splitter._chunk_size = chunk_size
                    chunks = st.session_state.processor.process_files(temp_paths)
                    st.session_state.memory.add_documents(chunks)
                    st.session_state.chunk_count += len(chunks)
                    
                    for path in temp_paths: os.remove(path)
                    st.success(f"✅ Added {len(chunks)} chunks!")
            else:
                st.warning("Upload a file first.")

    with col2:
        if st.button("Clear"):
            st.session_state.memory.clear()
            st.session_state.chunk_count = 0
            st.rerun()

# 3. CHAT INTERFACE
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            
            # 1. Ask Agent
            response, results, tool_used = st.session_state.agent.ask(
                prompt, 
                chat_history=st.session_state.messages
            )
            
            # 2. Show Badge
            if tool_used == "RAG": st.caption("🧠 Source: **Internal Docs**")
            elif tool_used == "WEB": st.caption(f"🌍 Source: **Web Search** (Tavily)")
            elif tool_used == "CHAT": st.caption("💬 Source: **Chat**")

            st.markdown(response)
            
            # 3. Visuals
            if results:
                st.divider()
                st.subheader("🔍 Reasoning Trace")
                if tool_used == "RAG":
                    query_vector = st.session_state.memory.get_embedding_model().embed_query(prompt)
                    for doc, score in results:
                        doc_vector = st.session_state.memory.get_embedding_model().embed_query(doc.page_content)
                        page_num = doc.metadata.get('page', 'Unknown')
                        if isinstance(page_num, int): page_num += 1
                        render_comparison_chart(doc.page_content, score, doc_vector, query_vector, f"Page {page_num}")
                elif tool_used == "WEB":
                    for doc in results:
                        st.markdown(f"#### 🔗 Source: **{doc.metadata.get('source', 'Web')}**")
                        st.info(doc.page_content)

            st.session_state.messages.append({"role": "assistant", "content": response})