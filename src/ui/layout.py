import streamlit as st


def setup_page():
    """
    Configures the Streamlit page layout with the restored Sliders.
    """
    st.set_page_config(
        page_title="Agentic RAG", 
        page_icon="🧠", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # --- CUSTOM CSS ---
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        :root {
            --ui-bg-top: #eef6ff;
            --ui-bg-bottom: #f7fbff;
            --ui-accent: #0f766e;
            --ui-accent-strong: #0a5f57;
            --ui-panel: rgba(255, 255, 255, 0.86);
            --ui-border: rgba(15, 118, 110, 0.20);
            --ui-text: #0f172a;
            --ui-muted: #475569;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 15% 0%, rgba(38, 198, 218, 0.16), transparent 38%),
                radial-gradient(circle at 90% 5%, rgba(56, 189, 248, 0.12), transparent 40%),
                linear-gradient(180deg, var(--ui-bg-top), var(--ui-bg-bottom));
            color: var(--ui-text);
        }

        [data-testid="stMain"] {
            background: transparent;
        }

        [data-testid="stMain"] p,
        [data-testid="stMain"] li,
        [data-testid="stMain"] label,
        [data-testid="stMain"] span,
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] * {
            color: var(--ui-text) !important;
        }

        [data-testid="stMain"] h1,
        [data-testid="stMain"] h2,
        [data-testid="stMain"] h3,
        [data-testid="stMain"] h4 {
            color: #0b1220 !important;
        }

        [data-testid="stAppViewContainer"] * {
            font-family: "IBM Plex Sans", "Segoe UI", Tahoma, sans-serif;
        }

        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
            max-width: 1180px;
            animation: fadeUp 0.45s ease-out;
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        h1, h2, h3 {
            font-family: "Space Grotesk", "Trebuchet MS", sans-serif;
            letter-spacing: -0.02em;
            color: #0b1220;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(175deg, #0f172a, #111827 40%, #1f2937 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: #e2e8f0;
        }

        [data-testid="stSidebar"] .stTextInput input {
            background: rgba(255, 255, 255, 0.09);
            border: 1px solid rgba(148, 163, 184, 0.40);
        }

        [data-testid="stSidebar"] .stFileUploader > div {
            background: rgba(255, 255, 255, 0.06);
            border: 1px dashed rgba(148, 163, 184, 0.55);
            border-radius: 12px;
        }

        [data-testid="stSidebar"] .stButton > button {
            border-radius: 10px;
            border: 1px solid rgba(125, 211, 252, 0.35);
            background: linear-gradient(120deg, #0f766e, #0e7490);
            color: #ecfeff;
            font-weight: 600;
        }

        [data-testid="stSidebar"] .stDownloadButton > button {
            border-radius: 10px;
            border: 1px solid rgba(125, 211, 252, 0.35);
            background: linear-gradient(120deg, #0f766e, #0e7490);
            color: #ecfeff;
            font-weight: 600;
        }

        [data-testid="stMain"] .stButton > button,
        [data-testid="stMain"] .stDownloadButton > button {
            border-radius: 10px;
            border: 1px solid #cbd5e1;
            background: #ffffff;
            color: #0f172a !important;
            font-weight: 600;
        }

        [data-testid="stMain"] .stButton > button:hover,
        [data-testid="stMain"] .stDownloadButton > button:hover {
            border-color: #0e7490;
            color: #0e7490 !important;
        }

        .hero-shell {
            background: linear-gradient(135deg, rgba(14, 116, 144, 0.10), rgba(15, 118, 110, 0.14));
            border: 1px solid var(--ui-border);
            border-radius: 18px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 22px rgba(2, 132, 199, 0.12);
        }

        .hero-kicker {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.72rem;
            font-weight: 700;
            color: var(--ui-accent);
            margin-bottom: 0.1rem;
        }

        .hero-title {
            font-family: "Space Grotesk", "Trebuchet MS", sans-serif;
            font-size: 1.6rem;
            font-weight: 700;
            color: #111827;
            margin: 0;
        }

        .hero-subtitle {
            margin: 0.35rem 0 0;
            color: var(--ui-muted);
            font-size: 0.95rem;
        }

        [data-testid="stChatMessage"] {
            border-radius: 16px;
            padding: 0.65rem 0.9rem;
            margin-bottom: 0.85rem;
            border: 1px solid #dbe2ea;
            background: #ffffff;
            box-shadow: 0 6px 14px rgba(15, 23, 42, 0.06);
        }

        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li,
        [data-testid="stChatMessage"] span,
        [data-testid="stChatMessage"] strong {
            color: #0f172a !important;
        }

        [data-testid="stChatInput"] textarea {
            background: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 10px !important;
        }

        [data-testid="stTextArea"] textarea {
            background: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 10px !important;
        }

        .tool-pill {
            display: inline-block;
            margin-top: 0.45rem;
            padding: 0.28rem 0.56rem;
            border-radius: 999px;
            border: 1px solid rgba(15, 118, 110, 0.35);
            color: var(--ui-accent-strong);
            font-size: 0.78rem;
            font-weight: 600;
            background: rgba(15, 118, 110, 0.10);
        }

        .insight-card {
            border: 1px solid rgba(14, 116, 144, 0.24);
            border-radius: 14px;
            background: linear-gradient(140deg, #ffffff, #f8fdff);
            box-shadow: 0 6px 16px rgba(14, 116, 144, 0.10);
            padding: 0.72rem 0.88rem;
            margin-top: 0.65rem;
        }

        .insight-title {
            font-family: "Space Grotesk", "Trebuchet MS", sans-serif;
            font-size: 0.92rem;
            font-weight: 700;
            color: #0e7490;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }

        .soft-divider {
            height: 1px;
            border: 0;
            background: linear-gradient(90deg, transparent, rgba(100, 116, 139, 0.34), transparent);
            margin: 0.6rem 0 1rem;
        }

        footer { visibility: hidden; }
        #MainMenu { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## Agent Control")
        st.caption("Configure keys, load documents, and tune retrieval.")
        
        # Configuration
        with st.expander("Credentials", expanded=True):
            if "groq_key" not in st.session_state:
                st.session_state.groq_key = ""
            if "tavily_key" not in st.session_state:
                st.session_state.tavily_key = ""

            st.caption("Get free keys from [Groq](https://console.groq.com/keys) and [Tavily](https://app.tavily.com/home).")
            
            groq_api_key = st.text_input(
                "Groq API Key",
                type="password",
                key="groq_key_input",
                value=st.session_state.groq_key,
                placeholder="gsk_...",
            )
            tavily_api_key = st.text_input(
                "Tavily API Key (optional)",
                type="password",
                key="tavily_key_input",
                value=st.session_state.tavily_key,
                placeholder="tvly-...",
            )
            
            st.session_state.groq_key = groq_api_key
            st.session_state.tavily_key = tavily_api_key

        st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
        
        # Knowledge Base
        st.markdown("### Data Source")
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Use one or multiple PDFs to build your local knowledge base.",
        )

        # --- RESTORED SLIDERS ---
        st.markdown("### Brain Tuning")
        
        # Slider 1: Retrieval Depth (Top K)
        retrieval_k = st.slider(
            "Retrieval Depth (Chunks)",
            min_value=1,
            max_value=10,
            value=5,
            help="Higher depth improves recall but can add latency.",
        )
        
        # Slider 2: Chunk Size
        chunk_size = st.slider(
            "Chunk Size (Characters)",
            min_value=200,
            max_value=2000,
            value=1000,
            step=100,
            help="Smaller chunks are precise; larger chunks preserve context.",
        )

        # Memory Management
        st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
        if st.button("Reset Brain", type="secondary", use_container_width=True):
            if "memory_manager" in st.session_state:
                st.session_state.memory_manager.clear()
            st.session_state.messages = []
            if "processed_state" in st.session_state:
                del st.session_state["processed_state"]
            st.toast("Brain memory wiped.")
            st.rerun()

    return uploaded_files, groq_api_key, tavily_api_key, retrieval_k, chunk_size
