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
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .stChatMessage { border-radius: 15px; padding: 10px; margin-bottom: 10px; }
        [data-testid="stChatMessage"]:nth-child(odd) { background-color: rgba(33, 150, 243, 0.1); border: 1px solid rgba(33, 150, 243, 0.2); }
        [data-testid="stChatMessage"]:nth-child(even) { background-color: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); }
        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## 🧠 **Agent Control**")
        
        # Configuration
        with st.expander("🔐 Credentials", expanded=True):
            if "groq_key" not in st.session_state: st.session_state.groq_key = ""
            if "tavily_key" not in st.session_state: st.session_state.tavily_key = ""

            st.caption(
                "New here? Get free API keys: "
                "[Groq](https://console.groq.com/keys) · "
                "[Tavily](https://app.tavily.com/home)"
            )
            
            groq_api_key = st.text_input("Groq API Key", type="password", key="groq_key_input", value=st.session_state.groq_key)
            tavily_api_key = st.text_input("Tavily API Key", type="password", key="tavily_key_input", value=st.session_state.tavily_key)
            
            st.session_state.groq_key = groq_api_key
            st.session_state.tavily_key = tavily_api_key

        st.markdown("---")
        
        # Knowledge Base
        st.markdown("### 📂 Data Source")
        uploaded_files = st.file_uploader(
            "Upload Documents (PDF)", 
            type=["pdf"], 
            accept_multiple_files=True
        )

        # --- RESTORED SLIDERS ---
        st.markdown("### ⚙️ Brain Tuning")
        
        # Slider 1: Retrieval Depth (Top K)
        retrieval_k = st.slider(
            "Retrieval Depth (Chunks)", 
            min_value=1, 
            max_value=10, 
            value=5,
            help="How many document chunks to read."
        )
        
        # Slider 2: Chunk Size
        chunk_size = st.slider(
            "Chunk Size (Characters)", 
            min_value=200, 
            max_value=2000, 
            value=1000, 
            step=100,
            help="Small = Facts. Large = Context."
        )

        # Memory Management
        st.markdown("---")
        if st.button("🗑️ Reset Brain", type="secondary"):
            if "memory_manager" in st.session_state:
                st.session_state.memory_manager.clear()
            st.session_state.messages = []
            if "processed_state" in st.session_state:
                del st.session_state["processed_state"]
            st.toast("Brain memory wiped!", icon="🧹")
            st.rerun()

    return uploaded_files, groq_api_key, tavily_api_key, retrieval_k, chunk_size
