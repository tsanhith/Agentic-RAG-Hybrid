import os
import tempfile
from datetime import datetime, timezone
from typing import List

import streamlit as st

from src.core.agent import AgentBrain
from src.core.memory import MemoryManager
from src.core.processing import DocumentProcessor
from src.ui.layout import setup_page
from src.ui.visuals import render_comparison_chart, render_sidebar_stats, render_source_badges


TOOL_LABELS = {
    "RAG": "Routed via Documents",
    "WEB": "Routed via Internet",
    "CHAT": "Routed via Logic",
    "MIXED": "Routed via Mixed Mode",
}


def build_chat_markdown(messages):
    if not messages:
        return "# Agentic Brain Chat Export\n\n_No messages yet._\n"

    lines = [
        "# Agentic Brain Chat Export",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC_",
        "",
    ]
    for msg in messages:
        role = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"## {role}")
        lines.append(msg.get("content", "").strip())
        tool = msg.get("tool")
        if tool:
            lines.append(f"_Tool: {TOOL_LABELS.get(tool, tool)}_")
        insight = msg.get("insight")
        if insight:
            lines.append("")
            lines.append("### Insight Card")
            lines.append(insight)
        lines.append("")
    return "\n".join(lines)


def _normalize_suggestions(raw_text: str) -> List[str]:
    suggestions = []
    seen = set()
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        while line and line[0] in "-*0123456789. )(":
            line = line[1:].strip()
        if not line:
            continue
        if not line.endswith("?"):
            line = f"{line.rstrip('.!')}?"
        if len(line) < 10:
            continue
        lowered = line.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        suggestions.append(line)
        if len(suggestions) == 3:
            break
    return suggestions


def _default_followups(tool: str) -> List[str]:
    if tool == "RAG":
        return [
            "Can you cite the strongest quote from the document?",
            "What key details might we have missed here?",
            "Can you turn this into an actionable checklist?",
        ]
    if tool == "WEB":
        return [
            "Can you verify this with multiple independent sources?",
            "What changed most recently about this topic?",
            "Can you summarize this into a 60-second brief?",
        ]
    return [
        "Can you explain this in simpler terms?",
        "What are the biggest trade-offs here?",
        "What should I do next step by step?",
    ]


def generate_followup_suggestions(question: str, answer: str, tool: str) -> List[str]:
    cache_key = f"{tool}|{question[:180]}|{answer[:260]}"
    cache = st.session_state.suggestion_cache
    if cache_key in cache:
        return cache[cache_key]

    defaults = _default_followups(tool)
    suggestions = []
    prompt = f"""
You are helping a user continue a research conversation.
Generate exactly 3 short, high-value follow-up questions.
Rules:
- Keep each question under 12 words.
- No numbering, no bullets, no extra text.
- Questions only.

User question: {question}
Assistant answer: {answer}
"""
    try:
        raw = st.session_state.agent.chat_llm.invoke(prompt)
        raw_text = raw.content if hasattr(raw, "content") else str(raw)
        suggestions = _normalize_suggestions(raw_text)
    except Exception:
        suggestions = []

    for default in defaults:
        if len(suggestions) >= 3:
            break
        if default.lower() not in {item.lower() for item in suggestions}:
            suggestions.append(default)

    suggestions = suggestions[:3]
    cache[cache_key] = suggestions
    return suggestions


def _default_insight(answer: str, tool: str) -> str:
    confidence = "High" if tool in {"RAG", "MIXED"} else "Medium"
    return (
        "### TL;DR\n"
        f"{answer[:300].strip()}...\n\n"
        "### Next Best Actions\n"
        "- Ask for source-specific verification.\n"
        "- Convert the answer into a step-by-step execution plan.\n"
        "- Identify any assumptions and validate them.\n\n"
        "### Risks / Unknowns\n"
        "- Some details may require fresher external validation.\n"
        "- Scope or constraints may still be underspecified.\n\n"
        "### Confidence\n"
        f"{confidence}\n"
    )


def generate_insight_card(question: str, answer: str, tool: str) -> str:
    cache_key = f"{tool}|{question[:180]}|{answer[:260]}"
    cache = st.session_state.insight_cache
    if cache_key in cache:
        return cache[cache_key]

    prompt = f"""
Create a concise executive insight card from this answer.
Use exactly this markdown structure:
### TL;DR
<2 sentences>

### Next Best Actions
- <action 1>
- <action 2>
- <action 3>

### Risks / Unknowns
- <risk 1>
- <risk 2>

### Confidence
<Low/Medium/High with one short reason>

Context:
User question: {question}
Answer tool mode: {tool}
Assistant answer: {answer}
"""
    try:
        raw = st.session_state.agent.chat_llm.invoke(prompt)
        insight = raw.content.strip() if hasattr(raw, "content") else str(raw).strip()
    except Exception:
        insight = _default_insight(answer, tool)

    if "### TL;DR" not in insight:
        insight = _default_insight(answer, tool)

    cache[cache_key] = insight
    return insight


def render_tool_pill(tool: str):
    st.markdown(f"<div class='tool-pill'>{TOOL_LABELS.get(tool, 'Routed via Other')}</div>", unsafe_allow_html=True)


def render_insight_card(insight_markdown: str, message_key: str):
    if not insight_markdown:
        return
    with st.container():
        st.markdown("<div class='insight-card'><div class='insight-title'>Insight Card</div></div>", unsafe_allow_html=True)
        st.markdown(insight_markdown)
        st.download_button(
            "Download Insight (.md)",
            data=insight_markdown,
            file_name=f"insight_{message_key}.md",
            mime="text/markdown",
            key=f"insight_download_{message_key}",
            use_container_width=True,
        )


def render_followup_buttons(suggestions: List[str], message_key: str):
    if not suggestions:
        return
    st.markdown("**Suggested follow-up questions**")
    cols = st.columns(len(suggestions))
    for i, suggestion in enumerate(suggestions):
        label = suggestion if len(suggestion) <= 48 else f"{suggestion[:45]}..."
        if cols[i].button(label, key=f"followup_{message_key}_{i}", use_container_width=True):
            st.session_state.pending_prompt = suggestion
            st.rerun()


def run_prompt(prompt: str, tavily_api_key: str, retrieval_k: int):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

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
                status.update(
                    label=f"Used Tool: {TOOL_LABELS.get(tool, 'Other').replace('Routed via ', '')}",
                    state="complete",
                    expanded=False,
                )

        st.markdown(response)

        render_tool_pill(tool)

        if tool in {"RAG", "MIXED"} and results:
            render_source_badges(results)
            model = st.session_state.memory_manager.get_embedding_model()
            for i, (doc, score) in enumerate(results):
                render_comparison_chart(
                    doc.page_content,
                    score,
                    model.embed_query(doc.page_content),
                    model.embed_query(prompt),
                    f"Source {i + 1}",
                )

        insight_markdown = ""
        if st.session_state.auto_insights:
            insight_markdown = generate_insight_card(prompt, response, tool)
            render_insight_card(insight_markdown, f"live_{len(st.session_state.messages)}")

        suggestions = generate_followup_suggestions(prompt, response, tool)
        render_followup_buttons(suggestions, f"live_{len(st.session_state.messages)}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "tool": tool,
            "insight": insight_markdown,
            "suggestions": suggestions,
        }
    )


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
if "memory_manager" not in st.session_state:
    st.session_state.memory_manager = MemoryManager()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_state" not in st.session_state:
    st.session_state.processed_state = None
if "suggestion_cache" not in st.session_state:
    st.session_state.suggestion_cache = {}
if "insight_cache" not in st.session_state:
    st.session_state.insight_cache = {}
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "auto_insights" not in st.session_state:
    st.session_state.auto_insights = True

# Session tools in sidebar
st.sidebar.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
st.sidebar.markdown("### Session")
st.sidebar.caption(f"Messages: {len(st.session_state.messages)}")
st.session_state.auto_insights = st.sidebar.toggle(
    "Auto Insight Cards",
    value=st.session_state.auto_insights,
    help="Generate an executive insight card for each assistant response.",
)
st.sidebar.download_button(
    label="Download Chat (.md)",
    data=build_chat_markdown(st.session_state.messages),
    file_name=f"agentic_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
    mime="text/markdown",
    use_container_width=True,
    disabled=not st.session_state.messages,
)

# 3. Agent initialization with key-change support
force_reinit = False
if "agent" in st.session_state:
    current_agent_has_tavily = st.session_state.agent.tavily is not None
    user_provided_tavily = bool(tavily_api_key)
    if user_provided_tavily and not current_agent_has_tavily:
        force_reinit = True

if "agent" not in st.session_state or force_reinit:
    if force_reinit:
        st.toast("Updating Agent with new keys.")
    st.session_state.agent = AgentBrain(groq_api_key, tavily_api_key, st.session_state.memory_manager)

# 4. Smart ingestion logic
if uploaded_files:
    current_state = {"files": frozenset({f.name for f in uploaded_files}), "chunk_size": chunk_size}

    if st.session_state.processed_state != current_state:
        st.sidebar.warning("Pending changes")
        if st.sidebar.button("Process Files", type="primary", use_container_width=True):
            processor = DocumentProcessor(chunk_size=chunk_size)
            with st.status("Building Knowledge Base...", expanded=True) as status:
                splits = []
                for file_obj in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file_obj.getbuffer())
                        tmp_path = tmp.name
                    try:
                        splits.extend(processor.process_files([tmp_path]))
                    finally:
                        os.remove(tmp_path)

                st.session_state.memory_manager.clear()
                st.session_state.memory_manager.ingest_docs(splits, status_container=status)
                st.session_state.processed_state = current_state
                status.update(label="Ready", state="complete", expanded=False)
                st.rerun()
    else:
        st.sidebar.success("System ready")

# 5. UI Stats
if st.session_state.memory_manager.vector_store:
    render_sidebar_stats(st.session_state.memory_manager.vector_store.index.ntotal)

# 6. Chat UI
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

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🧠"):
        st.markdown(msg["content"])
        if msg.get("role") == "assistant":
            if msg.get("tool"):
                render_tool_pill(msg["tool"])
            render_insight_card(msg.get("insight", ""), f"history_{idx}")
            render_followup_buttons(msg.get("suggestions", []), f"history_{idx}")

typed_prompt = st.chat_input("Ask anything...")
queued_prompt = st.session_state.pending_prompt
st.session_state.pending_prompt = None
active_prompt = queued_prompt or typed_prompt

if active_prompt:
    run_prompt(active_prompt, tavily_api_key, retrieval_k)
