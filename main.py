import os
import tempfile
import io
import csv
from datetime import datetime, timezone
from typing import List

import streamlit as st

from src.core.agent import AgentBrain
from src.core.memory import MemoryManager
from src.core.processing import DocumentProcessor
from src.ui.layout import setup_page
from src.ui.visuals import (
    normalize_source_results,
    render_comparison_chart,
    render_sidebar_stats,
    render_source_badges,
)


TOOL_LABELS = {
    "RAG": "Routed via Documents",
    "WEB": "Routed via Internet",
    "CHAT": "Routed via Logic",
    "MIXED": "Routed via Mixed Mode",
}
MAX_BATCH_QUESTIONS = 8


def make_message_id(prefix: str = "msg") -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"


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


def build_favorites_markdown(favorites):
    if not favorites:
        return "# Favorites Library\n\n_No saved answers yet._\n"

    lines = [
        "# Favorites Library",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC_",
        "",
    ]
    for idx, item in enumerate(favorites, start=1):
        lines.append(f"## {idx}. {item.get('question') or 'Saved Answer'}")
        lines.append("")
        lines.append(f"**Saved At:** {item.get('saved_at', '-')}")
        lines.append(f"**Tool Route:** {TOOL_LABELS.get(item.get('tool', ''), item.get('tool', '-'))}")
        lines.append("")
        lines.append(item.get("answer", "").strip())
        insight = item.get("insight")
        if insight:
            lines.append("")
            lines.append("### Insight Card")
            lines.append(insight.strip())
        lines.append("")
    return "\n".join(lines)


def build_batch_markdown(results):
    if not results:
        return "# Batch Q&A Report\n\n_No batch results yet._\n"
    lines = [
        "# Batch Q&A Report",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC_",
        "",
    ]
    for idx, item in enumerate(results, start=1):
        lines.append(f"## {idx}. {item['question']}")
        lines.append("")
        lines.append(f"**Tool Route:** {TOOL_LABELS.get(item['tool'], item['tool'])}")
        lines.append("")
        lines.append(item["answer"])
        insight = item.get("insight")
        if insight:
            lines.append("")
            lines.append("### Insight Card")
            lines.append(insight)
        lines.append("")
    return "\n".join(lines)


def build_batch_csv(results):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["question", "tool", "answer", "insight"])
    writer.writeheader()
    for item in results:
        writer.writerow(
            {
                "question": item.get("question", ""),
                "tool": TOOL_LABELS.get(item.get("tool", ""), item.get("tool", "")),
                "answer": item.get("answer", ""),
                "insight": item.get("insight", ""),
            }
        )
    return buffer.getvalue()


def parse_batch_questions(raw_input: str) -> List[str]:
    questions = []
    seen = set()
    for raw_line in raw_input.splitlines():
        line = raw_line.strip().strip("-*")
        if not line:
            continue
        lowered = line.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        questions.append(line)
        if len(questions) >= MAX_BATCH_QUESTIONS:
            break
    return questions


def save_to_favorites(question: str, answer: str, tool: str, insight: str, source_id: str) -> bool:
    favorites = st.session_state.favorites
    if any(item.get("source_id") == source_id for item in favorites):
        return False

    favorites.append(
        {
            "source_id": source_id,
            "question": (question or "Saved Answer").strip(),
            "answer": (answer or "").strip(),
            "tool": tool or "CHAT",
            "insight": (insight or "").strip(),
            "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        }
    )
    return True


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


def run_batch_questions(questions: List[str], tavily_api_key: str, retrieval_k: int, append_to_chat: bool):
    batch_results = []
    with st.status("Running Batch Q&A...", expanded=True) as status:
        for idx, question in enumerate(questions, start=1):
            status.write(f"Processing {idx}/{len(questions)}: {question}")
            if not st.session_state.memory_manager.vector_store and not tavily_api_key:
                answer = "I do not have knowledge access yet. Upload PDFs or add Tavily for web search."
                tool = "CHAT"
            else:
                answer, _, tool = st.session_state.agent.ask(
                    question,
                    chat_history=[],
                    k=retrieval_k,
                    status_container=None,
                )

            insight = generate_insight_card(question, answer, tool) if st.session_state.auto_insights else ""
            suggestions = generate_followup_suggestions(question, answer, tool)
            result = {
                "question": question,
                "answer": answer,
                "tool": tool,
                "insight": insight,
                "suggestions": suggestions,
            }
            batch_results.append(result)

            if append_to_chat:
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.messages.append(
                    {
                        "message_id": make_message_id("batchchat"),
                        "question": question,
                        "role": "assistant",
                        "content": answer,
                        "tool": tool,
                        "insight": insight,
                        "suggestions": suggestions,
                    }
                )

        status.update(label=f"Batch complete: {len(batch_results)} answers", state="complete", expanded=False)

    st.session_state.batch_results = batch_results


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
            normalized_results = normalize_source_results(results)
            render_source_badges(normalized_results)
            model = st.session_state.memory_manager.get_embedding_model()
            scored_results = [(doc, score) for doc, score in normalized_results if score is not None]
            for i, (doc, score) in enumerate(scored_results):
                render_comparison_chart(
                    doc.page_content,
                    score,
                    model.embed_query(doc.page_content),
                    model.embed_query(prompt),
                    f"Source {i + 1}",
                )
            if normalized_results and not scored_results:
                st.caption("Some sources are web/context snippets, so similarity charts are unavailable for this response.")

        insight_markdown = ""
        if st.session_state.auto_insights:
            insight_markdown = generate_insight_card(prompt, response, tool)
            render_insight_card(insight_markdown, f"live_{len(st.session_state.messages)}")

        suggestions = generate_followup_suggestions(prompt, response, tool)
        render_followup_buttons(suggestions, f"live_{len(st.session_state.messages)}")

    st.session_state.messages.append(
        {
            "message_id": make_message_id("chat"),
            "question": prompt,
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
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []
if "batch_questions_input" not in st.session_state:
    st.session_state.batch_questions_input = ""
if "batch_append_to_chat" not in st.session_state:
    st.session_state.batch_append_to_chat = False
if "favorites" not in st.session_state:
    st.session_state.favorites = []

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

st.sidebar.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
st.sidebar.markdown("### Favorites Library")
st.sidebar.caption(f"Saved answers: {len(st.session_state.favorites)}")
if st.session_state.favorites:
    st.sidebar.download_button(
        label="Download Favorites (.md)",
        data=build_favorites_markdown(st.session_state.favorites),
        file_name=f"favorites_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True,
    )
    if st.sidebar.button("Clear Favorites", use_container_width=True):
        st.session_state.favorites = []
        st.toast("Favorites cleared.")
        st.rerun()
    with st.sidebar.expander("View Saved", expanded=False):
        for i, item in enumerate(reversed(st.session_state.favorites[-10:]), start=1):
            st.markdown(f"**{i}. {item.get('question', 'Saved Answer')[:70]}**")
            st.caption(f"{item.get('saved_at', '')} | {TOOL_LABELS.get(item.get('tool', ''), item.get('tool', 'CHAT'))}")
else:
    st.sidebar.caption("Save answers from chat or batch results.")

st.sidebar.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
st.sidebar.markdown("### Batch Q&A Lab")
st.sidebar.caption("Run multiple questions at once and export a structured report.")
st.sidebar.text_area(
    "Questions (one per line)",
    key="batch_questions_input",
    height=170,
    placeholder="What are the top risks in this report?\nSummarize section 2 in bullet points\nWhat should we do next week?",
)
st.sidebar.checkbox("Append batch results to chat", key="batch_append_to_chat")
st.sidebar.caption(f"Max {MAX_BATCH_QUESTIONS} questions per run.")
if st.sidebar.button("Run Batch Q&A", type="primary", use_container_width=True):
    parsed_questions = parse_batch_questions(st.session_state.batch_questions_input)
    if not parsed_questions:
        st.warning("Add at least one valid question to run batch mode.")
    else:
        run_batch_questions(
            parsed_questions,
            tavily_api_key=tavily_api_key,
            retrieval_k=retrieval_k,
            append_to_chat=st.session_state.batch_append_to_chat,
        )
        st.toast(f"Batch run complete for {len(parsed_questions)} questions.")
        st.rerun()

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

if st.session_state.batch_results:
    st.markdown("### Batch Q&A Results")
    b1, b2, b3 = st.columns(3)
    b1.metric("Questions", len(st.session_state.batch_results))
    rag_count = sum(1 for item in st.session_state.batch_results if item.get("tool") == "RAG")
    web_count = sum(1 for item in st.session_state.batch_results if item.get("tool") == "WEB")
    b2.metric("Document-Routed", rag_count)
    b3.metric("Web-Routed", web_count)

    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "Download Batch Report (.md)",
        data=build_batch_markdown(st.session_state.batch_results),
        file_name=f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True,
    )
    dl2.download_button(
        "Download Batch Data (.csv)",
        data=build_batch_csv(st.session_state.batch_results),
        file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    for i, item in enumerate(st.session_state.batch_results, start=1):
        with st.expander(f"{i}. {item['question']}", expanded=False):
            st.markdown(item["answer"])
            render_tool_pill(item["tool"])
            batch_source_id = item.get("source_id", f"batch_{i}_{abs(hash(item['question'] + item['answer']))}")
            if st.button("Save to Favorites", key=f"save_batch_{batch_source_id}", use_container_width=True):
                saved = save_to_favorites(
                    question=item.get("question", ""),
                    answer=item.get("answer", ""),
                    tool=item.get("tool", "CHAT"),
                    insight=item.get("insight", ""),
                    source_id=batch_source_id,
                )
                st.toast("Saved to favorites." if saved else "Already in favorites.")
                st.rerun()
            if item.get("insight"):
                render_insight_card(item["insight"], f"batch_{i}")
            render_followup_buttons(item.get("suggestions", []), f"batch_{i}")

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
            message_source_id = msg.get("message_id", f"history_{idx}_{abs(hash(msg.get('content', '')))}")
            if st.button("Save to Favorites", key=f"save_chat_{message_source_id}"):
                saved = save_to_favorites(
                    question=msg.get("question", ""),
                    answer=msg.get("content", ""),
                    tool=msg.get("tool", "CHAT"),
                    insight=msg.get("insight", ""),
                    source_id=message_source_id,
                )
                st.toast("Saved to favorites." if saved else "Already in favorites.")
                st.rerun()
            render_insight_card(msg.get("insight", ""), f"history_{idx}")
            render_followup_buttons(msg.get("suggestions", []), f"history_{idx}")

typed_prompt = st.chat_input("Ask anything...")
queued_prompt = st.session_state.pending_prompt
st.session_state.pending_prompt = None
active_prompt = queued_prompt or typed_prompt

if active_prompt:
    run_prompt(active_prompt, tavily_api_key, retrieval_k)
