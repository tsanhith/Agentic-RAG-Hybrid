# 🧠 Agentic RAG Hybrid — Detailed Architecture (Mermaid)

This document provides a **detailed, visually clear Mermaid architecture** for the Agentic RAG Hybrid system. The diagrams are based on the current code flow and module layout.

---

## 1) 🌈 High-Level Experience Map (User Journey + System Routing)

```mermaid
flowchart LR
    U([User]) --> UI[Streamlit UI]

    subgraph UI_Sidebar["UI Sidebar: Controls + Uploads"]
        UP[Upload PDFs]
        K[Groq / Tavily Keys]
        T[Retrieval Depth + Chunk Size]
    end

    UI --> UP
    UI --> K
    UI --> T

    UP --> DP[DocumentProcessor]
    DP --> SPLIT[Chunk + Split]
    SPLIT --> MEM[MemoryManager]
    MEM -->|Embed| EMB[HF Embeddings]
    MEM -->|Store| FAISS[(FAISS Vector Store)]

    UI --> Q[User Question]
    Q --> AGENT[AgentBrain]

    AGENT --> REFINE[Contextualize + Refine]
    REFINE --> GROQ[Groq Llama-3]

    AGENT --> RAG{RAG Search?}
    RAG -->|Yes| FAISS
    FAISS --> CTX[Top-k Docs]
    CTX --> GROQ

    AGENT -->|Missing Info| WEB{Web Needed?}
    WEB -->|Yes| TAVILY[Tavily Search]
    TAVILY --> GROQ
    WEB -->|No| CHAT[Chat Fallback]
    CHAT --> GROQ

    GROQ --> ANS[Response + Sources]
    ANS --> UI
    UI --> U
```

---

## 2) 🧩 Detailed Runtime Sequence (Query Execution)

```mermaid
sequenceDiagram
    participant User
    participant UI as Streamlit UI
    participant Agent as AgentBrain
    participant Memory as MemoryManager
    participant Vector as FAISS
    participant Groq as Groq Llama-3
    participant Tavily as Tavily API

    User->>UI: Ask question
    UI->>Agent: ask(query, history, k)
    Agent->>Groq: Refine & normalize query
    Groq-->>Agent: Refined question

    Agent->>Memory: search(query, k, threshold)
    Memory->>Vector: similarity_search_with_score
    Vector-->>Memory: docs + scores
    Memory-->>Agent: docs + scores

    alt RAG answer found
        Agent->>Groq: RAG prompt + context
        Groq-->>Agent: grounded answer
    else Missing info / no docs
        alt Web needed and key exists
            Agent->>Tavily: search(query)
            Tavily-->>Agent: top results
            Agent->>Groq: Web prompt + snippets
            Groq-->>Agent: web answer
        else No web key / not needed
            Agent->>Groq: Chat prompt
            Groq-->>Agent: general response
        end
    end

    Agent-->>UI: response + tool mode + citations
    UI-->>User: Render answer + source badges
```

---

## 3) 🧱 Component Map (Modules + External Services)

```mermaid
graph TD
    subgraph UI["Streamlit UI"]
        LAYOUT[layout.py]
        VISUALS[visuals.py]
        MAIN[main.py]
    end

    subgraph CORE["Core Logic"]
        AGENT[agent.py]
        MEMORY[memory.py]
        PROCESS[processing.py]
    end

    subgraph SERVICES["External Services"]
        GROQ[Groq Llama-3]
        TAVILY[Tavily Search]
        HF[HuggingFace Embeddings]
    end

    subgraph STORAGE["Vector Storage"]
        FAISS[(FAISS Index)]
    end

    MAIN --> LAYOUT
    MAIN --> VISUALS
    MAIN --> AGENT
    MAIN --> MEMORY
    MAIN --> PROCESS

    PROCESS --> HF
    MEMORY --> FAISS
    AGENT --> GROQ
    AGENT --> TAVILY
```

---

## 4) 🔁 Ingestion Pipeline (Document Processing)

```mermaid
flowchart TD
    UP[Upload PDFs] --> TMP[Temp File Save]
    TMP --> LOAD[PyMuPDFLoader]
    LOAD --> SPLIT[RecursiveCharacterTextSplitter]
    SPLIT --> BATCH[Batch Embedding]
    BATCH --> FAISS[(FAISS Vector Store)]
```

---

## 5) 🧠 Decision Policy (Routing Logic)

```mermaid
flowchart TD
    Q[Incoming Question] --> REFINE[Contextualize + refine]
    REFINE --> SUBJECTIVE{Subjective?}
    SUBJECTIVE -->|Yes| CHAT[Chat Mode]
    SUBJECTIVE -->|No| RAGTRY[Search vector DB]
    RAGTRY --> FOUND{Docs found?}
    FOUND -->|Yes| ANSWER[RAG Answer]
    FOUND -->|No| WEBNEED{Web indicators?}
    WEBNEED -->|Yes| WEB[Web Search]
    WEBNEED -->|No| CHAT
```

---

### ✅ Tips for Rendering
- Paste each Mermaid block into a Mermaid-compatible renderer (GitHub, Mermaid Live Editor, Obsidian, or VS Code with Mermaid support).
- Use **dark theme** for a more cinematic feel.
