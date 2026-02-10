# 🧠 Agentic RAG: Intelligent Hybrid Search System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Llama 3](https://img.shields.io/badge/AI-Llama_3_70B-purple?style=for-the-badge)
![Groq](https://img.shields.io/badge/Inference-Groq-orange?style=for-the-badge)
![Tavily](https://img.shields.io/badge/Search-Tavily_AI-green?style=for-the-badge)

> **A next-generation Research Assistant that autonomously decides whether to read your documents or search the web.**

---

## 📸 System Architecture

### 🧩 Mermaid (Detailed + Easy-to-Understand)

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

```mermaid
flowchart TD
    UP[Upload PDFs] --> TMP[Temp File Save]
    TMP --> LOAD[PyMuPDFLoader]
    LOAD --> SPLIT[RecursiveCharacterTextSplitter]
    SPLIT --> BATCH[Batch Embedding]
    BATCH --> FAISS[(FAISS Vector Store)]
```

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

## 🚀 Overview

Standard RAG (Retrieval Augmented Generation) systems are limited—they only know what you upload. **Agentic RAG Hybrid** breaks this barrier by acting as an **autonomous agent**.

It possesses a "Brain" (Router) that evaluates every user query in real-time. If your uploaded PDFs contain the answer, it retrieves it with citation. If the query requires current events (e.g., _"Latest stock price"_) or public knowledge, it seamlessly switches to a **Live Web Search**.

### **Why is this better?**

- **Zero Hallucinations:** If the PDF lacks the answer, it doesn't make one up. It reports "Missing Info" and triggers a web search.
- **Context Aware:** It understands conversation history (e.g., "Who is he?" -> "Prabhas").
- **Self-Correcting:** If a search fails, it falls back to general logic or alternative strategies.

---

## ✨ Key Features

### 🧠 **1. Intelligent Routing**

The system classifies user intent into three streams:

- **📚 RAG Mode:** For specific questions about uploaded files (Legal docs, Manuals, Reports).
- **🌍 Web Mode:** For questions about live events, news, or public figures (via Tavily API).
- **💬 Chat Mode:** For greetings, coding tasks, or logic puzzles.

### 🛡️ **2. Smart Fallback & Self-Correction**

The "Agentic" loop ensures reliability:

1.  **Check Local DB:** Always prioritizes your private data.
2.  **Verify Content:** If the retrieved text is irrelevant, the Agent rejects it.
3.  **Auto-Switch:** Automatically triggers a Web Search if the local DB fails.

### ⚡ **3. High-Performance Tech Stack**

- **Inference:** Powered by **Groq** for lightning-fast Llama-3 responses.
- **Storage:** Uses **FAISS** (Facebook AI Similarity Search) for vector storage.
- **Ingestion:** **PyMuPDF** for 10x faster PDF parsing compared to standard loaders.

---

## 🛠️ Installation Guide

### Prerequisites

- Python 3.10+
- Groq API Key (Free tier available)
- Tavily API Key (Free tier available)

### 1. Clone the Repository

```bash
git clone [https://github.com/tsanhith/Agentic-RAG-Hybrid.git](https://github.com/tsanhith/Agentic-RAG-Hybrid.git)
cd Agentic-RAG-Hybrid
```
