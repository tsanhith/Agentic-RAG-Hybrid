from typing import List, Tuple, Optional, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from tavily import TavilyClient
import re
import sys

# Terminal Colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class AgentBrain:
    def __init__(self, groq_api_key: str, tavily_api_key: Optional[str], memory_manager: Any):
        self.memory = memory_manager
        
        # Tools
        self.tavily = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None
        
        # LLMs
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=groq_api_key)
        self.chat_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, api_key=groq_api_key)

        # --- PROMPTS ---
        self.context_prompt = ChatPromptTemplate.from_template("""
        Rewrite the "Latest Question" to be a standalone sentence. 
        Resolve pronouns (he, it, they) using History.
        
        History: {chat_history}
        Question: {question}
        Refined Question:
        """)

        self.search_query_prompt = ChatPromptTemplate.from_template("""
        Convert this into a targeted Google search query.
        Question: "{question}"
        Search Query:
        """)

        self.rag_prompt = ChatPromptTemplate.from_template("""
        Answer based ONLY on the Context. 
        If the answer is NOT in the context, output exactly: "MISSING_INFO".
        
        Context: {context}
        Question: {question}
        """)

        self.web_prompt = ChatPromptTemplate.from_template("""
        Answer using the Search Results. Mention dates/sources if available.
        Results: {context}
        Question: {question}
        """)

        self.rag_relevance_prompt = ChatPromptTemplate.from_template("""
        Decide if the provided Context is relevant to the Question.
        Respond with only "YES" or "NO".

        Context: {context}
        Question: {question}
        """)

        self.chat_prompt = ChatPromptTemplate.from_template("User: {question}\nAssistant:")

    def ask(self, query: str, chat_history: Optional[List[dict]] = None, k: int = 5, status_container: Any = None) -> Tuple[str, List[Document], str]:
        
        if chat_history is None:
            chat_history = []

        print(f"\n{Colors.HEADER}=== NEW QUERY ==={Colors.ENDC}")
        print(f"{Colors.BLUE}[Input]:{Colors.ENDC} {query}")

        try:
            # 1. Contextualize (Refine)
            if status_container: status_container.write("🧠 Refining context...")
            history_text = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history[-3:]])
            refined_query = (self.context_prompt | self.llm | StrOutputParser()).invoke(
                {"chat_history": history_text, "question": query}
            ).strip()
            print(f"{Colors.CYAN}[Refined]:{Colors.ENDC} {refined_query}")

            # ---------------------------------------------------------
            # STRATEGY: RAG FIRST -> FALLBACK TO WEB
            # ---------------------------------------------------------
            
            # Step 1: Check if it's purely conversational (Hi, Hello)
            # Simple heuristic: Short greetings usually don't need data.
            normalized_query = self._normalize_query(refined_query)
            if self._is_small_talk(normalized_query):
                print(f"{Colors.BOLD}[Strategy]:{Colors.ENDC} Small talk detected. Skipping DB.")
                return self._run_chat(refined_query, status_container), [], "CHAT"

            # Step 2: Try RAG (Always First)
            if status_container: status_container.write(f"📚 Searching Knowledge Base...")
            
            # We use a slightly looser threshold (1.5) to catch 'maybe' relevant docs
            results = self.memory.search(refined_query, k=k, score_threshold=1.5)
            
            if self._should_use_rag(results) and self._is_rag_relevant(refined_query, results):
                print(f"{Colors.GREEN}[RAG]:{Colors.ENDC} Found {len(results)} potential docs.")
                context_text = "\n\n".join([doc.page_content for doc, score in results])
                
                # Ask LLM to validate content
                response = (self.rag_prompt | self.llm | StrOutputParser()).invoke(
                    {"context": context_text, "question": refined_query}
                )
                
                if "MISSING_INFO" not in response:
                    # SUCCESS: Answer found in DB
                    return response, results, "RAG"
                else:
                    print(f"{Colors.WARNING}[RAG]:{Colors.ENDC} Docs found but answer missing. Switching to WEB.")
            else:
                print(f"{Colors.WARNING}[RAG]:{Colors.ENDC} No relevant docs found. Switching to WEB.")

            # Step 3: Fallback to Web (if RAG failed)
            if self._requires_fresh_info(normalized_query):
                if self.tavily:
                    print(f"{Colors.BOLD}[Strategy]:{Colors.ENDC} Fresh info requested. Using Web.")
                    return self._run_web_search(refined_query, status_container)
                print(f"{Colors.FAIL}[Fallback]:{Colors.ENDC} Fresh info requested but no Web Key. Using Chat.")
                return self._run_chat(refined_query, status_container, prefix="ℹ️ **Note:** Live data unavailable.\n\n"), [], "CHAT"

            if self.tavily:
                return self._run_web_search(refined_query, status_container)
            else:
                # Fallback to Chat if no Web Key
                print(f"{Colors.FAIL}[Fallback]:{Colors.ENDC} Web needed but no Key. Using Logic.")
                return self._run_chat(refined_query, status_container, prefix="ℹ️ **Note:** Not found in documents.\n\n"), [], "CHAT"

        except Exception as e:
             return f"⚠️ **System Error:** {str(e)}", [], "CHAT"

    # --- HELPER FUNCTIONS ---

    def _run_web_search(self, query, status_container):
        if status_container: status_container.write("🌍 Searching the Web...")
        print(f"{Colors.GREEN}[Web Search]:{Colors.ENDC} Initiating...")
        
        try:
            # Generate Keywords
            search_query = (self.search_query_prompt | self.llm | StrOutputParser()).invoke({"question": query}).strip()
            print(f"{Colors.BLUE}[Query]:{Colors.ENDC} {search_query}")
            
            results = self.tavily.search(query=search_query, search_depth="basic", max_results=5)
            if not results.get('results'): raise ValueError("No results.")
            
            context_str = "\n\n".join([f"Source: {r['title']}\nSnippet: {r['content']}" for r in results['results']])
            web_docs = [Document(page_content=r['content'], metadata={"source": r['title'], "page": "Web"}) for r in results['results'][:3]]
            
            response = (self.web_prompt | self.llm | StrOutputParser()).invoke({"context": context_str, "question": query})
            return response, web_docs, "WEB"
            
        except Exception as e:
            print(f"{Colors.FAIL}[Web Error]:{Colors.ENDC} {e}")
            return self._run_chat(query, status_container, prefix=f"⚠️ **Web Failed:** {e}\n\n"), [], "CHAT"

    def _run_chat(self, query, status_container, prefix=""):
        if status_container: status_container.write("💬 Thinking...")
        response = (self.chat_prompt | self.chat_llm | StrOutputParser()).invoke({"question": query})
        return prefix + response

    def _normalize_query(self, query: str) -> str:
        cleaned = re.sub(r"[^\w\s]", "", query)
        return re.sub(r"\s+", " ", cleaned).strip().lower()

    def _is_small_talk(self, normalized_query: str) -> bool:
        if not normalized_query:
            return True

        conversational_patterns = [
            r"^(hi|hello|hey)( there)?$",
            r"^good (morning|afternoon|evening)$",
            r"^(thanks|thank you|thx)$",
            r"^how are you( doing)?$",
            r"^whats up$",
            r"^what is up$",
        ]

        return any(re.match(pattern, normalized_query) for pattern in conversational_patterns)

    def _requires_fresh_info(self, normalized_query: str) -> bool:
        freshness_markers = {
            "latest",
            "today",
            "current",
            "recent",
            "news",
            "breaking",
            "right now",
            "as of",
            "updated",
        }
        return any(marker in normalized_query for marker in freshness_markers)

    def _is_rag_relevant(self, query: str, results: List[Tuple[Document, float]]) -> bool:
        context_text = "\n\n".join([doc.page_content for doc, _score in results])
        relevance = (self.rag_relevance_prompt | self.llm | StrOutputParser()).invoke(
            {"context": context_text, "question": query}
        ).strip().upper()
        return relevance == "YES"

    def _should_use_rag(self, results: List[Tuple[Document, float]], min_context_chars: int = 200, min_doc_chars: int = 50) -> bool:
        if not results:
            return False

        total_chars = sum(len(doc.page_content or "") for doc, _score in results)
        if total_chars < min_context_chars:
            return False

        if all(len(doc.page_content or "") < min_doc_chars for doc, _score in results):
            return False

        return True
