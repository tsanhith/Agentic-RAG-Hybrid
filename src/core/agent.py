from typing import List, Tuple, Optional, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from tavily import TavilyClient
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
    MAX_SEARCH_QUERY_LENGTH = 400

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
        Keep it concise and under 350 characters.
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

            sub_questions = self._split_compound_query(refined_query)
            if len(sub_questions) > 1:
                return self._answer_compound(sub_questions, status_container, k=k)

            # ---------------------------------------------------------
            # STRATEGY: RAG FIRST -> FALLBACK TO WEB
            # ---------------------------------------------------------
            
            # Step 1: Check if it's purely conversational (Hi, Hello)
            # Simple heuristic: Short greetings usually don't need data.
            greetings = ["hi", "hello", "hey", "good morning", "thanks"]
            if refined_query.lower() in greetings:
                 print(f"{Colors.BOLD}[Strategy]:{Colors.ENDC} Greeting detected. Skipping DB.")
                 return self._run_chat(refined_query, status_container), [], "CHAT"

            return self._answer_single(refined_query, status_container, k=k)

        except Exception as e:
             return f"⚠️ **System Error:** {str(e)}", [], "CHAT"

    # --- HELPER FUNCTIONS ---

    def _run_web_search(self, query, status_container):
        if status_container: status_container.write("🌍 Searching the Web...")
        print(f"{Colors.GREEN}[Web Search]:{Colors.ENDC} Initiating...")
        
        try:
            # Generate Keywords
            search_query = (self.search_query_prompt | self.llm | StrOutputParser()).invoke({"question": query}).strip()
            search_query = self._truncate_search_query(search_query)
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

    def _truncate_search_query(self, query: str) -> str:
        if len(query) <= self.MAX_SEARCH_QUERY_LENGTH:
            return query
        truncated = query[: self.MAX_SEARCH_QUERY_LENGTH].rsplit(" ", 1)[0]
        return truncated if truncated else query[: self.MAX_SEARCH_QUERY_LENGTH]

    def _is_subjective_query(self, query: str) -> bool:
        lowered = query.lower()
        subjective_phrases = [
            "what do you think",
            "your opinion",
            "do you believe",
            "how do you feel",
            "is it bad",
            "is it good",
            "should i",
            "should we",
            "morally",
            "ethically",
            "religion",
            "religious",
            "worship",
            "idol",
            "molest",
            "harmful",
        ]
        return any(phrase in lowered for phrase in subjective_phrases)

    def _answer_single(self, query: str, status_container: Any, k: int) -> Tuple[str, List[Document], str]:
        if self._is_subjective_query(query):
            print(f"{Colors.BOLD}[Strategy]:{Colors.ENDC} Subjective query detected. Using CHAT.")
            return self._run_chat(query, status_container), [], "CHAT"

        if status_container:
            status_container.write("📚 Searching Knowledge Base...")

        results = self.memory.search(query, k=k, score_threshold=1.5)
        if not results and self.memory.vector_store:
            print(f"{Colors.WARNING}[RAG]:{Colors.ENDC} No matches under threshold. Falling back to top results.")
            results = self.memory.search(query, k=k, score_threshold=None)

        if results:
            print(f"{Colors.GREEN}[RAG]:{Colors.ENDC} Found {len(results)} potential docs.")
            context_text = "\n\n".join([doc.page_content for doc, score in results])

            response = (self.rag_prompt | self.llm | StrOutputParser()).invoke(
                {"context": context_text, "question": query}
            )

            if "MISSING_INFO" not in response:
                return response, results, "RAG"
            print(f"{Colors.WARNING}[RAG]:{Colors.ENDC} Docs found but answer missing. Switching to WEB.")
        else:
            print(f"{Colors.WARNING}[RAG]:{Colors.ENDC} No relevant docs found. Switching to WEB.")

        if self.tavily:
            return self._run_web_search(query, status_container)

        print(f"{Colors.FAIL}[Fallback]:{Colors.ENDC} Web needed but no Key. Using Logic.")
        return self._run_chat(query, status_container, prefix="ℹ️ **Note:** Not found in documents.\n\n"), [], "CHAT"

    def _answer_compound(self, sub_questions: List[str], status_container: Any, k: int) -> Tuple[str, List[Document], str]:
        responses = []
        all_docs: List[Document] = []
        modes: List[str] = []

        for sub_question in sub_questions:
            response, docs, mode = self._answer_single(sub_question, status_container, k=k)
            responses.append(f"**{sub_question.strip()}**\n{response}")
            all_docs.extend(docs)
            modes.append(mode)

        combined_mode = "RAG" if all(mode == "RAG" for mode in modes) else "MIXED"
        return "\n\n".join(responses), all_docs, combined_mode

    def _split_compound_query(self, query: str) -> List[str]:
        cleaned = query.strip()
        question_parts = [part.strip() for part in cleaned.split("?") if part.strip()]
        if len(question_parts) > 1:
            return [part + "?" for part in question_parts]

        lowered = cleaned.lower()
        if " and " in lowered:
            wh_words = ["what", "why", "how", "when", "where", "who", "which"]
            wh_count = sum(1 for word in wh_words if f" {word} " in f" {lowered} ")
            if wh_count >= 2:
                left, right = cleaned.split(" and ", 1)
                return [left.strip(), right.strip()]

        return [cleaned]
