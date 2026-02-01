from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tavily import TavilyClient

class AgentBrain:
    def __init__(self, groq_api_key, tavily_api_key, memory_manager):
        self.memory = memory_manager
        
        # 1. Initialize Optional Web Search
        if tavily_api_key:
            self.tavily = TavilyClient(api_key=tavily_api_key)
        else:
            self.tavily = None
        
        # 2. Initialize LLMs
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=groq_api_key)
        self.router_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=groq_api_key)

        # 3. CONTEXTUALIZER
        self.context_prompt = ChatPromptTemplate.from_template("""
        You are a conversation helper. 
        If the user says a simple greeting (hi, hello, hey), RETURN IT AS IS.
        Otherwise, rewrite the question to be standalone using the history.
        Chat History: {chat_history}
        Question: {question}
        Standalone Question:
        """)

        # 4. ROUTER
        self.router_prompt = ChatPromptTemplate.from_template("""
        Classify the user intent.
        1. RAG: Needs specific facts from uploaded documents.
        2. WEB: Needs real-time info, news, or verification.
        3. CHAT: Greetings, compliments, or casual talk.

        Query: {question}
        Return ONLY: RAG, WEB, or CHAT.
        """)

        # 5. PROMPTS
        self.rag_prompt = ChatPromptTemplate.from_template("""
        Answer using ONLY the Context. 
        Context: {context}
        Question: {question}
        """)

        self.web_prompt = ChatPromptTemplate.from_template("""
        Answer using the Web Search Results.
        Web Results: {context}
        Question: {question}
        """)

        self.chat_prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Reply naturally.
        User: {question}
        """)

    def ask(self, query, chat_history=[], score_threshold=0.3):
        # A. Contextualize
        history_text = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history[-3:]])
        chain = self.context_prompt | self.llm | StrOutputParser()
        refined_query = chain.invoke({"chat_history": history_text, "question": query}).strip()
        print(f"🧠 Thinking: {refined_query}")

        # B. Decide
        router_chain = self.router_prompt | self.router_llm | StrOutputParser()
        decision = router_chain.invoke({"question": refined_query}).strip().upper()
        print(f"🤖 Decision: {decision}")

        # --- SMART FALLBACK (The Fix) ---
        # If Router wants WEB, but we have no Key, force it to check RAG.
        if "WEB" in decision and not self.tavily:
            print("⚠️ Web requested but no Key. Falling back to RAG.")
            decision = "RAG"
        # --------------------------------

        # PATH: RAG
        if "RAG" in decision:
            results = self.memory.search(refined_query, k=3, score_threshold=score_threshold)
            
            # CASE: No documents found
            if not results:
                # 1. If we have a Key, try Web Search
                if self.tavily:
                    print("⚠️ RAG empty. Auto-switching to WEB.")
                    decision = "WEB"
                
                # 2. [NEW] If NO Key, fallback to General Knowledge (Llama-3 Brain)
                else:
                    print("⚠️ RAG empty. Switching to General Knowledge.")
                    chain = self.chat_prompt | self.llm | StrOutputParser()
                    response = chain.invoke({"question": refined_query})
                    # We add a disclaimer so the user knows this isn't from their docs
                    return f"ℹ️ **Note:** This information comes from general knowledge, not your documents.\n\n{response}", [], "CHAT"

            else:
                # Standard RAG Success
                context_text = "\n\n".join([doc.page_content for doc, score in results])
                chain = self.rag_prompt | self.llm | StrOutputParser()
                response = chain.invoke({"context": context_text, "question": refined_query})
                return response, results, "RAG"

        # PATH: WEB (Only runs if we have the key)
        if "WEB" in decision and self.tavily:
            print(f"🌍 Searching Web... (Tavily)")
            try:
                search_result = self.tavily.search(query=refined_query, search_depth="basic", max_results=5)
                context_str = "\n\n".join([f"Source: {res['title']}\nSnippet: {res['content']}" for res in search_result['results']])
                
                # Fake Docs for UI
                from langchain_core.documents import Document
                web_docs = [Document(page_content=res['content'], metadata={"source": res['title'], "page": "Web"}) for res in search_result['results'][:2]]

                chain = self.web_prompt | self.llm | StrOutputParser()
                response = chain.invoke({"context": context_str, "question": refined_query})
                return response, web_docs, "WEB"

            except Exception as e:
                return f"⚠️ Search Error: {str(e)}", [], "CHAT"

        # PATH: CHAT
        chain = self.chat_prompt | self.llm | StrOutputParser()
        response = chain.invoke({"question": refined_query})
        return response, [], "CHAT"