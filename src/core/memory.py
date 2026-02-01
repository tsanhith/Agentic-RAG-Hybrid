import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class MemoryManager:
    def __init__(self):
        print("🧠 Initializing Memory Manager...")
        
        # 1. The Encoder
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 2. The Database Path
        self.persist_dir = "./chromadb"
        
        # 3. Initialize the Database
        self.vector_store = Chroma(
            collection_name="agent_memory",
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir
        )

    def search(self, query, k=3, score_threshold=0.3):
        """
        Returns: List of tuples (Document, Score)
        """
        results = self.vector_store.similarity_search_with_relevance_scores(
            query, 
            k=k,
            score_threshold=score_threshold
        )
        return results

    def add_documents(self, chunks):
        self.vector_store.add_documents(chunks)

    def clear(self):
        """
        Wipes the memory and RE-INITIALIZES it so it's ready for new data.
        """
        # 1. Delete the collection from Chroma
        try:
            self.vector_store.delete_collection()
        except:
            pass # Use pass in case collection didn't exist
            
        # 2. RE-CREATE the empty database connection immediately
        # This fixes the "Collection not initialized" error
        self.vector_store = Chroma(
            collection_name="agent_memory",
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir
        )
        
    def get_embedding_model(self):
        return self.embeddings