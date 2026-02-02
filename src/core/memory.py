import time
from langchain_community.vectorstores import FAISS
from src.core.processing import get_embeddings

class MemoryManager:
    def __init__(self, embedding_model=None):
        """
        Initializes the Memory Manager.
        Args:
            embedding_model: Optional. If None, loads the default model.
        """
        if embedding_model is None:
            self.embeddings = get_embeddings()
        else:
            self.embeddings = embedding_model
            
        self.vector_store = None

    def ingest_docs(self, splits, status_container=None):
        """
        Ingests documents in batches to show progress in the UI.
        """
        total_chunks = len(splits)
        # Process 100 chunks at a time for speed
        batch_size = 100
        
        if total_chunks == 0:
            return

        # --- THE FIX: Create the Progress Bar ONCE ---
        progress_bar = None
        if status_container:
            progress_bar = status_container.progress(0, text="Starting embedding...")

        # 1. Initialize with first batch
        first_batch = splits[:batch_size]
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(first_batch, self.embeddings)
        else:
            self.vector_store.add_documents(first_batch)
            
        # Update the existing bar
        if progress_bar:
            progress_bar.progress(min(batch_size / total_chunks, 1.0), 
                                text=f"Embedded {min(batch_size, total_chunks)}/{total_chunks} chunks...")

        # 2. Process remaining batches
        for i in range(batch_size, total_chunks, batch_size):
            batch = splits[i : i + batch_size]
            self.vector_store.add_documents(batch)
            
            # Update the SAME bar (don't create new ones)
            if progress_bar:
                progress = min((i + batch_size) / total_chunks, 1.0)
                progress_bar.progress(progress, 
                                    text=f"Embedded {min(i + batch_size, total_chunks)}/{total_chunks} chunks...")
                time.sleep(0.01)

    def get_embedding_model(self):
        """Returns the active embedding model."""
        return self.embeddings
        
    def search(self, query, k=5, score_threshold=None):
        """
        Searches for vectors similar to the query.
        """
        if not self.vector_store:
            return []

        # Always use similarity_search_with_score so we get (doc, score) tuples
        results_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
        
        if score_threshold is not None:
            # Filter results. Note: For FAISS L2 distance, Lower score = Better match.
            filtered_results = [
                (doc, score) for doc, score in results_with_scores 
                if score <= score_threshold
            ]
            return filtered_results
            
        return results_with_scores
    
    def clear(self):
        """Clears the vector store memory."""
        self.vector_store = None