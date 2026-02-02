import os
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader  # FAST LOADER
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    """
    Returns the HuggingFace embedding model.
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

class DocumentProcessor:
    def __init__(self, chunk_size=1000):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200
        )

    def process_files(self, file_paths):
        """
        Loads multiple PDFs/Text files and returns a list of split chunks.
        """
        all_splits = []
        
        for path in file_paths:
            try:
                # Optimized: PyMuPDFLoader is significantly faster than PyPDFLoader
                loader = PyMuPDFLoader(path)
                docs = loader.load()
                splits = self.splitter.split_documents(docs)
                all_splits.extend(splits)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                
        return all_splits
        
    def process_file(self, uploaded_file):
        """
        Legacy method for single file processing.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
            
        try:
            return self.process_files([tmp_path])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)