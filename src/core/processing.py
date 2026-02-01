from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """
        Args:
            chunk_size (int): Characters per chunk. Lower this for precise data (e.g., 500).
            chunk_overlap (int): Overlap to keep sentences intact (e.g., 100).
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_files(self, file_paths):
        raw_docs = []
        for path in file_paths:
            try:
                if path.endswith(".pdf"):
                    loader = PyPDFLoader(path)
                else:
                    loader = TextLoader(path)
                raw_docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading {path}: {e}")

        chunks = self.splitter.split_documents(raw_docs)
        return chunks