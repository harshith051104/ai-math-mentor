import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any
from aimath.config.settings import Settings as AppSettings

class VectorStore:
    """
    Manages RAG operations using ChromaDB.
    """
    def __init__(self, collection_name: str = "math_knowledge_base"):
        self.client = chromadb.PersistentClient(path=str(AppSettings.CHROMA_DB_DIR))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """
        Add documents to the vector store.
        ChromaDB handles embedding automatically if no embedding function is provided (uses DefaultEmbeddingFunction).
        """
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, query_text: str, n_results: int = 3) -> Dict[str, Any]:
        """
        Retrieve relevant context for a query.
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results

    def get_count(self) -> int:
        return self.collection.count()
