import os
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings

CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
COLLECTION_NAME = "local_rag_collection"


def ensure_chroma_dir() -> None:
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)


class VectorStoreManager:
    def __init__(self):
        ensure_chroma_dir()

    def load_store(self, embeddings: Embeddings) -> Chroma | None:
        """Load an existing ChromaDB store if available."""
        try:
            if not os.path.isdir(CHROMA_DB_DIR):
                return None
            if not os.listdir(CHROMA_DB_DIR):
                return None
            return Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
            )
        except Exception:
            return None

    def create_store(self, documents: List[Document], embeddings: Embeddings) -> Chroma:
        """Create a new ChromaDB store and persist it locally."""
        store = Chroma.from_documents(
            documents,
            embeddings,
            persist_directory=CHROMA_DB_DIR,
            collection_name=COLLECTION_NAME,
        )
        store.persist()
        return store
