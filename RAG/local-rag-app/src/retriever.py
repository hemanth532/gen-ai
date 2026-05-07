from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

DEFAULT_TOP_K = 3


class Retriever:
    def __init__(self, vector_store: Chroma, top_k: int = DEFAULT_TOP_K):
        self.retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    def retrieve(self, query: str) -> List[Document]:
        """Return the top-k relevant document chunks for the query."""
        return self.retriever.get_relevant_documents(query)
