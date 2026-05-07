import os
import tempfile
from pathlib import Path
from typing import Any, List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def load_documents(uploaded_files: List[Any]) -> List[Document]:
    """Load PDF and TXT files into LangChain Document objects."""
    documents: List[Document] = []

    for uploaded_file in uploaded_files:
        filename = getattr(uploaded_file, "name", None)
        if not filename:
            continue

        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = uploaded_file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(temp_path)
            else:
                loader = TextLoader(temp_path, encoding="utf-8")

            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source"] = filename
            documents.extend(loaded_docs)
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    return documents
