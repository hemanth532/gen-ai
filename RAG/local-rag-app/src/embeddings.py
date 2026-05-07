import os
import json
from dataclasses import dataclass
from typing import List, Any, Dict

import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "llama3.2:1b")
DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_EMBEDDING_TIMEOUT", "60"))


@dataclass
class OllamaEmbeddings:
    model: str = DEFAULT_EMBEDDING_MODEL
    base_url: str = DEFAULT_BASE_URL

    def _embed(self, text: str) -> List[float]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": text,
        }

        response = requests.post(
            f"{self.base_url}/v1/embeddings",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        result = response.json()

        if isinstance(result, dict) and "data" in result:
            first_item = result["data"][0]
            if isinstance(first_item, dict) and "embedding" in first_item:
                return first_item["embedding"]

        raise RuntimeError("Unexpected response format from Ollama embeddings endpoint")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)


def get_embeddings() -> OllamaEmbeddings:
    """Return a local Ollama embeddings client."""
    return OllamaEmbeddings()
