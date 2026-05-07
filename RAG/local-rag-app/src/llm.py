import os
import json
from dataclasses import dataclass
from typing import Any, Dict

import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
DEFAULT_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "512"))


@dataclass
class OllamaLLM:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS

    def generate(self, prompt: str) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=60,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Failed to connect to Ollama at {self.base_url}. Is Ollama running?") from e
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Ollama API error: {response.status_code} {response.text}") from e
        
        result = response.json()

        if isinstance(result, dict):
            if "results" in result:
                text_parts = []
                for item in result["results"]:
                    if isinstance(item, dict):
                        if "content" in item:
                            for content_chunk in item["content"]:
                                if isinstance(content_chunk, dict) and content_chunk.get("type") == "output_text":
                                    text_parts.append(content_chunk.get("text", ""))
                        elif "text" in item:
                            text_parts.append(item.get("text", ""))
                return "".join(text_parts).strip()
            if "text" in result:
                return result.get("text", "").strip()

        raise RuntimeError("Unexpected response format from Ollama generate endpoint")


def get_llm() -> OllamaLLM:
    """Return a local Ollama LLM client."""
    return OllamaLLM()
