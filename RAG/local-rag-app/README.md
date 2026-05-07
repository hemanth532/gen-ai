# Local RAG App

A fully local Retrieval-Augmented Generation (RAG) application using Ollama, ChromaDB, LangChain, and Streamlit.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Ollama locally from https://ollama.com/docs.
4. Pull the model:
   ```bash
   ollama pull llama3.2:1b
   ```
5. Run the Ollama server:
   ```bash
   ollama run llama3.2:1b
   ```

## Run the app

```bash
streamlit run app.py
```

## Features

- Multiple PDF and TXT file uploads
- Document chunking with overlap
- Local embeddings and ChromaDB persistence
- Strict RAG prompt enforcement
- Conversational chat memory
- Streamlit UI with sidebar document list

## Notes

- The app only answers from uploaded documents.
- If the answer is not contained in the retrieved context, it returns:
  `I cannot answer that based on the uploaded documents.`
- The ChromaDB database is persisted under `./chroma_db`.
