# gen-ai project

This repository contains a small Python AI project with a Streamlit chat interface, LLM integration, and retrieval-augmented generation (RAG) examples.

## Overview

- `chat_app.py`: Streamlit-based chat UI using the Ollama API and the Qwen model.
- `LLM/`: Local LLM integration examples.
  - `llmgemini.py`: Google Gemini GenAI example.
  - `llmollama.py`: Ollama-related code (not shown).
  - `llmopenai.py`: OpenAI integration example (not shown).
- `RAG/`: Retrieval-augmented generation examples.
  - `doc_RAG.py`: Document loader, chunking, embeddings, and conversational RAG flow using LangChain and Chroma.
  - `md_RAG.py`, `MultiRAG.py`, `pdf_RAG.py`, `txt_RAG.py`: Additional RAG scripts.
- `data/`: Sample input and prompt files for experiments.
- `requirements.txt`: Python dependencies needed to run the project.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

1. Create a virtual environment (if not already)
   python -m venv venv
2. Activate the environment
   venv\Scripts\activate
3. Install dependencies in the activated environment
   pip install -r requirements.txt
4. Run the chat app
   streamlit run chat_app.py
5. When finished, deactivate
   deactivate


## Run the chat app

```bash
streamlit run multiRAG_UI.py

streamlit run chat_app.py
```

## Run a RAG example

```bash
py RAG\pdf_RAG.py
```

Then open the local Streamlit URL shown in the terminal.

## Notes

- The project uses `ollama`, `openai`, and `google-generativeai` libraries for different LLM backends.
- The RAG example uses `langchain`, `chromadb`, and `unstructured` for document loading and vector search.
- Make sure your API keys and local LLM endpoints are configured before running the scripts.
