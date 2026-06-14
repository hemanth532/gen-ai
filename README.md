# gen-ai project

This repository contains a collection of Python AI experiments and tools for Gen AI, Agentic AI, local LLM usage, audio/video/text processing, and retrieval-augmented generation (RAG).

## Getting Started for Beginners

1. Make sure Python is installed on your machine.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the example chat app:
   ```bash
   streamlit run chat_app.py
   ```
5. Explore RAG examples in the `RAG/` folder once the chat app is running.

## Current Project Structure

- `chat_app.py`: Streamlit chat interface using local or remote LLM backends.
- `Agentic Frameworks/`: Agent-oriented demos and frameworks.
  - `autogen_demo.py`
  - `crewai_demo.py`
  - `googleADK_demo.py`
  - `Lang Framworks/`: Examples for LangChain, LangFuse, LangGraph, and LangSmith.
- `audioToText/`: Audio transcription demos.
- `data/`: Sample input documents and prompt files.
- `documentation/`: Project documentation and training notes.
  - `Gen AI and Agentic AI.docx`: Core project documentation summarizing Gen AI, Agentic AI, RAG, MCP, and prompt engineering.
- `enterpriseStandard/`: Enterprise-standard AI examples.
- `imageToText/`: Image-to-text extraction demos.
- `LLM/`: Local and cloud LLM integration examples.
  - `llmgemini.py`, `llmollama.py`, `llmopenai.py`
- `multiModel/`: Multi-model orchestration examples.
- `n8n/`: Automation and workflow integration assets.
- `RAG/`: Retrieval-augmented generation and multi-RAG examples.
  - `doc_RAG.py`, `md_RAG.py`, `multiRAG.py`, `pdf_RAG.py`, `txt_RAG.py`, `search_demo.py`

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Recommended setup:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

Run the main chat app:

```bash
streamlit run chat_app.py
```

Run a RAG example:

```bash
python RAG\pdf_RAG.py
```

## Beginner-Friendly Documentation

This README includes a beginner-friendly overview of the project, drawn from the `documentation/Gen AI and Agentic AI.docx` training notes.

### What this project teaches

- Gen AI is a way to build systems that generate text, translate language, and answer questions.
- Agentic AI adds the ability to take actions, not just answer questions.
- This repo includes examples for chat apps, local model use, RAG, audio transcription, image-to-text, and agent workflows.

### Key beginner concepts

- LLM = Large Language Model. It is trained on large amounts of text and can generate responses.
- Tokens are the pieces of text sent to the model and returned by the model. They are also how usage is measured and billed.
- Context is the text the model uses to understand the current request.
- Prompt engineering means designing the inputs to the model so it gives good answers.

### Prompt engineering framework

Use `RTC CO` to build clear prompts:

- R = Role: define the assistant’s role.
- T = Task: explain what you want the model to do.
- C = Context: include any background information.
- C = Constraints: set rules or guardrails.
- O = Output: state the desired result or format.

### Common prompt types

- Zero-shot: ask directly without examples.
- Single-shot: provide one example.
- Multi-shot: provide multiple examples.
- Prompt chaining: split a task into smaller prompts.

### RAG vs MCP (beginner view)

- RAG (Retrieval-Augmented Generation): the model answers from documents or knowledge sources.
  - Use this when you need accurate, grounded answers from your own data.
- MCP (Model Context Protocol): the model chooses tools or actions.
  - Use this when you need the AI to perform real actions like calling APIs or writing files.

### Basic flow examples

- RAG flow:
  - Query → Vector database → Context → LLM → Answer
- MCP flow:
  - Query → LLM → Tool selection → API call → Result

### Why this repository is useful

- It shows how to build practical AI tools with local and cloud models.
- It teaches the difference between answer-focused systems (`RAG`) and action-focused systems (`MCP`).
- It includes examples for building chat apps, custom persona bots, and document-based search.

### Practical takeaways from the DOCX notes

- A custom GPT can be made by defining a persona, constraints, and tone. For example, a resume builder or meal planner chatbot.
- Guardrails help the AI stay on task and avoid unwanted outputs.
- Local models like `ollama` are useful when privacy or offline use is important.
- Useful AI workflows include OCR for PDFs/images, speech-to-text, and comparing results across models.

## Notes

- Configure API keys and local endpoints before running the scripts.
- `ollama` is used for local model execution and experimentation.
- `langchain`, `chromadb`, and `unstructured` are used for RAG examples.
- `documentation/Gen AI and Agentic AI.docx` remains the detailed reference file for this repository.
