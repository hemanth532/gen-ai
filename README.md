# 🚀 Gen AI & Agentic AI Projects Repository

A collection of Generative AI, RAG, Agentic AI, Multi-Agent, MCP, and Workflow Automation projects built using Python, LangChain, LangGraph, CrewAI, Ollama, OpenAI, and Vector Databases.

---

# 📚 Table of Contents

* Overview
* Prerequisites
* Installation
* Project Structure
* Core Concepts

  * LLM
  * Prompt Engineering
  * RAG
  * MCP
  * Agentic AI
  * Multi-Agent Systems
  * Vector Databases
  * Workflow Orchestration
* Running Projects
* Environment Variables
* Technology Stack
* Troubleshooting

---

# Overview

This repository contains implementations of:

* Simple RAG
* Agentic RAG
* Graph RAG
* Corrective RAG (CRAG)
* AI Agents
* Multi-Agent Systems
* Enterprise Workflows
* OCR Applications
* Voice AI Assistants
* Document Processing Pipelines

---

# Prerequisites

## Software

Install the following:

### Python

```bash
python --version
```

Recommended: Python 3.11+

### Git

```bash
git --version
```

### Docker (Optional)

```bash
docker --version
docker compose version
```

### Ollama (Optional)

```bash
ollama --version
```

Download:
https://ollama.com

---

# Installation

## Clone Repository

```bash
git clone <repository-url>
cd gen-ai-projects
```

## Create Virtual Environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Project Structure

```text
gen-ai/
│
├── Agentic Frameworks/
│   ├── autogen_demo.py
│   ├── crewai_demo.py
│   ├── googleADK_demo.py
│   └── Lang Framworks/
│       ├── langchain_demo.py
│       ├── langfuse_demo.py
│       ├── langgraph_demo.py
│       └── langsmith_demo.py
│
├── audioToText/
│   ├── audioToTextDemo.py
│   └── audioToTextDemo2.py
│
├── data/
│   ├── input.txt
│   ├── input2.txt
│   └── prompt.txt
│
├── documentation/
│   ├── Gen_AI_and_Agentic_AI_summary.txt
│   ├── GPT_Links.txt
│   └── n8n/
│       └── n8n Agent_Latest News Summarizer (1).json
│
├── enterpriseStandard/
│   ├── main.py
│   └── requirements.txt
│
├── imageToText/
│   ├── imageText.py
│   └── imageText2.py
│
├── LLM/
│   ├── llmgemini.py
│   ├── llmollama.py
│   └── llmopenai.py
│
├── multiModel/
│   ├── multiModel.py
│   └── multiModelRefactor.py
│
├── n8n/
│
├── RAG/
│   ├── doc_RAG.py
│   ├── md_RAG.py
│   ├── multiRAG_UI.py
│   ├── multiRAG.py
│   ├── pdf_RAG.py
│   ├── search_demo.py
│   ├── txt_RAG.py
│   ├── data/
│   │   ├── company_policy.txt
│   │   ├── rag_intro.md
│   │   └── rag_intro.txt
│   └── local-rag-app/
│       ├── app.py
│       ├── README.md
│       ├── requirements.txt
│       ├── chroma_db/
│       ├── multirag_db/
│       └── src/
│
├── chat_app.py
├── read_write_file.py
├── sample.py
├── requirements.txt
└── README.md
```

---

# Core Concepts

# Large Language Models (LLMs)

LLMs generate responses by predicting the next token.

Examples:

* GPT-4o
* GPT-5
* Claude
* Gemini
* Llama
* Qwen

Responsibilities:

* Text Generation
* Summarization
* Classification
* Reasoning

---

# Prompt Engineering

Framework:

## RTCCO

Role

Task

Context

Constraints

Output

Example:

```text
You are a senior HR assistant.

Task:
Answer employee leave questions.

Context:
Use company policy.

Constraints:
Do not make assumptions.

Output:
Provide concise responses.
```

---

# Retrieval-Augmented Generation (RAG)

RAG combines retrieval and generation.

## Flow

```text
User Query
      ↓
Embedding
      ↓
Vector Search
      ↓
Relevant Chunks
      ↓
LLM
      ↓
Answer
```

## Components

1. Document Loader
2. Chunking
3. Embeddings
4. Vector Database
5. Retriever
6. LLM

---

# Model Context Protocol (MCP)

MCP enables AI to interact with tools.

## Flow

```text
User Request
      ↓
LLM
      ↓
Tool Selection
      ↓
API Call
      ↓
Response
```

Examples:

* Leave Application
* Ticket Creation
* CRM Updates
* Database Queries

---

# Agentic AI

Agent = LLM + Memory + Tools + Reasoning

## Agent Lifecycle

```text
Think
  ↓
Decide
  ↓
Act
  ↓
Observe
```

Capabilities:

* Planning
* Decision Making
* Tool Usage
* Memory

---

# Multi-Agent Systems

Multiple agents collaborate together.

Example:

```text
Manager Agent
       ↓
Research Agent
       ↓
Search Tool

Manager Agent
       ↓
HR Agent
       ↓
Payroll Tool
```

Frameworks:

* LangGraph
* CrewAI
* AutoGen
* OpenAI Agents SDK

---

# Vector Databases

Store embeddings for semantic search.

Popular Options:

* ChromaDB
* Pinecone
* Weaviate
* Supabase Vector
* FAISS

---

# Workflow Orchestration

Coordinates multiple workflows.

## Example

```text
Customer Message
      ↓
Sentiment Analysis
      ↓
Priority Detection
      ↓
Knowledge Search
      ↓
AI Response
      ↓
CRM Update
```

Tools:

* LangGraph
* n8n
* Airflow
* Temporal

---

# Running Projects

## Simple RAG

### Start Ollama

```bash
ollama run llama3
```

### Run Application

```bash
python app.py
```

---

## Streamlit Application

```bash
streamlit run app.py
```

---

## FastAPI Application

```bash
uvicorn app:app --reload
```

---

## React Frontend

```bash
npm install
npm run dev
```

---

# Environment Variables

Create a .env file

```env
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
GOOGLE_API_KEY=your_key

MODEL_NAME=gpt-4o

VECTOR_DB=chroma
```

Load:

```python
from dotenv import load_dotenv

load_dotenv()
```

---

# Technology Stack

Core Platform

* Python 3.11+
* Streamlit
* FastAPI
* Ollama

AI & Agent Frameworks

* LangChain
* Autogen
* CrewAI
* LangGraph
* OpenAI SDK
* Google Generative AI

LLMs

* Ollama models
* OpenAI GPT models
* Google Gemini

RAG & Vector Search

* ChromaDB
* FAISS
* BM25
* LangChain document loaders and retrievers

OCR & Speech

* OpenAI Whisper
* faster-whisper
* pytesseract
* EasyOCR

Utilities

* python-docx
* unstructured
* Pillow
* dotenv

---

# Troubleshooting

## Ollama Not Found

```bash
ollama --version
```

Verify installation.

---

## Tesseract Error

Install:

```bash
tesseract --version
```

Add to PATH.

---

## FFmpeg Error

Install FFmpeg:

```bash
ffmpeg -version
```

Add ffmpeg/bin to PATH.

---

## OpenAI API Error

Verify:

```bash
echo %OPENAI_API_KEY%
```

or

```bash
printenv OPENAI_API_KEY
```

---

# Learning Roadmap

1. Python
2. APIs
3. Prompt Engineering
4. RAG
5. Vector Databases
6. LangChain
7. LangGraph
8. CrewAI
9. MCP
10. Agentic AI
11. Multi-Agent Systems
12. Enterprise AI Applications

---

⭐ This repository serves as a hands-on learning and implementation hub for Generative AI and Agentic AI projects.
