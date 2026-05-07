import os
import streamlit as st

from src.chat_pipeline import ChatPipeline
from src.document_loader import load_documents
from src.embeddings import get_embeddings
from src.llm import get_llm
from src.retriever import Retriever
from src.utils import chunk_documents
from src.vectorstore import VectorStoreManager


PAGE_TITLE = "Local Ollama RAG"
PAGE_ICON = "🤖"
UPLOAD_TYPES = ["pdf", "txt"]
REFUSAL_MESSAGE = "I cannot answer that based on the uploaded documents."


def init_session_state() -> None:
    st.session_state.setdefault("uploaded_files", [])
    st.session_state.setdefault("documents", [])
    st.session_state.setdefault("vector_store", None)
    st.session_state.setdefault("retriever", None)
    st.session_state.setdefault("chat_pipeline", None)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("last_query", "")


def load_resources() -> None:
    if st.session_state.get("llm") is None:
        st.session_state["llm"] = get_llm()
    if st.session_state.get("embeddings") is None:
        st.session_state["embeddings"] = get_embeddings()
    if st.session_state.get("vector_manager") is None:
        st.session_state["vector_manager"] = VectorStoreManager()


def handle_file_upload(uploaded_files):
    if not uploaded_files:
        return

    documents = load_documents(uploaded_files)
    if not documents:
        return

    chunked_documents = chunk_documents(documents)
    vector_store = st.session_state["vector_manager"].load_store(st.session_state["embeddings"])

    if vector_store is None:
        vector_store = st.session_state["vector_manager"].create_store(chunked_documents, st.session_state["embeddings"])
    else:
        vector_store.add_documents(chunked_documents)

    retriever = Retriever(vector_store)
    chat_pipeline = ChatPipeline(st.session_state["llm"], retriever)

    st.session_state["uploaded_files"] = [uploaded_file.name for uploaded_file in uploaded_files]
    st.session_state["documents"] = documents
    st.session_state["vector_store"] = vector_store
    st.session_state["retriever"] = retriever
    st.session_state["chat_pipeline"] = chat_pipeline


def render_sidebar() -> None:
    with st.sidebar:
        st.title("Uploaded Documents")
        if st.session_state["uploaded_files"]:
            for filename in st.session_state["uploaded_files"]:
                st.markdown(f"- {filename}")
        else:
            st.markdown("No documents uploaded yet.")
        st.markdown("---")
        if st.button("Clear chat"):
            st.session_state["chat_history"] = []
            st.session_state["last_query"] = ""


def render_chat_history() -> None:
    if not st.session_state["chat_history"]:
        st.info("Upload documents and ask a question to start the conversation.")
        return

    for entry in st.session_state["chat_history"]:
        if entry["role"] == "user":
            st.markdown(f"**You:** {entry['message']}")
        else:
            st.markdown(f"**Assistant:** {entry['message']}")


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    st.title(PAGE_TITLE)
    st.write("A local retrieval-augmented generation app powered by Ollama and ChromaDB.")

    init_session_state()
    load_resources()

    with st.expander("Upload documents", expanded=True):
        uploaded_files = st.file_uploader(
            "Choose PDF or TXT files",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
        )
        if uploaded_files:
            handle_file_upload(uploaded_files)

    render_sidebar()

    with st.form(key="query_form", clear_on_submit=False):
        query = st.text_input("Ask a question", value=st.session_state.get("last_query", ""), key="query_input")
        submit_button = st.form_submit_button("Send")

    if submit_button and query:
        st.session_state["last_query"] = query
        if st.session_state["chat_pipeline"] is None:
            st.error("Please upload at least one document before asking a question.")
        else:
            with st.spinner("Retrieving relevant context and generating an answer..."):
                answer = st.session_state["chat_pipeline"].run(query)
                st.session_state["chat_history"].append({"role": "user", "message": query})
                st.session_state["chat_history"].append({"role": "assistant", "message": answer})

    render_chat_history()


if __name__ == "__main__":
    main()
