import os
import tempfile
from pathlib import Path

import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage

DATA_PATH = "data"
PERSIST_DIR = "multirag_db"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20
LLM_MODEL = "llama3.2:1b"
LLM_BASE_URL = "http://localhost:11434"
DEFAULT_TOP_K = 3
PROMPT_TEMPLATE = (
    "You are a helpful assistant that provides information about RAG. "
    "You must give the information only from the uploaded document and do not provide any information outside of the document.\n\n"
    "{history}\n\n"
    "Context:\n{context}\n\n"
    "Human:\n{input}"
)

llm = ChatOllama(model=LLM_MODEL, base_url=LLM_BASE_URL, temperature=0.1)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
embeddings = OllamaEmbeddings(model=LLM_MODEL)


def save_uploaded_file(uploaded_file):
    file_name = Path(uploaded_file.name).name
    suffix = Path(file_name).suffix
    temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(temp_fd, "wb") as temp_file:
        temp_file.write(uploaded_file.read())
    return temp_path


def load_documents_from_path(path):
    documents = []
    if not os.path.isdir(path):
        return documents

    for file_name in sorted(os.listdir(path)):
        file_path = os.path.join(path, file_name)
        lower_name = file_name.lower()

        if lower_name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif lower_name.endswith(".txt"):
            loader = TextLoader(file_path)
        elif lower_name.endswith((".docx", ".doc")):
            loader = UnstructuredWordDocumentLoader(file_path)
        elif lower_name.endswith(".md"):
            loader = UnstructuredFileLoader(file_path)
        else:
            print(f"Unsupported file format: {file_name}")
            continue

        documents.extend(loader.load())

    return documents


def build_vectorstore(documents):
    if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
        )

    if not documents:
        return None

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )


def format_history(history_messages):
    if not history_messages:
        return "No history yet."

    formatted = []
    for message in history_messages:
        role = "Human" if isinstance(message, HumanMessage) else "AI"
        formatted.append(f"{role}: {message.content}")
    return "\n".join(formatted)


def build_prompt(history, context, user_input):
    return PROMPT_TEMPLATE.format(history=history, context=context, input=user_input)


def create_retriever(store, top_k=DEFAULT_TOP_K):
    return store.as_retriever(search_kwargs={"k": top_k}) if store else None


def init_streamlit_state(retriever, vectorstore):
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = vectorstore
    if "retriever" not in st.session_state:
        st.session_state.retriever = retriever
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "top_k" not in st.session_state:
        st.session_state.top_k = DEFAULT_TOP_K
    if "status_message" not in st.session_state:
        st.session_state.status_message = "Ready"


def render_chat_history():
    if not st.session_state.chat_history:
        st.info("No conversation yet. Ask a question to start the chat.")
        return

    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(message.content)


def run_console_chat(retriever):
    if retriever is None:
        print("No documents are available for retrieval. Add files to the data folder or upload documents in the Streamlit UI.")
        return

    chat_history = []
    print("\nDocuments conversational RAG (type 'exit' to end the conversation)\n")

    while True:
        user_input = input("User: ")
        if user_input.strip().lower() == "exit":
            print("Conversation ended.")
            break

        retrieved_docs = retriever.invoke(user_input)
        context = "\n".join(doc.page_content for doc in retrieved_docs)
        print(f"Retrieved Context:\n{context}\n")

        history_text = format_history(chat_history)
        response = llm.invoke(build_prompt(history_text, context, user_input))

        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(response)

        print(f"Assistant: {response.content}\n")


def run_streamlit_app(retriever, vectorstore):
    init_streamlit_state(retriever, vectorstore)

    with st.sidebar:
        st.header("RAG Chat Settings")
        st.session_state.top_k = st.slider(
            "Retriever top k",
            min_value=1,
            max_value=10,
            value=st.session_state.top_k,
            help="Number of retrieved chunks to use as context for the LLM.",
        )
        st.markdown("---")
        st.markdown("### Status")
        st.info(st.session_state.status_message)
        st.markdown("---")
        st.markdown("### Help")
        st.write("Upload documents, then ask a question. The assistant answers from the indexed documents.")
        st.write("Click **Clear history** to start a new conversation.")

    st.title("Multi-Document RAG Chatbot")
    st.write("Upload documents, build an index, and chat with the LLM using only document context.")

    with st.expander("Upload documents and build index", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, DOCX, or MD files",
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            st.session_state.status_message = "Processing uploaded documents..."
            st.write(f"Processing {len(uploaded_files)} file(s)...")
            new_documents = []

            for uploaded_file in uploaded_files:
                file_path = save_uploaded_file(uploaded_file)
                lower_name = file_path.lower()

                if lower_name.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif lower_name.endswith(".txt"):
                    loader = TextLoader(file_path)
                elif lower_name.endswith((".docx", ".doc")):
                    loader = UnstructuredWordDocumentLoader(file_path)
                elif lower_name.endswith(".md"):
                    loader = UnstructuredFileLoader(file_path)
                else:
                    st.warning(f"Unsupported file format: {uploaded_file.name}")
                    continue

                new_documents.extend(loader.load())

            if new_documents:
                new_chunks = text_splitter.split_documents(new_documents)
                if st.session_state.vectorstore is None:
                    st.session_state.vectorstore = Chroma.from_documents(
                        documents=new_chunks,
                        embedding=embeddings,
                        persist_directory=PERSIST_DIR,
                    )
                else:
                    st.session_state.vectorstore.add_documents(new_chunks)
                    st.session_state.vectorstore.persist()

                st.session_state.retriever = create_retriever(
                    st.session_state.vectorstore,
                    top_k=st.session_state.top_k,
                )
                st.session_state.status_message = "Index created and ready."
                st.success("Documents processed and indexed successfully.")
            else:
                st.session_state.status_message = "Upload failed: no valid documents found."
                st.info("No valid documents were uploaded.")

    if st.session_state.retriever is None:
        st.warning("No document index is available yet. Add files to the data folder or upload documents.")
        return

    if st.button("Clear history"):
        st.session_state.chat_history = []
        st.session_state.status_message = "Conversation history cleared."

    st.markdown("---")
    st.subheader("Ask the assistant")
    question_col, action_col = st.columns([4, 1])
    user_query = question_col.text_input(
        "Your question",
        placeholder="Type a question about your documents...",
        key="user_query_input",
    )
    ask_button = action_col.button("Ask")

    if ask_button and user_query:
        retrieved_docs = st.session_state.retriever.invoke(user_query)
        context = "\n".join(doc.page_content for doc in retrieved_docs)

        with st.expander("Retrieved context", expanded=False):
            for index, doc in enumerate(retrieved_docs, start=1):
                st.write(f"**Chunk {index}:**")
                st.write(doc.page_content)

        history_text = format_history(st.session_state.chat_history)
        response = llm.invoke(build_prompt(history_text, context, user_query))

        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(response)
        st.session_state.status_message = "Assistant responded."

    st.markdown("---")
    st.subheader("Conversation")
    render_chat_history()


# Load documents and build the vector index.
documents = load_documents_from_path(DATA_PATH)
vectorstore = None
retriever = None
if documents:
    chunks = text_splitter.split_documents(documents)
    vectorstore = build_vectorstore(chunks)
else:
    vectorstore = build_vectorstore([])

if vectorstore is not None:
    retriever = create_retriever(vectorstore)

if __name__ == "__main__":
    if st.runtime.exists():
        run_streamlit_app(retriever, vectorstore)
    else:
        run_console_chat(retriever)
