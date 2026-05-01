#from langchain_ollama import ChatOllama #deprecated langchain-community

## Start Local LLM
## PDF Reader
## PDF chunking
## Embeddings
## Vector DB
## LLM for processing the extracted text
## Convesation Loop

from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,AIMessage

# Start Local LLM
llm = ChatOllama(
    model= "llama3.2:1b",
    base_url="http://localhost:11434",
    temperature=0.1
)
print("Local LLM started")

# PDF Reader
loader = PyPDFLoader("data/rag_intro.pdf")
pages = loader.load()
print(f"Total Pages: {len(pages)}") 

# PDF chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = text_splitter.split_documents(pages)
print(f"Total Chunks: {len(chunks)}")

# Embeddings
embeddings = OllamaEmbeddings(model="llama3.2:1b")
# Vector DB
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings,persist_directory="./chroma_rag_ db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("Vector DB created and retriever set up")

# LLM for processing the extracted text
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that provides information about RAG.You must give the informtion only from the uploaded document and do not provide any information outside of the document."),
    ("system", "{history}"),
    ("system","context:{context}"),
    ("human", "{input}")
])
chat_history = []
print("\n PDF conversational RAG,(Type 'exit' to end the conversation) \n")

# Convesation Loop
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        print("Conversation ended.")
        break

    retrieved_docs = retriever.invoke(user_input)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    print(f"Retrieved Context:\n{context}\n")

    chat_history.append(HumanMessage(content=user_input))
    response = llm(prompt.format(history=chat_history, context=context, input=user_input))
    chat_history.append(AIMessage(content=response))
    print(f"Assistant: {response}\n")