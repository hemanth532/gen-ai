from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from rank_bm25 import BM25Okapi

# Load the text document
loader = TextLoader("data/company_policy.txt")  
documents = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=10)
chunks = text_splitter.split_documents(documents)

# Create embeddings
embeddings = OllamaEmbeddings(model="llama3.2:1b")

# Create a FAISS vector store
vectorstore = FAISS.from_documents(chunks, embeddings)


# Example query
query = "What is RBI-AML-2024 Policy"
semantics_results = vectorstore.similarity_search(query,k=1)


#for result in results:
    #print(result.page_content)  

texts=[chunk.page_content for chunk in chunks]
tokenized_docs = [doc.split(" ") for doc in texts]
bm25 = BM25Okapi(tokenized_docs)


tokenized_query = query.split(" ")
bm25_results = bm25.get_top_n(tokenized_query, texts, n=1)
print("BM25 Result:")
print(bm25_results[0])

print("Semantic Search Result:")
print(semantics_results[0].page_content)