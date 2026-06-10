# Maximal Margimal Relevance
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore 
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    Document(page_content="Langchain makes it easy to work with LLMs"),
    Document(page_content="Langchain helps developers build LLM applications easily"),
    Document(page_content="Chroma is a vector database optimized for LLM based search"),
    Document(page_content="Embeddings convert text into high-dimensional vectors"),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="OpenAI provides powerful embedding models"),
]

vector_store = QdrantVectorStore.from_documents(
    documents=documents,
    embedding=embedding_model,
    location=":memory:",
    collection_name="qdrant_collection_demo"
)

retriever1 = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "lambda_mult": 0.25 # 0 means full diversity, 1 means full relevance
    }
)

retriever2 = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "lambda_mult": 0.50 # 0 means full diversity, 1 means full relevance
    }
)

retriever3 = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "lambda_mult": 0.75 # 0 means full diversity, 1 means full relevance
    }
)

query = "What is langchain?"

result1 = retriever1.invoke(query)
result2 = retriever2.invoke(query)
result3 = retriever3.invoke(query)

print("0.25 LAMBDA MULT:")
for r1 in result1:
    print(r1.page_content)

print("---")

print("0.5 LAMBDA MULT:")
for r2 in result2:
    print(r2.page_content)

print("---")

print("0.75 LAMBDA MULT:")
for r3 in result3:
    print(r3.page_content)

print("---")
