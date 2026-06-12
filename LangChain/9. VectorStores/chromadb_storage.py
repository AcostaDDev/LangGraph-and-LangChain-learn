from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

texts = [
    "Large language models are trained on massive datasets",
    "Large language models (llms) are particularly trained using transformers",
    "Chroma is a lightweight vector stored used in langchain",
    "Embeddings convert text into numerical representation"
]

vector_store = Chroma.from_texts(
    texts=texts,
    embedding=embedding_model,
    collection_name="langchain_chroma_demo"
)

query = "Tell me more about llms"
results = vector_store.similarity_search_with_score(query, k=2)

for r in results:
    print(r)