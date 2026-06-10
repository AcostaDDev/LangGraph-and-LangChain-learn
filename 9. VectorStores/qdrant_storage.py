from langchain_huggingface import HuggingFaceEmbeddings
# Importación moderna e independiente
from langchain_qdrant import QdrantVectorStore 
from dotenv import load_dotenv

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

cricket_texts = [
    "Cricket is a bat-and-ball sport played between two teams of eleven players on a large oval-shaped field.",
    "The batting team scores runs by hitting the ball and running between wickets while trying to avoid being dismissed.",
    "The bowling team tries to dismiss batters, restrict scoring opportunities, and maintain pressure throughout the match.",
    "Cricket is especially popular in countries like India, Australia, and England, where it has a long sporting tradition.",
    "International cricket is played in Test, One Day, and T20 formats, each offering a unique style of competition."
]

vector_store = QdrantVectorStore.from_texts(
    texts=cricket_texts,
    embedding=embedding_model,
    location=":memory:",
    collection_name="qdrant_collection_demo"
)

query = "How do I play cricket?"

results = vector_store.similarity_search_with_score(query, k=2)

for doc, score in results:
    print(f"Puntuación: {score}\nTexto: {doc.page_content}\n")