from retrievers.wikipedia_retriever import MyWikipediaRetriever


retriever = MyWikipediaRetriever(top_k_results=2, lang="es")
query = "Inteligencia Artificial"

docs = retriever.invoke(query)

for i, doc in enumerate(docs):
    print("Content: \n", doc.metadata["summary"])