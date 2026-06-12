import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()


# 1. Load a pdf document
from utils.import_loader import import_loader
MyPdfLoader = import_loader("pdf_loader")
loader = MyPdfLoader("LangChain/11. RAG/documents/tfm-david.pdf")
documents = loader.load()

# 2. Split Documents into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

docs = text_splitter.split_documents(documents)

# 3. Create embeddings
from langchain_huggingface import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Store embeddings in vector DataBase (Chroma)
from langchain_chroma import Chroma
vector_storage = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    collection_name="source_aware_rag"
)

# 5. Similarity Search
query = "What is this document mainly about?"
retrieved_docs = vector_storage.similarity_search(
    query=query,
    k=10
)

# 6. Build context + source
context_text = ""
sources = []

for i, doc in enumerate(retrieved_docs):
    page = doc.metadata.get("page", "N/A")
    source = doc.metadata.get("source", "PDF")

    context_text += f"\n Chunk {i+1}:\n{doc.page_content}. Page: {page}\n"
    sources.append({
        "chunk": i+1,
        "page": page,
        "source": source,
        "content": doc.page_content
    })

# 7. Prompt llm with context
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

llm = ChatGroq(model="llama-3.3-70b-versatile")
prompt = PromptTemplate(
    template="""
Answer the question only using the context below. Also provide the actual chunk used during the retrieval along with the page number of the document where i can find that chunk.

Context: {context}.

Question: {question}
""",
    input_variables=['context', 'question']
)
parser = StrOutputParser()

chain = prompt | llm | parser

answer = chain.invoke({
    "context": context_text,
    "question": query
})

print(answer)
print("\n\n")
print("----------------------------")
for src in sources:
    print(f"Chunk: {src['chunk']} | Page: {src['page']}")
    print(src['content'])
    print("----------------------------")