import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()


# 1. Load a pdf document
from utils.import_loader import import_loader
MyPdfLoader = import_loader("pdf_loader")
loader = MyPdfLoader("11. RAG/documents/tfm-david.pdf")
documents = loader.load()

# 2. Split Documents into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
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
    collection_name="rag_collection"
)

# 5. Retriever
retriever = vector_storage.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10, 
        "lambda_mult": 0.5
    }
)

# 6. Initialize LLM
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.3-70b-versatile")

# 7. Prompt Template
from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate(
    template="""
You are an AI assistant. Use the following context to answer the question.
If the answer is not present in the context, say you don't know.

context: {context}.

question: {question}
""",
    input_variables=['context', 'question']
)

# 8. Output Parser
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()

# 9. Build a RAG Chain

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": lambda x: x
    } | prompt | llm | parser
)

# 10. Ask a question
query = "What is this document mainly about?"
response = rag_chain.invoke(query)

print(response)